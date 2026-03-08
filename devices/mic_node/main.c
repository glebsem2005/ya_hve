/*
 * ForestGuard ESP32 Mic Node Firmware
 *
 * Duty cycle: wake -> listen 2s -> onset check -> record 3s (if triggered) -> LoRa send -> deep sleep 30s
 * Power: 3x AA lithium batteries (6 months with deep sleep). No solar panel.
 * Radio: LoRa 868 MHz, SF7, BW 125 kHz, 14 dBm TX power
 * Mic: ICS-43434 I2S MEMS microphone (24-bit in 32-bit frame)
 * Mesh: flood-based relay with hop_count < MAX_HOPS
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ---------- Audio configuration ---------- */

#define SAMPLE_RATE         16000
#define SAMPLE_BITS         16
#define CHUNK_SECONDS       1
#define CHUNK_SAMPLES       (SAMPLE_RATE * CHUNK_SECONDS)
#define CHUNK_BYTES         (CHUNK_SAMPLES * (SAMPLE_BITS / 8))
#define NODE_ID             0x01

/* ---------- I2S MEMS configuration (ICS-43434) ---------- */

#define I2S_SCK_PIN         26
#define I2S_WS_PIN          25
#define I2S_SD_PIN          22
#define I2S_SAMPLE_BITS     32      /* ICS-43434: 24-bit data in 32-bit frame */

typedef struct {
    uint8_t  port;
    uint8_t  sck_pin;
    uint8_t  ws_pin;
    uint8_t  sd_pin;
    uint32_t sample_rate;
    uint8_t  bits_per_sample;
} I2SConfig;

/* ---------- LoRa 868 MHz configuration ---------- */

#define RADIO_FREQ_MHZ      868
#define RADIO_SF             7       /* Spreading Factor */
#define RADIO_BW_KHZ        125
#define RADIO_TX_POWER_DBM  14      /* max for 868 MHz (EU regulation) */

typedef struct {
    uint16_t freq_mhz;
    uint8_t  sf;
    uint16_t bw_khz;
    uint8_t  tx_power_dbm;
} LoRaConfig;

/* ---------- Deep sleep / onset detection ---------- */

#define DEEP_SLEEP_DURATION_US  (30ULL * 1000000ULL)  /* 30 seconds */
#define LISTEN_WINDOW_MS        2000                   /* 2s listening window */
#define ENERGY_THRESHOLD        500                    /* RMS threshold for onset */
#define RECORD_DURATION_MS      3000                   /* 3s full recording after onset */
#define LISTEN_SAMPLES          (SAMPLE_RATE * LISTEN_WINDOW_MS / 1000)
#define RECORD_SAMPLES          (SAMPLE_RATE * RECORD_DURATION_MS / 1000)

/* ---------- Mesh networking ---------- */

#define MAX_HOPS            3       /* 3 hops x ~10 km = 30 km range */

/* ---------- Data structures ---------- */

typedef struct __attribute__((packed)) {
    char     riff[4];
    uint32_t file_size;
    char     wave[4];
    char     fmt[4];
    uint32_t fmt_size;
    uint16_t audio_format;
    uint16_t num_channels;
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample;
    char     data[4];
    uint32_t data_size;
} WavHeader;

typedef struct {
    uint8_t  node_id;
    uint32_t timestamp;
    uint16_t payload_size;
    uint8_t  payload[CHUNK_BYTES + sizeof(WavHeader)];
} RadioPacket;

typedef struct {
    uint8_t  packet_id[16];     /* UUID for deduplication */
    uint8_t  source_node;
    uint8_t  hop_count;
    uint8_t  max_hops;
    uint16_t payload_size;
    uint8_t  payload[CHUNK_BYTES + sizeof(WavHeader)];
} MeshPacket;

/* ---------- HAL declarations (platform-specific) ---------- */

extern void     hal_i2s_init(const I2SConfig *cfg);
extern void     hal_i2s_read(int16_t *buf, uint32_t len);
extern void     hal_radio_send(const uint8_t *data, uint16_t len);
extern uint32_t hal_millis(void);
extern void     hal_sleep_ms(uint32_t ms);

/* Deep sleep HAL */
extern void     hal_deep_sleep_us(uint64_t us);
extern void     hal_enable_wakeup_timer(uint64_t us);
extern uint8_t  hal_wakeup_reason(void);

/* LoRa HAL */
extern void     hal_lora_init(const LoRaConfig *cfg);
extern void     hal_lora_send(const uint8_t *data, uint16_t len);
extern int      hal_lora_recv(uint8_t *buf, uint16_t max_len, uint32_t timeout_ms);

/* ---------- WAV header ---------- */

static void build_wav_header(WavHeader *h, uint32_t data_bytes) {
    memcpy(h->riff, "RIFF", 4);
    h->file_size       = 36 + data_bytes;
    memcpy(h->wave, "WAVE", 4);
    memcpy(h->fmt,  "fmt ", 4);
    h->fmt_size        = 16;
    h->audio_format    = 1;
    h->num_channels    = 1;
    h->sample_rate     = SAMPLE_RATE;
    h->byte_rate       = SAMPLE_RATE * 1 * (SAMPLE_BITS / 8);
    h->block_align     = 1 * (SAMPLE_BITS / 8);
    h->bits_per_sample = SAMPLE_BITS;
    memcpy(h->data, "data", 4);
    h->data_size       = data_bytes;
}

/* ---------- Onset detection (firmware-level) ---------- */

static uint32_t compute_rms(const int16_t *buf, uint32_t len) {
    uint64_t sum_sq = 0;
    for (uint32_t i = 0; i < len; i++) {
        int32_t s = (int32_t)buf[i];
        sum_sq += (uint64_t)(s * s);
    }
    return (uint32_t)sqrt((double)sum_sq / (double)len);
}

/* ---------- Mesh relay ---------- */

static void try_mesh_relay(void) {
    /*
     * Check for incoming mesh packets from other nodes.
     * If hop_count < MAX_HOPS, increment and retransmit.
     */
    static MeshPacket rx_pkt;
    int rx_len = hal_lora_recv((uint8_t *)&rx_pkt, sizeof(rx_pkt), 100);

    if (rx_len <= 0)
        return;

    if (rx_pkt.hop_count >= rx_pkt.max_hops)
        return;

    /* Relay: increment hop count, retransmit */
    rx_pkt.hop_count++;
    hal_lora_send((uint8_t *)&rx_pkt,
                  sizeof(rx_pkt.packet_id) +
                  sizeof(rx_pkt.source_node) +
                  sizeof(rx_pkt.hop_count) +
                  sizeof(rx_pkt.max_hops) +
                  sizeof(rx_pkt.payload_size) +
                  rx_pkt.payload_size);
}

/* ---------- Main ---------- */

int main(void) {
    static int16_t     listen_buf[LISTEN_SAMPLES];
    static int16_t     record_buf[RECORD_SAMPLES];
    static RadioPacket pkt;

    /* Initialize I2S MEMS microphone */
    I2SConfig i2s_cfg = {
        .port            = 0,
        .sck_pin         = I2S_SCK_PIN,
        .ws_pin          = I2S_WS_PIN,
        .sd_pin          = I2S_SD_PIN,
        .sample_rate     = SAMPLE_RATE,
        .bits_per_sample = I2S_SAMPLE_BITS,
    };
    hal_i2s_init(&i2s_cfg);

    /* Initialize LoRa radio (868 MHz) */
    LoRaConfig lora_cfg = {
        .freq_mhz     = RADIO_FREQ_MHZ,
        .sf            = RADIO_SF,
        .bw_khz        = RADIO_BW_KHZ,
        .tx_power_dbm  = RADIO_TX_POWER_DBM,
    };
    hal_lora_init(&lora_cfg);

    while (1) {
        /*
         * Step 1: Listen window (2 seconds)
         * Compute RMS energy to detect sharp sounds.
         */
        hal_i2s_read(listen_buf, LISTEN_SAMPLES);
        uint32_t rms = compute_rms(listen_buf, LISTEN_SAMPLES);

        /*
         * Step 2: Onset check
         * Analog of Python onset detector (energy_ratio > 8.0),
         * but using absolute RMS threshold at firmware level.
         */
        if (rms < ENERGY_THRESHOLD) {
            /* Quiet -- check for mesh relay, then deep sleep */
            try_mesh_relay();
            hal_deep_sleep_us(DEEP_SLEEP_DURATION_US);
            /* ESP32 restarts from main() after deep sleep */
            continue;
        }

        /*
         * Step 3: Record full chunk (3 seconds) after onset trigger
         */
        hal_i2s_read(record_buf, RECORD_SAMPLES);

        /*
         * Step 4: Build WAV packet and send via LoRa
         */
        WavHeader hdr;
        build_wav_header(&hdr, CHUNK_BYTES);

        pkt.node_id      = NODE_ID;
        pkt.timestamp    = hal_millis();
        pkt.payload_size = sizeof(WavHeader) + CHUNK_BYTES;

        memcpy(pkt.payload, &hdr, sizeof(WavHeader));
        memcpy(pkt.payload + sizeof(WavHeader), record_buf, CHUNK_BYTES);

        hal_lora_send((uint8_t *)&pkt,
                      sizeof(pkt.node_id) +
                      sizeof(pkt.timestamp) +
                      sizeof(pkt.payload_size) +
                      pkt.payload_size);

        /*
         * Step 5: Check for mesh relay opportunities, then deep sleep
         */
        try_mesh_relay();
        hal_deep_sleep_us(DEEP_SLEEP_DURATION_US);
    }
}
