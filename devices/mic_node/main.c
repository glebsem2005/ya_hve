#include <stdint.h>
#include <string.h>

#define SAMPLE_RATE     16000
#define SAMPLE_BITS     16
#define CHUNK_SECONDS   1
#define CHUNK_SAMPLES   (SAMPLE_RATE * CHUNK_SECONDS)
#define CHUNK_BYTES     (CHUNK_SAMPLES * (SAMPLE_BITS / 8))
#define RADIO_FREQ_MHZ  433
#define NODE_ID         0x01

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

extern void     hal_i2s_read(int16_t *buf, uint32_t len);
extern void     hal_radio_send(const uint8_t *data, uint16_t len);
extern uint32_t hal_millis(void);
extern void     hal_sleep_ms(uint32_t ms);

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


int main(void) {
    static int16_t     audio_buf[CHUNK_SAMPLES];
    static RadioPacket pkt;

    while (1) {
        hal_i2s_read(audio_buf, CHUNK_SAMPLES);
        WavHeader hdr;
        build_wav_header(&hdr, CHUNK_BYTES);

        pkt.node_id      = NODE_ID;
        pkt.timestamp    = hal_millis();
        pkt.payload_size = sizeof(WavHeader) + CHUNK_BYTES;

        memcpy(pkt.payload, &hdr, sizeof(WavHeader));
        memcpy(pkt.payload + sizeof(WavHeader), audio_buf, CHUNK_BYTES);

        hal_radio_send((uint8_t *)&pkt,
                       sizeof(pkt.node_id) +
                       sizeof(pkt.timestamp) +
                       sizeof(pkt.payload_size) +
                       pkt.payload_size);
    }
}
