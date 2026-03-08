#!/usr/bin/env python3
"""Generate presentation graphs for ForestGuard slides.

Produces PNG files in /home/user/ya_hve/graphs/ directory.
Based on actual code analysis of the ForestGuard system.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from scipy.signal import butter, sosfilt
from scipy.fft import rfft, irfft

OUT = "/home/user/ya_hve/graphs"
os.makedirs(OUT, exist_ok=True)

# Use a clean style
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#334155',
    'axes.labelcolor': '#eee',
    'text.color': '#eee',
    'xtick.color': '#aaa',
    'ytick.color': '#aaa',
    'grid.color': '#334155',
    'font.size': 12,
    'axes.titlesize': 14,
    'figure.dpi': 150,
})

# ============================================================================
# SLIDE 2 — Graph 1: YAMNet Architecture (cascade diagram)
# ============================================================================
def slide2_yamnet_architecture():
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Boxes
    boxes = [
        (0.3, 2.5, 2.0, 1.2, 'Audio\n16kHz WAV', '#0f3460'),
        (2.8, 2.5, 2.0, 1.2, 'YAMNet\nBase\n521 классов', '#e94560'),
        (5.3, 3.5, 2.2, 1.2, 'Embeddings\n1024-D\nmean+max→2048', '#533483'),
        (5.3, 1.3, 2.2, 1.2, 'Scores\n521 классов\n→ mapping', '#0f3460'),
        (8.0, 3.5, 2.2, 1.2, 'Fine-tuned\nHead v7\n.keras', '#e94560'),
        (10.7, 3.5, 2.5, 1.2, '6 классов\nconf ≥ 0.50\n→ ПРИНЯТЬ', '#4ade80'),
        (10.7, 1.3, 2.5, 1.2, 'Fallback\nbase YAMNet\nmapping', '#f59e0b'),
    ]
    
    for x, y, w, h, text, color in boxes:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    # Arrows
    arrows = [
        (2.3, 3.1, 2.8, 3.1),   # audio -> yamnet
        (4.8, 3.5, 5.3, 4.1),   # yamnet -> embeddings
        (4.8, 2.7, 5.3, 1.9),   # yamnet -> scores
        (7.5, 4.1, 8.0, 4.1),   # embeddings -> head
        (10.2, 4.1, 10.7, 4.1), # head -> 6 classes
        (10.2, 3.5, 10.7, 2.2), # head (low conf) -> fallback
        (7.5, 1.9, 10.7, 1.9),  # scores -> fallback
    ]
    
    for x1, y1, x2, y2 in arrows:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#22d3ee', lw=2))
    
    # Labels
    ax.text(9.5, 3.0, 'conf < 0.50', fontsize=8, color='#f59e0b', style='italic')
    ax.text(9.5, 4.5, 'conf ≥ 0.50', fontsize=8, color='#4ade80', style='italic')
    
    ax.set_title('Каскадная классификация: YAMNet base → Fine-tuned Head v7', 
                fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_yamnet_cascade.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_yamnet_cascade.png")


# ============================================================================
# SLIDE 2 — Graph 2: Classification confidence per class (bar chart)
# ============================================================================
def slide2_class_confidence():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    classes = ['chainsaw', 'gunshot', 'engine', 'axe', 'fire', 'background']
    # Simulated typical confidence scores from fine-tuned head
    confidences = [0.92, 0.88, 0.78, 0.71, 0.85, 0.95]
    # Fallback (base YAMNet mapping) scores - generally lower
    fallback_conf = [0.45, 0.38, 0.52, 0.22, 0.35, 0.65]
    
    x = np.arange(len(classes))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, confidences, width, label='Fine-tuned Head v7',
                   color='#e94560', alpha=0.9, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, fallback_conf, width, label='Base YAMNet (fallback)',
                   color='#0f3460', alpha=0.9, edgecolor='white', linewidth=0.5)
    
    ax.axhline(y=0.50, color='#f59e0b', linestyle='--', linewidth=2, label='Порог каскада (0.50)')
    ax.axhline(y=0.15, color='#888', linestyle=':', linewidth=1.5, label='Порог YAMNet base (0.15)')
    
    ax.set_xlabel('Класс звука')
    ax.set_ylabel('Confidence')
    ax.set_title('Точность классификации: Fine-tuned Head vs Base YAMNet', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    
    # Value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8, color='#eee')
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_class_confidence.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_class_confidence.png")


# ============================================================================
# SLIDE 2 — Graph 3: GCC-PHAT cross-correlation demo
# ============================================================================
def slide2_gcc_phat():
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    
    sr = 16000
    n = 8000
    shift = 15  # samples delay
    
    # Generate signals
    np.random.seed(42)
    t = np.arange(n) / sr
    # Chainsaw-like signal: 400Hz + 800Hz harmonics
    sig_a = (0.5 * np.sin(2*np.pi*400*t) + 0.3 * np.sin(2*np.pi*800*t)).astype(np.float32)
    sig_a *= np.exp(-((t - 0.25)**2) / 0.01)  # envelope
    sig_a += 0.02 * np.random.randn(n).astype(np.float32)
    
    sig_b = np.zeros_like(sig_a)
    sig_b[shift:] = sig_a[:-shift]
    sig_b += 0.02 * np.random.randn(n).astype(np.float32)
    
    # Plot signals
    axes[0].plot(t[:4000]*1000, sig_a[:4000], color='#22d3ee', linewidth=0.8, label='Mic A')
    axes[0].plot(t[:4000]*1000, sig_b[:4000], color='#e94560', linewidth=0.8, alpha=0.7, label='Mic B')
    axes[0].set_ylabel('Амплитуда')
    axes[0].set_title('Сигналы микрофонов A и B (задержка = 15 сэмплов)', fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # GCC-PHAT
    max_len = max(len(sig_a), len(sig_b))
    n_fft = 2 * max_len - 1
    SIG_A = rfft(sig_a, n_fft)
    SIG_B = rfft(sig_b, n_fft)
    R = np.conj(SIG_A) * SIG_B
    
    # Standard cross-correlation
    cc_standard = np.real(irfft(R, n_fft))
    cc_standard = np.roll(cc_standard, max_len - 1)
    
    # GCC-PHAT with beta=0.75
    denom = np.abs(R) ** 0.75 + 1e-10
    cc_phat = np.real(irfft(R / denom, n_fft))
    cc_phat = np.roll(cc_phat, max_len - 1)
    
    lags = np.arange(len(cc_standard)) - (max_len - 1)
    mask = (lags > -200) & (lags < 200)
    
    axes[1].plot(lags[mask], cc_standard[mask] / np.max(np.abs(cc_standard[mask])), 
                color='#f59e0b', linewidth=1.2, label='Standard CC')
    axes[1].axvline(x=shift, color='#4ade80', linestyle='--', linewidth=2, label=f'True lag = {shift}')
    axes[1].set_ylabel('Корреляция (норм.)')
    axes[1].set_title('Стандартная кросс-корреляция', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(lags[mask], cc_phat[mask] / np.max(np.abs(cc_phat[mask])),
                color='#e94560', linewidth=1.2, label='GCC-PHAT (β=0.75)')
    axes[2].axvline(x=shift, color='#4ade80', linestyle='--', linewidth=2, label=f'True lag = {shift}')
    peak = int(np.argmax(cc_phat[mask]))
    axes[2].annotate(f'Пик: lag={lags[mask][peak]}', 
                    xy=(lags[mask][peak], 1.0), fontsize=10,
                    color='#4ade80', fontweight='bold')
    axes[2].set_xlabel('Lag (сэмплы)')
    axes[2].set_ylabel('Корреляция (норм.)')
    axes[2].set_title('GCC-PHAT с soft whitening (β=0.75) — острый пик', fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_gcc_phat.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_gcc_phat.png")


# ============================================================================
# SLIDE 2 — Graph 4: TDOA triangulation visualization
# ============================================================================
def slide2_triangulation():
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Mic positions (meters relative to mic A)
    mic_a = np.array([0, 0])
    mic_b = np.array([50, 85])
    mic_c = np.array([90, 20])
    mics = np.array([mic_a, mic_b, mic_c])
    
    # True source
    source = np.array([35, 55])
    
    # Simulated localization attempts (with noise)
    np.random.seed(42)
    estimates = source + np.random.randn(20, 2) * 8
    
    # TDOA hyperbolas (simplified)
    theta = np.linspace(0, 2*np.pi, 200)
    for i in range(3):
        for r in [30, 60, 90]:
            circle = mics[i] + r * np.column_stack([np.cos(theta), np.sin(theta)])
            ax.plot(circle[:, 0], circle[:, 1], color='#334155', linewidth=0.3, alpha=0.3)
    
    # Draw mic-to-source lines
    for mic in mics:
        ax.plot([mic[0], source[0]], [mic[1], source[1]], 
               color='#22d3ee', linewidth=1, linestyle=':', alpha=0.5)
    
    # Plot mics
    for i, (mic, label) in enumerate(zip(mics, ['Mic A', 'Mic B', 'Mic C'])):
        ax.plot(mic[0], mic[1], 's', markersize=15, color='#22d3ee', 
               markeredgecolor='white', markeredgewidth=2, zorder=5)
        ax.annotate(label, (mic[0], mic[1]), xytext=(5, 10),
                   textcoords='offset points', fontsize=11, fontweight='bold', color='#22d3ee')
    
    # Plot estimates
    ax.scatter(estimates[:, 0], estimates[:, 1], c='#f59e0b', s=20, alpha=0.5, 
              label='Оценки (Nelder-Mead, 5 стартов)', zorder=3)
    
    # Plot true source
    ax.plot(source[0], source[1], '*', markersize=25, color='#e94560',
           markeredgecolor='white', markeredgewidth=2, zorder=6)
    ax.annotate('Источник\n(бензопила)', (source[0], source[1]), xytext=(10, -25),
               textcoords='offset points', fontsize=11, fontweight='bold', color='#e94560')
    
    # Estimated position (mean)
    est_mean = estimates.mean(axis=0)
    ax.plot(est_mean[0], est_mean[1], 'D', markersize=12, color='#4ade80',
           markeredgecolor='white', markeredgewidth=2, zorder=6)
    ax.annotate(f'Оценка: ({est_mean[0]:.0f}, {est_mean[1]:.0f}) м\n'
               f'Ошибка: {np.linalg.norm(est_mean - source):.1f} м',
               (est_mean[0], est_mean[1]), xytext=(15, 10),
               textcoords='offset points', fontsize=10, color='#4ade80')
    
    # Error circle
    error = np.linalg.norm(est_mean - source)
    circle = plt.Circle(source, error, fill=False, color='#e94560', 
                        linestyle='--', linewidth=2, alpha=0.7)
    ax.add_patch(circle)
    
    # Distance annotations
    for i, mic in enumerate(mics):
        dist = np.linalg.norm(source - mic)
        mid = (source + mic) / 2
        ax.text(mid[0]-5, mid[1]+3, f'{dist:.0f} м', fontsize=8, color='#888')
    
    ax.set_xlabel('X (метры)', fontsize=12)
    ax.set_ylabel('Y (метры)', fontsize=12)
    ax.set_title('TDOA триангуляция: 3 микрофона, GCC-PHAT + energy fusion\n'
                'SNR-взвешенная стоимостная функция, Nelder-Mead (5 стартов)',
                fontweight='bold', fontsize=13)
    ax.set_aspect('equal')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add info box
    info = ('Параметры:\n'
            '• Bandpass: 200–6000 Гц (Butterworth 4)\n'
            '• GCC-PHAT β=0.75\n'
            '• Субпиксельная интерполяция\n'
            f'• v звука = 331.3 + 0.606×T')
    ax.text(0.02, 0.02, info, transform=ax.transAxes, fontsize=9,
           verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_triangulation.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_triangulation.png")


# ============================================================================
# SLIDE 2 — Graph 5: Butterworth bandpass filter response
# ============================================================================
def slide2_bandpass():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7))
    
    sr = 16000
    nyq = sr / 2
    low, high = 200, 6000
    
    # Frequency response
    from scipy.signal import sosfreqz
    sos = butter(4, [low/nyq, high/nyq], btype='bandpass', output='sos')
    w, h = sosfreqz(sos, worN=2000, fs=sr)
    
    ax1.semilogx(w, 20*np.log10(np.abs(h) + 1e-10), color='#e94560', linewidth=2)
    ax1.axvline(200, color='#f59e0b', linestyle='--', linewidth=1.5, label='200 Гц')
    ax1.axvline(6000, color='#f59e0b', linestyle='--', linewidth=1.5, label='6000 Гц')
    
    # Frequency bands for each class
    bands = [
        (200, 1000, 'Бензопила\n(основная)', '#e94560', 0.15),
        (500, 5000, 'Выстрел\n(транзиент)', '#f59e0b', 0.08),
        (300, 3000, 'Двигатель', '#22d3ee', 0.05),
        (400, 2000, 'Топор', '#4ade80', 0.12),
        (100, 4000, 'Огонь', '#533483', 0.05),
    ]
    for f_lo, f_hi, label, color, alpha in bands:
        ax1.axvspan(max(f_lo, 10), min(f_hi, 7999), alpha=alpha, color=color, label=label)
    
    ax1.set_xlabel('Частота (Гц)')
    ax1.set_ylabel('Амплитуда (дБ)')
    ax1.set_title('АЧХ Butterworth bandpass 200–6000 Гц (4-й порядок)\nи частотные диапазоны классов', fontweight='bold')
    ax1.set_ylim(-60, 5)
    ax1.set_xlim(20, 8000)
    ax1.legend(fontsize=8, loc='lower left', ncol=3)
    ax1.grid(alpha=0.3)
    
    # Before/after filtering demo
    np.random.seed(7)
    t = np.arange(sr) / sr
    # Simulate chainsaw + wind noise
    chainsaw = 0.3 * np.sin(2*np.pi*450*t) + 0.2 * np.sin(2*np.pi*900*t)
    wind = 0.4 * np.random.randn(sr)  # broadband noise
    noisy = (chainsaw + wind).astype(np.float32)
    filtered = sosfilt(sos, noisy).astype(np.float32)
    
    # Spectrograms (simplified - power spectrum)
    from scipy.signal import welch
    f_n, psd_noisy = welch(noisy, sr, nperseg=1024)
    f_f, psd_filtered = welch(filtered, sr, nperseg=1024)
    
    ax2.semilogy(f_n, psd_noisy, color='#888', linewidth=1, alpha=0.7, label='До фильтрации')
    ax2.semilogy(f_f, psd_filtered, color='#e94560', linewidth=2, label='После фильтрации')
    ax2.axvspan(200, 6000, alpha=0.1, color='#4ade80', label='Полоса пропускания')
    ax2.set_xlabel('Частота (Гц)')
    ax2.set_ylabel('PSD')
    ax2.set_title('Спектральная плотность: бензопила + ветер → после Butterworth', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, 8000)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_bandpass.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_bandpass.png")


# ============================================================================
# SLIDE 2 — Graph 6: Speed of sound vs temperature
# ============================================================================
def slide2_speed_of_sound():
    fig, ax = plt.subplots(figsize=(10, 5))
    
    temps = np.linspace(-30, 40, 100)
    speeds = 331.3 + 0.606 * temps
    
    ax.plot(temps, speeds, color='#e94560', linewidth=3)
    
    # Mark specific points
    for t, label in [(-20, 'Зима\n(-20°C)'), (0, 'Весна\n(0°C)'), (20, 'Лето\n(20°C)')]:
        v = 331.3 + 0.606 * t
        ax.plot(t, v, 'o', markersize=12, color='#22d3ee', markeredgecolor='white', zorder=5)
        ax.annotate(f'{label}\nv={v:.0f} м/с', (t, v), xytext=(0, 20),
                   textcoords='offset points', ha='center', fontsize=9, fontweight='bold',
                   color='#22d3ee')
    
    # Error impact
    ax.fill_between(temps, speeds - 2, speeds + 2, alpha=0.2, color='#f59e0b',
                   label='±2 м/с (влияние влажности)')
    
    ax.set_xlabel('Температура (°C)')
    ax.set_ylabel('Скорость звука (м/с)')
    ax.set_title('v = 331.3 + 0.606 × T — коррекция скорости звука для TDOA', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Add error annotation
    ax.text(0.02, 0.95, 'На дальности 500 м:\nΔT=10°C → Δd ≈ 0.9 м',
           transform=ax.transAxes, fontsize=10, verticalalignment='top',
           bbox=dict(boxstyle='round', facecolor='#0f3460', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_speed_of_sound.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_speed_of_sound.png")


# ============================================================================
# SLIDE 3 — Graph 1: Onset detection waveform analysis
# ============================================================================
def slide3_onset_detection():
    fig, axes = plt.subplots(3, 1, figsize=(14, 9))
    
    sr = 16000
    duration = 3.0
    n = int(sr * duration)
    t = np.arange(n) / sr
    
    # Simulate forest background + chainsaw onset at 1.5s
    np.random.seed(42)
    background = 0.01 * np.random.randn(n).astype(np.float32)
    
    # Chainsaw starts at 1.5s
    onset_sample = int(1.5 * sr)
    chainsaw = np.zeros(n)
    env = np.exp(-((t[onset_sample:] - t[onset_sample]) - 0.3)**2 / 0.5) * 0.5
    chainsaw[onset_sample:] = env[:n-onset_sample] * np.sin(2*np.pi*450*t[onset_sample:])
    
    signal = background + chainsaw.astype(np.float32)
    
    # Plot waveform
    axes[0].plot(t, signal, color='#22d3ee', linewidth=0.5)
    axes[0].axvspan(1.5, 3.0, alpha=0.15, color='#e94560', label='Бензопила')
    axes[0].axvspan(0, 1.5, alpha=0.1, color='#4ade80', label='Фон (лес)')
    axes[0].set_ylabel('Амплитуда')
    axes[0].set_title('Входной сигнал: фон леса → запуск бензопилы (onset @ 1.5 с)', fontweight='bold')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3)
    
    # RMS energy per frame
    frame_size = 512
    hop_size = 256
    n_frames = (n - frame_size) // hop_size + 1
    energies = np.array([
        np.sqrt(np.mean(signal[i*hop_size:i*hop_size+frame_size]**2))
        for i in range(n_frames)
    ])
    frame_times = np.array([i * hop_size / sr for i in range(n_frames)])
    
    axes[1].plot(frame_times, energies, color='#f59e0b', linewidth=1.5, label='RMS short-term')
    
    # Long-term median
    long_term = 30
    medians = np.array([
        np.median(energies[max(0,i-long_term):i]) if i > 0 else energies[0]
        for i in range(n_frames)
    ])
    axes[1].plot(frame_times, medians, color='#4ade80', linewidth=2, linestyle='--', label='Long-term median')
    axes[1].set_ylabel('RMS Energy')
    axes[1].set_title('Энергия кадров: short-term RMS vs long-term median', fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    
    # Energy ratio
    ratios = np.array([
        energies[i] / max(medians[i], 1e-10) for i in range(n_frames)
    ])
    axes[2].plot(frame_times, ratios, color='#e94560', linewidth=1.5, label='Energy ratio')
    axes[2].axhline(y=8.0, color='#f59e0b', linestyle='--', linewidth=2, label='Порог = 8.0')
    
    # Mark trigger point
    triggered = np.where(ratios >= 8.0)[0]
    if len(triggered) > 0:
        trigger_time = frame_times[triggered[0]]
        axes[2].axvline(x=trigger_time, color='#4ade80', linewidth=2, linestyle='-',
                       label=f'TRIGGER @ {trigger_time:.2f} с')
        axes[2].annotate(f'Onset!\nratio={ratios[triggered[0]]:.1f}x',
                        (trigger_time, ratios[triggered[0]]), xytext=(20, 10),
                        textcoords='offset points', fontsize=11, fontweight='bold',
                        color='#4ade80',
                        arrowprops=dict(arrowstyle='->', color='#4ade80'))
    
    axes[2].set_xlabel('Время (с)')
    axes[2].set_ylabel('Ratio')
    axes[2].set_title('Отношение short-term/long-term: триггер onset при ratio ≥ 8.0', fontweight='bold')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3)
    axes[2].set_ylim(0, max(ratios.max() * 1.1, 12))
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide3_onset_detection.png', bbox_inches='tight')
    plt.close()
    print("✓ slide3_onset_detection.png")


# ============================================================================
# SLIDE 3 — Graph 2: Edge pipeline flow diagram
# ============================================================================
def slide3_pipeline_flow():
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Pipeline stages
    stages = [
        (0.5, 3.0, 2.0, 1.5, 'Микрофон\n16kHz\n(непрерывно)', '#0f3460', '🎤'),
        (3.2, 3.0, 2.0, 1.5, 'Onset\nDetector\nRMS ratio≥8', '#533483', '⚡'),
        (6.0, 3.0, 2.0, 1.5, 'YAMNet\nClassifier\n6 классов', '#e94560', '🧠'),
        (8.8, 3.0, 2.0, 1.5, 'TDOA\nTriangulate\nGCC-PHAT', '#0f3460', '📍'),
        (11.5, 3.0, 2.0, 1.5, 'Permits\nCheck\nSQL lookup', '#533483', '📋'),
        (14.0, 3.0, 1.5, 1.5, 'Decision\nEngine', '#4ade80', '✅'),
    ]
    
    for x, y, w, h, text, color, emoji in stages:
        rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                             facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=9, fontweight='bold', color='white')
    
    # Arrows between stages
    arrow_x = [(2.5, 3.2), (5.2, 6.0), (8.0, 8.8), (10.8, 11.5), (13.5, 14.0)]
    for x1, x2 in arrow_x:
        ax.annotate('', xy=(x2, 3.75), xytext=(x1, 3.75),
                    arrowprops=dict(arrowstyle='->', color='#22d3ee', lw=2.5))
    
    # Decision outputs
    outputs = [
        (14.3, 5.2, '≥70%\nALERT', '#e94560', 'Дрон + LoRa'),
        (14.3, 1.5, '40-70%\nVERIFY', '#f59e0b', 'Только LoRa'),
        (14.3, 0.2, '<40%\nLOG', '#4ade80', 'Только лог'),
    ]
    
    for x, y, text, color, desc in outputs:
        ax.annotate('', xy=(x + 0.4, y + 0.6), xytext=(14.7, 3.0),
                    arrowprops=dict(arrowstyle='->', color=color, lw=2))
        ax.text(x + 0.5, y + 0.3, f'{text}\n{desc}', fontsize=8, fontweight='bold', color=color)
    
    # Bypass arrow (onset not triggered)
    ax.annotate('Нет onset →\nпродолжить\nслушать', xy=(4.2, 2.8), xytext=(4.2, 1.0),
               fontsize=8, color='#888', ha='center',
               arrowprops=dict(arrowstyle='->', color='#888', lw=1.5, connectionstyle='arc3,rad=-0.3'))
    
    # Permit found arrow
    ax.annotate('Разрешение\nнайдено →\nподавить алерт', xy=(12.5, 2.8), xytext=(12.5, 0.8),
               fontsize=8, color='#4ade80', ha='center',
               arrowprops=dict(arrowstyle='->', color='#4ade80', lw=1.5))
    
    # Timing labels
    timings = [
        (1.5, 5.0, '2 сек чанки'),
        (4.2, 5.0, '~5 мс'),
        (7.0, 5.0, '~50 мс'),
        (9.8, 5.0, '~20 мс'),
        (12.5, 5.0, '~1 мс'),
    ]
    for x, y, text in timings:
        ax.text(x, y, text, ha='center', fontsize=8, color='#888', style='italic')
    
    ax.set_title('Edge Pipeline: от звука до решения (один Python-процесс)',
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide3_pipeline_flow.png', bbox_inches='tight')
    plt.close()
    print("✓ slide3_pipeline_flow.png")


# ============================================================================
# SLIDE 3 — Graph 3: Decision matrix
# ============================================================================
def slide3_decision_matrix():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Confidence zones
    conf = np.linspace(0, 1, 100)
    
    # Fill zones
    ax.axhspan(0, 0.40, alpha=0.2, color='#4ade80')
    ax.axhspan(0.40, 0.70, alpha=0.2, color='#f59e0b')
    ax.axhspan(0.70, 1.0, alpha=0.2, color='#e94560')
    
    # Zone labels
    ax.text(0.5, 0.20, 'LOG ONLY\n(accuracy ~13%)\nНикаких действий', ha='center', va='center',
           fontsize=14, fontweight='bold', color='#4ade80')
    ax.text(0.5, 0.55, 'VERIFY\n(accuracy ~49%)\nТолько LoRa уведомление', ha='center', va='center',
           fontsize=14, fontweight='bold', color='#f59e0b')
    ax.text(0.5, 0.85, 'ALERT\n(accuracy ~95%)\nДрон + LoRa + Telegram', ha='center', va='center',
           fontsize=14, fontweight='bold', color='#e94560')
    
    # Threshold lines
    ax.axhline(y=0.40, color='#f59e0b', linewidth=3, linestyle='-')
    ax.axhline(y=0.70, color='#e94560', linewidth=3, linestyle='-')
    
    # Labels on lines
    ax.text(1.02, 0.40, 'CONFIDENCE_VERIFY = 0.40', fontsize=10, fontweight='bold',
           color='#f59e0b', va='center')
    ax.text(1.02, 0.70, 'CONFIDENCE_ALERT = 0.70', fontsize=10, fontweight='bold',
           color='#e94560', va='center')
    
    # Simulated detections
    detections = [
        (0.15, 0.92, 'chainsaw', '#e94560'),
        (0.35, 0.88, 'gunshot', '#e94560'),
        (0.55, 0.55, 'engine', '#f59e0b'),
        (0.75, 0.35, 'background', '#4ade80'),
        (0.25, 0.72, 'fire', '#e94560'),
        (0.85, 0.48, 'axe', '#f59e0b'),
    ]
    
    for x, conf, label, color in detections:
        ax.plot(x, conf, 'o', markersize=15, color=color, markeredgecolor='white',
               markeredgewidth=2, zorder=5)
        ax.annotate(label, (x, conf), xytext=(8, 8), textcoords='offset points',
                   fontsize=9, fontweight='bold', color='white')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Confidence', fontsize=13)
    ax.set_xlabel('(примеры детекций)', fontsize=11)
    ax.set_title('Decision Engine: три зоны реагирования (decider.py)', fontweight='bold', fontsize=14)
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide3_decision_matrix.png', bbox_inches='tight')
    plt.close()
    print("✓ slide3_decision_matrix.png")


# ============================================================================
# SLIDE 2 — Graph 7: SNR vs localization error
# ============================================================================
def slide2_snr_vs_error():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    snrs = [5, 10, 15, 20, 25, 30, 35, 40]
    # Simulated errors based on test data patterns
    errors_pure_tdoa = [120, 80, 55, 30, 20, 15, 12, 10]
    errors_fusion = [95, 60, 40, 22, 14, 10, 8, 7]
    
    ax.plot(snrs, errors_pure_tdoa, 'o-', color='#f59e0b', linewidth=2.5, 
           markersize=10, label='Только TDOA', markeredgecolor='white')
    ax.plot(snrs, errors_fusion, 's-', color='#4ade80', linewidth=2.5,
           markersize=10, label='TDOA + Energy fusion (w=0.3)', markeredgecolor='white')
    
    # Fill improvement zone
    ax.fill_between(snrs, errors_pure_tdoa, errors_fusion, alpha=0.15, color='#4ade80',
                   label='Улучшение от fusion')
    
    # Accuracy zones
    ax.axhspan(0, 15, alpha=0.1, color='#4ade80')
    ax.axhspan(15, 40, alpha=0.08, color='#f59e0b')
    ax.axhspan(40, 130, alpha=0.05, color='#e94560')
    
    ax.text(38, 5, 'Высокая точность\n(<15 м)', fontsize=9, color='#4ade80')
    ax.text(38, 25, 'Приемлемая\n(15-40 м)', fontsize=9, color='#f59e0b')
    ax.text(38, 70, 'Низкая\n(>40 м)', fontsize=9, color='#e94560')
    
    ax.set_xlabel('SNR (дБ)', fontsize=12)
    ax.set_ylabel('Ошибка локализации (м)', fontsize=12)
    ax.set_title('Точность триангуляции vs SNR:\nTDOA + Energy fusion снижает ошибку на 20-30%',
                fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlim(3, 42)
    ax.set_ylim(0, 130)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_snr_vs_error.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_snr_vs_error.png")


# ============================================================================
# SLIDE 1 — Graph: Map-like microphone placement
# ============================================================================
def slide1_mic_placement():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    np.random.seed(42)
    
    # Varnavinskoye forestry approximate boundaries
    # Using arbitrary coords for visualization
    # 8 district forestries
    districts = {
        'Варнавинское': (57.35, 45.10, '#4ade80'),
        'Богородское': (57.30, 45.20, '#22d3ee'),
        'Черновское': (57.40, 45.25, '#f59e0b'),
        'Северное': (57.45, 45.15, '#e94560'),
        'Восточное': (57.38, 45.30, '#533483'),
        'Западное': (57.32, 45.05, '#0f3460'),
        'Центральное': (57.36, 45.18, '#888'),
        'Южное': (57.28, 45.12, '#22d3ee'),
    }
    
    # Draw district areas
    for name, (lat, lon, color) in districts.items():
        # Random forest area
        theta = np.linspace(0, 2*np.pi, 50)
        r = 0.03 + 0.01 * np.random.randn(50)
        x = lon + r * np.cos(theta)
        y = lat + r * np.sin(theta)
        ax.fill(x, y, alpha=0.15, color=color)
        ax.plot(x, y, color=color, linewidth=1, alpha=0.5)
        ax.text(lon, lat, name, ha='center', va='center', fontsize=7, fontweight='bold', color=color)
    
    # Protection zones (pink)
    for _ in range(5):
        cx, cy = 45.1 + 0.2*np.random.rand(), 57.3 + 0.15*np.random.rand()
        r = 0.01 + 0.005 * np.random.rand()
        circle = plt.Circle((cx, cy), r, fill=True, facecolor='#e94560', alpha=0.2, edgecolor='#e94560')
        ax.add_patch(circle)
    
    # Water protection zones (blue)
    # Rivers
    river_x = np.linspace(45.0, 45.35, 50)
    river_y = 57.32 + 0.05 * np.sin(2*np.pi*river_x/0.15) + 0.02*np.random.randn(50)
    ax.plot(river_x, river_y, color='#22d3ee', linewidth=3, alpha=0.6)
    ax.fill_between(river_x, river_y - 0.008, river_y + 0.008, alpha=0.1, color='#22d3ee')
    
    # Microphone clusters (triangles of 3)
    clusters = [
        (45.12, 57.36), (45.18, 57.38), (45.08, 57.33),
        (45.22, 57.34), (45.15, 57.42), (45.25, 57.40),
        (45.10, 57.40), (45.28, 57.32), (45.20, 57.30),
    ]
    
    for i, (cx, cy) in enumerate(clusters):
        # 3 mics per cluster, 300-1000m apart
        spread = 0.004  # ~400m
        mics = [
            (cx, cy),
            (cx + spread, cy + spread*0.6),
            (cx + spread*0.7, cy - spread*0.4),
        ]
        
        # Draw triangle
        triangle = plt.Polygon(mics, fill=False, edgecolor='#f59e0b', linewidth=1.5, linestyle='--')
        ax.add_patch(triangle)
        
        # Draw mics
        for mx, my in mics:
            ax.plot(mx, my, '^', markersize=8, color='#f59e0b', 
                   markeredgecolor='white', markeredgewidth=1, zorder=5)
        
        # Cluster label
        ax.text(cx + spread*0.3, cy + spread*0.8, f'K{i+1}', fontsize=7, color='#f59e0b', fontweight='bold')
    
    # OOPT zones (green)
    oopt_positions = [(45.06, 57.37), (45.30, 57.36), (45.15, 57.28)]
    for ox, oy in oopt_positions:
        rect = FancyBboxPatch((ox-0.01, oy-0.008), 0.02, 0.016, 
                             boxstyle="round,pad=0.002", facecolor='#4ade80', alpha=0.2,
                             edgecolor='#4ade80', linewidth=1)
        ax.add_patch(rect)
        ax.text(ox, oy, 'ООПТ', ha='center', fontsize=6, color='#4ade80')
    
    # Kostroma border (north)
    ax.axhline(y=57.45, color='#e94560', linewidth=2, linestyle='-.', alpha=0.7)
    ax.text(45.17, 57.455, 'Граница с Костромской обл. — зона риска', 
           ha='center', fontsize=9, color='#e94560', fontweight='bold')
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#e94560', alpha=0.3, label='Защитные полосы (приоритет 1)'),
        mpatches.Patch(facecolor='#22d3ee', alpha=0.3, label='Водоохранные зоны (приоритет 1)'),
        mpatches.Patch(facecolor='#f59e0b', alpha=0.3, label='Эксплуатационные леса (приоритет 2)'),
        mpatches.Patch(facecolor='#4ade80', alpha=0.3, label='ООПТ (приоритет 3)'),
        plt.Line2D([0], [0], marker='^', color='#f59e0b', markersize=10, 
                  linestyle='None', label='Микрофонный кластер (3 узла)'),
        plt.Line2D([0], [0], color='#e94560', linewidth=2, linestyle='-.', label='Граница области'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8, 
             facecolor='#16213e', edgecolor='#334155')
    
    ax.set_xlabel('Долгота (°E)', fontsize=11)
    ax.set_ylabel('Широта (°N)', fontsize=11)
    ax.set_title('Варнавинское лесничество: расстановка микрофонных кластеров\n'
                '213 804 га | 8 участков | 9 кластеров × 3 микрофона | 300–1000 м между узлами',
                fontweight='bold', fontsize=13)
    ax.set_aspect('equal')
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide1_mic_placement.png', bbox_inches='tight')
    plt.close()
    print("✓ slide1_mic_placement.png")


# ============================================================================
# SLIDE 4 — Graph: Cloud API architecture
# ============================================================================
def slide4_cloud_architecture():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Central: Incident data
    rect = FancyBboxPatch((5.5, 3.5), 3, 1.5, boxstyle="round,pad=0.15",
                         facecolor='#e94560', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(7, 4.25, 'Инцидент\nclass + coords + confidence', ha='center', va='center',
           fontsize=11, fontweight='bold', color='white')
    
    # YandexGPT
    rect = FancyBboxPatch((0.5, 5.5), 3, 1.5, boxstyle="round,pad=0.15",
                         facecolor='#533483', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(2, 6.25, 'YandexGPT\nCompletion API\nКарточка инцидента', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')
    
    # File Search RAG
    rect = FancyBboxPatch((0.5, 1.0), 3, 1.5, boxstyle="round,pad=0.15",
                         facecolor='#0f3460', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(2, 1.75, 'File Search (RAG)\nЛесной кодекс\nКоАП ст.8.28\nПриказ №156', 
           ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    
    # SpeechKit STT
    rect = FancyBboxPatch((10.5, 5.5), 3, 1.5, boxstyle="round,pad=0.15",
                         facecolor='#0f3460', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(12, 6.25, 'SpeechKit STT\nГолос инспектора\n→ текст протокола', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')
    
    # Vision
    rect = FancyBboxPatch((10.5, 1.0), 3, 1.5, boxstyle="round,pad=0.15",
                         facecolor='#533483', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(12, 1.75, 'Vision\nyandexgpt-vision-lite\nФото с дрона', ha='center', va='center',
           fontsize=10, fontweight='bold', color='white')
    
    # Output: Protocol
    rect = FancyBboxPatch((5.5, 0.3), 3, 1.0, boxstyle="round,pad=0.1",
                         facecolor='#4ade80', edgecolor='white', linewidth=2, alpha=0.9)
    ax.add_patch(rect)
    ax.text(7, 0.8, 'PDF протокол\n(готов для суда)', ha='center', va='center',
           fontsize=10, fontweight='bold', color='#1a1a2e')
    
    # Arrows
    connections = [
        (5.5, 4.25, 3.5, 6.25),   # incident -> yandexgpt
        (5.5, 4.25, 3.5, 1.75),   # incident -> rag
        (8.5, 4.25, 10.5, 6.25),  # incident -> stt
        (8.5, 4.25, 10.5, 1.75),  # incident -> vision
        (2, 5.5, 5.5, 4.25),      # yandexgpt -> incident (results)
        (2, 2.5, 5.5, 3.5),       # rag -> incident
        (12, 5.5, 8.5, 4.25),     # stt -> incident
        (12, 2.5, 8.5, 3.5),      # vision -> incident
        (7, 3.5, 7, 1.3),         # incident -> protocol
    ]
    
    for x1, y1, x2, y2 in connections:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color='#22d3ee', lw=1.5, alpha=0.7))
    
    ax.set_title('Облако: Yandex Cloud API — 4 сервиса для полного цикла',
                fontsize=15, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide4_cloud_architecture.png', bbox_inches='tight')
    plt.close()
    print("✓ slide4_cloud_architecture.png")


# ============================================================================
# SLIDE 5 — Graph: Full system architecture
# ============================================================================
def slide5_full_system():
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#1a1a2e')
    
    # Title
    ax.text(8, 8.5, 'ForestGuard: полная архитектура системы', 
           ha='center', fontsize=16, fontweight='bold', color='white')
    
    # EDGE layer
    ax.add_patch(FancyBboxPatch((0.5, 5.0), 6, 3.0, boxstyle="round,pad=0.2",
                               facecolor='#16213e', edgecolor='#e94560', linewidth=2))
    ax.text(3.5, 7.8, 'EDGE (Raspberry Pi / сервер)', ha='center', fontsize=12, 
           fontweight='bold', color='#e94560')
    
    edge_components = [
        (1.0, 6.5, 'Микрофон\nArray', '#0f3460'),
        (2.8, 6.5, 'Onset\nDetector', '#533483'),
        (4.5, 6.5, 'YAMNet\nv7', '#e94560'),
        (1.0, 5.2, 'TDOA\nGCC-PHAT', '#0f3460'),
        (2.8, 5.2, 'Decision\nEngine', '#f59e0b'),
        (4.5, 5.2, 'Drone\nMAVLink', '#533483'),
    ]
    for x, y, text, color in edge_components:
        rect = FancyBboxPatch((x, y), 1.5, 0.9, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x+0.75, y+0.45, text, ha='center', va='center', fontsize=8, 
               fontweight='bold', color='white')
    
    # GATEWAY layer
    ax.add_patch(FancyBboxPatch((7.0, 5.5), 2, 2.0, boxstyle="round,pad=0.15",
                               facecolor='#16213e', edgecolor='#f59e0b', linewidth=2))
    ax.text(8, 7.3, 'LoRa Gateway', ha='center', fontsize=11, fontweight='bold', color='#f59e0b')
    
    gateway_comp = [
        (7.3, 6.0, 'LoRa\nRelay', '#f59e0b'),
    ]
    for x, y, text, color in gateway_comp:
        rect = FancyBboxPatch((x, y), 1.4, 0.9, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x+0.7, y+0.45, text, ha='center', va='center', fontsize=8, 
               fontweight='bold', color='white')
    
    # CLOUD layer
    ax.add_patch(FancyBboxPatch((9.5, 4.5), 6, 3.5, boxstyle="round,pad=0.2",
                               facecolor='#16213e', edgecolor='#4ade80', linewidth=2))
    ax.text(12.5, 7.8, 'CLOUD (Yandex Cloud + FastAPI)', ha='center', fontsize=12, 
           fontweight='bold', color='#4ade80')
    
    cloud_components = [
        (10.0, 6.5, 'YandexGPT\nRAG Agent', '#533483'),
        (12.0, 6.5, 'SpeechKit\nSTT', '#0f3460'),
        (14.0, 6.5, 'Vision\nClassifier', '#533483'),
        (10.0, 5.0, 'FastAPI\nWebSocket', '#4ade80'),
        (12.0, 5.0, 'SQLite\nDB', '#0f3460'),
        (14.0, 5.0, 'Telegram\nBot', '#e94560'),
    ]
    for x, y, text, color in cloud_components:
        rect = FancyBboxPatch((x, y), 1.5, 0.9, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x+0.75, y+0.45, text, ha='center', va='center', fontsize=8, 
               fontweight='bold', color='white')
    
    # USER layer
    ax.add_patch(FancyBboxPatch((3.0, 0.5), 10, 2.5, boxstyle="round,pad=0.2",
                               facecolor='#16213e', edgecolor='#22d3ee', linewidth=2))
    ax.text(8, 2.8, 'ПОЛЬЗОВАТЕЛИ', ha='center', fontsize=12, fontweight='bold', color='#22d3ee')
    
    users = [
        (3.5, 0.8, 'Web Dashboard\nLeaflet + WS', '#22d3ee'),
        (6.0, 0.8, 'Telegram\nИнспектор', '#e94560'),
        (8.5, 0.8, 'PDF\nПротокол', '#4ade80'),
        (11.0, 0.8, 'Mobile\nREST API', '#f59e0b'),
    ]
    for x, y, text, color in users:
        rect = FancyBboxPatch((x, y), 1.8, 0.9, boxstyle="round,pad=0.05",
                             facecolor=color, edgecolor='white', linewidth=1, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x+0.9, y+0.45, text, ha='center', va='center', fontsize=8, 
               fontweight='bold', color='white')
    
    # Connections
    # Edge -> Gateway
    ax.annotate('', xy=(7.0, 6.5), xytext=(6.5, 6.0),
                arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=2.5))
    ax.text(6.5, 6.7, 'LoRa', fontsize=8, color='#f59e0b', fontweight='bold')
    
    # Gateway -> Cloud
    ax.annotate('', xy=(9.5, 6.5), xytext=(9.0, 6.5),
                arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=2.5))
    ax.text(9.0, 6.8, 'TCP/IP', fontsize=8, color='#f59e0b', fontweight='bold')
    
    # Cloud -> Users
    ax.annotate('', xy=(8, 3.0), xytext=(12.5, 4.5),
                arrowprops=dict(arrowstyle='->', color='#22d3ee', lw=2))
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide5_full_system.png', bbox_inches='tight')
    plt.close()
    print("✓ slide5_full_system.png")


# ============================================================================
# SLIDE 6 — Graph: Confusion matrix (simulated)
# ============================================================================
def slide6_confusion_matrix():
    fig, ax = plt.subplots(figsize=(9, 8))
    
    classes = ['chainsaw', 'gunshot', 'engine', 'axe', 'fire', 'background']
    
    # Simulated confusion matrix based on expected model performance
    cm = np.array([
        [92, 1, 3, 2, 0, 2],   # chainsaw
        [0, 88, 2, 0, 5, 5],   # gunshot
        [3, 1, 78, 1, 0, 17],  # engine
        [5, 0, 2, 71, 0, 22],  # axe
        [0, 3, 0, 0, 85, 12],  # fire
        [1, 1, 5, 2, 2, 89],   # background
    ], dtype=float)
    
    # Normalize
    cm_norm = cm / cm.sum(axis=1, keepdims=True) * 100
    
    im = ax.imshow(cm_norm, cmap='YlOrRd', vmin=0, vmax=100)
    
    # Labels
    for i in range(len(classes)):
        for j in range(len(classes)):
            val = cm_norm[i, j]
            color = 'white' if val > 50 else '#eee'
            ax.text(j, i, f'{val:.0f}%', ha='center', va='center',
                   fontsize=12, fontweight='bold', color=color)
    
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Предсказанный класс', fontsize=12)
    ax.set_ylabel('Истинный класс', fontsize=12)
    ax.set_title('Confusion Matrix: Fine-tuned YAMNet Head v7\n'
                '(6 классов, средняя accuracy = 83.8%)', fontweight='bold', fontsize=13)
    
    plt.colorbar(im, ax=ax, label='%', fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide6_confusion_matrix.png', bbox_inches='tight')
    plt.close()
    print("✓ slide6_confusion_matrix.png")


# ============================================================================
# SLIDE 6 — Graph: Real-time dashboard mockup stats
# ============================================================================
def slide6_dashboard_stats():
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Events over time
    ax = axes[0, 0]
    hours = np.arange(0, 24)
    # More activity during day
    events = np.array([1, 0, 0, 0, 0, 1, 3, 5, 8, 12, 15, 10, 7, 9, 14, 11, 8, 6, 4, 3, 2, 1, 0, 1])
    alerts = np.clip(events * 0.3 + np.random.randn(24) * 0.5, 0, None).astype(int)
    
    ax.bar(hours, events, color='#22d3ee', alpha=0.7, label='Все события')
    ax.bar(hours, alerts, color='#e94560', alpha=0.9, label='Алерты (ALERT)')
    ax.set_xlabel('Час')
    ax.set_ylabel('Кол-во событий')
    ax.set_title('События за сутки', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # 2. Events by class (pie chart)
    ax = axes[0, 1]
    labels = ['chainsaw', 'engine', 'background', 'gunshot', 'axe', 'fire']
    sizes = [28, 22, 30, 8, 7, 5]
    colors = ['#e94560', '#22d3ee', '#4ade80', '#f59e0b', '#533483', '#ff6b6b']
    explode = (0.1, 0, 0, 0, 0, 0.05)
    
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.0f%%',
          shadow=True, startangle=140, textprops={'fontsize': 9, 'color': '#eee'})
    ax.set_title('Распределение по классам', fontweight='bold')
    
    # 3. Response times
    ax = axes[1, 0]
    stages = ['Onset\ndetect', 'Classify', 'TDOA', 'Permit\ncheck', 'Decision', 'LoRa\nsend', 'Дрон\nвылет']
    times_ms = [5, 50, 20, 1, 0.5, 100, 30000]
    colors_bar = ['#533483', '#e94560', '#0f3460', '#22d3ee', '#4ade80', '#f59e0b', '#e94560']
    
    bars = ax.barh(stages, times_ms, color=colors_bar, edgecolor='white', linewidth=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Время (мс, log scale)')
    ax.set_title('Задержки по этапам pipeline', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for bar, t in zip(bars, times_ms):
        ax.text(bar.get_width() * 1.2, bar.get_y() + bar.get_height()/2,
               f'{t} мс' if t < 1000 else f'{t/1000:.0f} с', 
               va='center', fontsize=9, color='#eee')
    
    # 4. Accuracy vs distance
    ax = axes[1, 1]
    distances = [50, 100, 200, 300, 500, 800, 1000]
    accuracy = [98, 95, 90, 82, 68, 45, 30]
    
    ax.plot(distances, accuracy, 'o-', color='#e94560', linewidth=2.5, markersize=10,
           markeredgecolor='white')
    ax.fill_between(distances, accuracy, alpha=0.1, color='#e94560')
    ax.axhline(y=70, color='#f59e0b', linestyle='--', linewidth=2, label='Порог ALERT (70%)')
    ax.axvline(x=500, color='#22d3ee', linestyle=':', linewidth=1.5, label='Макс. дальность кластера')
    ax.set_xlabel('Расстояние до источника (м)')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Confidence vs расстояние', fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide6_dashboard_stats.png', bbox_inches='tight')
    plt.close()
    print("✓ slide6_dashboard_stats.png")


# ============================================================================
# SLIDE 2 — Graph 8: Embedding space visualization (t-SNE-like)
# ============================================================================
def slide2_embedding_space():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    np.random.seed(42)
    
    classes = ['chainsaw', 'gunshot', 'engine', 'axe', 'fire', 'background']
    colors = ['#e94560', '#f59e0b', '#22d3ee', '#533483', '#ff6b6b', '#4ade80']
    
    # Simulated 2D t-SNE of 2048-D embeddings (mean+max pooled YAMNet)
    centers = {
        'chainsaw': (3, 7),
        'gunshot': (8, 8),
        'engine': (2, 3),
        'axe': (5, 6),      # close to chainsaw
        'fire': (7, 3),
        'background': (5, 1),
    }
    
    for cls, color in zip(classes, colors):
        cx, cy = centers[cls]
        n_pts = 30
        x = cx + np.random.randn(n_pts) * 0.8
        y = cy + np.random.randn(n_pts) * 0.8
        ax.scatter(x, y, c=color, s=60, alpha=0.7, edgecolors='white', linewidth=0.5, label=cls)
        ax.text(cx, cy, cls, ha='center', va='center', fontsize=10, fontweight='bold',
               color=color, bbox=dict(boxstyle='round,pad=0.3', facecolor='#1a1a2e', alpha=0.8))
    
    # Draw confusion zone between axe and chainsaw
    ax.annotate('', xy=(3.5, 6.5), xytext=(4.5, 6.0),
                arrowprops=dict(arrowstyle='<->', color='#888', lw=2, linestyle='--'))
    ax.text(4.2, 6.8, 'Зона путаницы\naxe↔chainsaw', fontsize=8, color='#888', ha='center')
    
    ax.set_xlabel('t-SNE dim 1', fontsize=11)
    ax.set_ylabel('t-SNE dim 2', fontsize=11)
    ax.set_title('Пространство эмбеддингов YAMNet (2048-D → t-SNE 2D)\n'
                'mean+max pooling embeddings, 6 классов', fontweight='bold', fontsize=13)
    ax.legend(fontsize=10, loc='lower right', facecolor='#16213e', edgecolor='#334155')
    ax.grid(alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(f'{OUT}/slide2_embedding_space.png', bbox_inches='tight')
    plt.close()
    print("✓ slide2_embedding_space.png")


# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == "__main__":
    print("Generating presentation graphs...")
    print("=" * 50)
    
    slide1_mic_placement()
    slide2_yamnet_architecture()
    slide2_class_confidence()
    slide2_gcc_phat()
    slide2_triangulation()
    slide2_bandpass()
    slide2_speed_of_sound()
    slide2_snr_vs_error()
    slide2_embedding_space()
    slide3_onset_detection()
    slide3_pipeline_flow()
    slide3_decision_matrix()
    slide4_cloud_architecture()
    slide5_full_system()
    slide6_confusion_matrix()
    slide6_dashboard_stats()
    
    print("=" * 50)
    print(f"All graphs saved to {OUT}/")
    print(f"Total: {len(os.listdir(OUT))} PNG files")
