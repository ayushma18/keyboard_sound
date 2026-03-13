"""
Ultimate Keystroke Segmentation Module — v2 Ultimate

COMPLETE FEATURE SET:
✓ Multi-scale STE peak-finding (v2 core)
✓ Local SNR gates (10ms window) 
✓ Peak amplitude + crest factor checks
✓ Envelope-based template matching
✓ Bandpass filtering + overlap guard
✓ Diagnostic rejection logging
✓ Batch processing with session discovery
✓ CSV-time guidance with automatic peak refinement
✓ EMBEDDED WAVEFORM PREVIEW with offset adjustment
✓ LIVE VISUALIZATION while adjusting offset
✓ FULL-AUDIO VERIFICATION VIEWER after extraction
✓ Interactive playback controls (zoom, pan, navigate)
✓ High-quality matplotlib rendering
"""
import os
import csv
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import signal
from scipy.ndimage import uniform_filter1d
import threading

try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.widgets import Slider
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS (ALL V2 CORE)
# ═══════════════════════════════════════════════════════════════════════════════

def _short_time_energy(audio: np.ndarray, win_samples: int) -> np.ndarray:
    """Sliding-window mean of squared samples."""
    sq = audio.astype(np.float64) ** 2
    return uniform_filter1d(sq, size=max(1, win_samples), mode='constant', origin=0)


def _estimate_noise_floor(audio: np.ndarray, sr: int,
                           frame_dur: float = 0.050,
                           quantile: float = 0.05) -> float:
    """Estimate noise floor RMS using the quietest frames."""
    frame_len = int(frame_dur * sr)
    n_frames = len(audio) // frame_len
    if n_frames < 5:
        return 1e-7
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    floor = float(np.percentile(rms_per_frame, quantile * 100))
    return max(floor, 1e-7)


def _local_peak_rms(audio: np.ndarray, center: int, sr: int,
                     window_ms: float = 10.0) -> float:
    """RMS of short window centered on peak."""
    half_win = int(window_ms / 1000.0 * sr / 2)
    lo = max(0, center - half_win)
    hi = min(len(audio), center + half_win)
    if hi <= lo:
        return 0.0
    seg = audio[lo:hi].astype(np.float64)
    return float(np.sqrt(np.mean(seg ** 2)))


def _find_waveform_peak(audio: np.ndarray, center: int, search_radius: int) -> int:
    """Index of max |amplitude| within ±search_radius of center."""
    lo = max(0, center - search_radius)
    hi = min(len(audio), center + search_radius)
    if hi <= lo:
        return center
    return int(lo + np.argmax(np.abs(audio[lo:hi])))


def _rms(segment: np.ndarray) -> float:
    return float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))


def _amplitude_envelope(audio: np.ndarray, sr: int, smooth_ms: float = 5.0) -> np.ndarray:
    """Amplitude envelope for template matching."""
    env = np.abs(audio.astype(np.float64))
    smooth_samples = max(1, int(smooth_ms / 1000.0 * sr))
    return uniform_filter1d(env, size=smooth_samples, mode='constant')


def _max_ncc(template: np.ndarray, candidate: np.ndarray, max_lag: int) -> float:
    """Maximum normalized cross-correlation over ±max_lag range."""
    n = min(len(template), len(candidate))
    if n == 0:
        return 0.0
    t = template[:n].astype(np.float64)
    c = candidate[:n].astype(np.float64)
    t_m = t - np.mean(t)
    t_norm = np.linalg.norm(t_m)
    if t_norm < 1e-12:
        return 0.0
    best_corr = -1.0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            t_slice = t_m[:n - lag]
            c_slice = c[lag:n]
        else:
            t_slice = t_m[-lag:n]
            c_slice = c[:n + lag]
        if len(t_slice) == 0:
            continue
        c_slice = c_slice - np.mean(c_slice)
        c_norm = np.linalg.norm(c_slice)
        t_sl_norm = np.linalg.norm(t_slice)
        if c_norm < 1e-12 or t_sl_norm < 1e-12:
            continue
        corr = float(np.dot(t_slice, c_slice) / (t_sl_norm * c_norm))
        if corr > best_corr:
            best_corr = corr
    return best_corr


def _crest_factor(segment: np.ndarray) -> float:
    """Peak-to-RMS ratio."""
    rms = _rms(segment)
    if rms < 1e-12:
        return 0.0
    return float(np.max(np.abs(segment))) / rms


def _bandpass(seg, sr, low, high):
    """Butterworth bandpass filter."""
    nyq = sr / 2.0
    lo = max(low, 1) / nyq
    hi = min(high, nyq - 1) / nyq
    if lo >= hi:
        return seg
    sos = signal.butter(4, [lo, hi], btype='band', output='sos')
    return signal.sosfiltfilt(sos, seg).astype(seg.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE EXTRACTION (V2 + CSV GUIDANCE)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_keystrokes(
    audio: np.ndarray,
    sr: int,
    keystroke_log: List[Dict],
    *,
    segment_duration: float = 0.430,
    pre_trigger: float = 0.10,
    search_radius_sec: float = 0.25,
    peak_snr_db: float = 6.0,
    min_crest_factor: float = 1.8,
    min_peak_amplitude: float = 0.0,
    enable_template_verify: bool = True,
    template_corr_threshold: float = 0.20,
    template_build_count: int = 20,
    enable_bandpass: bool = False,
    bandpass_low: int = 50,
    bandpass_high: int = 5000,
    overlap_guard_sec: float = 0.02,
    verbose: bool = True,
    progress_callback=None,
) -> Tuple[List[Dict], Dict]:
    """CSV-guided extraction with v2 peak-finding + quality gates."""

    if audio.ndim == 2:
        mono = np.mean(audio, axis=1).astype(np.float64)
        is_stereo = True
    else:
        mono = audio.astype(np.float64)
        is_stereo = False

    n_audio = len(mono)
    noise_floor_rms = _estimate_noise_floor(mono, sr)
    
    frame_len = int(0.05 * sr)
    n_frames = n_audio // frame_len
    if n_frames > 5:
        frames = mono[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        quiet_mask = frame_rms <= np.percentile(frame_rms, 15)
        if np.any(quiet_mask):
            quiet_audio = frames[quiet_mask].ravel()
            noise_peak_amp = float(np.percentile(np.abs(quiet_audio), 99))
        else:
            noise_peak_amp = noise_floor_rms * 3
    else:
        noise_peak_amp = noise_floor_rms * 3

    if min_peak_amplitude <= 0:
        min_peak_amplitude = noise_peak_amp * 2.0

    snr_linear = 10 ** (peak_snr_db / 20.0)
    local_energy_threshold = noise_floor_rms * snr_linear

    if verbose:
        print(f"\n  ┌─ Recording Analysis ─────────────────────────")
        print(f"  │ Audio length   : {n_audio/sr:.2f}s")
        print(f"  │ Noise floor RMS: {noise_floor_rms:.6f}")
        print(f"  │ Min peak amp   : {min_peak_amplitude:.6f}")
        print(f"  │ CSV entries    : {len(keystroke_log)}")
        print(f"  └────────────────────────────────────────────────")

    ste_2ms = _short_time_energy(mono, max(1, int(0.002 * sr)))
    ste_10ms = _short_time_energy(mono, max(1, int(0.010 * sr)))
    ste_50ms = _short_time_energy(mono, max(1, int(0.050 * sr)))
    ste_combined = np.cbrt(ste_2ms * ste_10ms * ste_50ms)

    sorted_log = sorted(keystroke_log, key=lambda x: x['relative_time'])
    n_total = len(sorted_log)

    duration_samples = int(segment_duration * sr)
    pre_samples = int(pre_trigger * sr)
    search_radius = int(search_radius_sec * sr)
    overlap_guard = int(overlap_guard_sec * sr)

    segments: List[Dict] = []
    template_bank: List[np.ndarray] = []
    avg_template_env: Optional[np.ndarray] = None
    last_peak_sample = -999999
    rejection_log = []

    stats = {
        'total_csv': n_total,
        'saved': 0,
        'rejected_silent': 0,
        'rejected_low_snr': 0,
        'rejected_low_peak': 0,
        'rejected_low_transient': 0,
        'rejected_template': 0,
        'rejected_overlap': 0,
        'noise_floor_rms': noise_floor_rms,
        'noise_peak_amp': noise_peak_amp,
        'min_peak_amplitude': min_peak_amplitude,
    }

    for idx, entry in enumerate(sorted_log):
        csv_time = entry['relative_time']
        csv_sample = int(csv_time * sr)
        key = entry['key']
        timestamp = entry.get('timestamp', '')

        # CSV-guided search: find peak near CSV time
        lo = max(0, csv_sample - search_radius)
        hi = min(n_audio, csv_sample + search_radius)

        if hi <= lo:
            segments.append(_reject(entry, 'boundary'))
            stats['rejected_silent'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        region_ste = ste_combined[lo:hi]
        energy_peak = lo + int(np.argmax(region_ste))
        refine_r = int(0.005 * sr)
        peak_sample = _find_waveform_peak(mono, energy_peak, refine_r)

        # Overlap guard
        if abs(peak_sample - last_peak_sample) < overlap_guard:
            segments.append(_reject(entry, 'overlap', peak_sample=peak_sample))
            stats['rejected_overlap'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        # Peak amplitude gate
        peak_amp = float(np.abs(mono[peak_sample])) if 0 <= peak_sample < n_audio else 0.0
        tiny_r = int(0.002 * sr)
        plo = max(0, peak_sample - tiny_r)
        phi = min(n_audio, peak_sample + tiny_r)
        peak_amp = float(np.max(np.abs(mono[plo:phi]))) if phi > plo else peak_amp

        if peak_amp < min_peak_amplitude:
            segments.append(_reject(entry, 'low_peak', peak_sample=peak_sample, peak_amp=peak_amp))
            stats['rejected_low_peak'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        # Local SNR gate
        local_rms = _local_peak_rms(mono, peak_sample, sr, window_ms=10.0)

        if local_rms < local_energy_threshold:
            snr_db = 20.0 * np.log10(max(local_rms, 1e-12) / noise_floor_rms)
            segments.append(_reject(entry, 'low_snr', peak_sample=peak_sample, snr_db=snr_db, peak_amp=peak_amp))
            stats['rejected_low_snr'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        snr_db = 20.0 * np.log10(local_rms / noise_floor_rms)

        # Extract segment
        start = max(0, peak_sample - pre_samples)
        end = start + duration_samples
        if end > n_audio:
            end = n_audio
            start = max(0, end - duration_samples)

        seg = mono[start:end]
        if len(seg) < duration_samples:
            seg = np.pad(seg, (0, duration_samples - len(seg)))
        seg = seg[:duration_samples]

        # Crest factor check
        cf = _crest_factor(seg)
        if cf < min_crest_factor:
            segments.append(_reject(entry, 'low_transient', peak_sample=peak_sample, snr_db=snr_db, peak_amp=peak_amp, crest=cf))
            stats['rejected_low_transient'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        # Template verification
        template_corr = 1.0
        if enable_template_verify and avg_template_env is not None:
            s = seg[:len(avg_template_env)]
            if len(s) == len(avg_template_env):
                s_env = _amplitude_envelope(s, sr)
                lag_samples = int(0.015 * sr)
                template_corr = _max_ncc(avg_template_env, s_env, lag_samples)
                if template_corr < template_corr_threshold:
                    segments.append(_reject(entry, 'template', peak_sample=peak_sample, snr_db=snr_db, peak_amp=peak_amp, template_corr=template_corr))
                    stats['rejected_template'] += 1
                    _progress(progress_callback, idx, n_total)
                    continue

        # ACCEPTED
        final_seg = seg.copy()
        if enable_bandpass:
            final_seg = _bandpass(final_seg, sr, bandpass_low, bandpass_high)

        if is_stereo:
            final_seg = np.column_stack([final_seg, final_seg])

        segments.append({
            'audio': final_seg,
            'key': key,
            'timestamp': timestamp,
            'csv_time': csv_time,
            'peak_sample': peak_sample,
            'start_sample': start,
            'end_sample': end,
            'snr_db': snr_db,
            'peak_amp': peak_amp,
            'crest_factor': cf,
            'template_corr': template_corr,
            'time_diff': abs(peak_sample / sr - csv_time),
            'status': 'ok',
        })
        stats['saved'] += 1
        last_peak_sample = peak_sample

        # Build template
        if enable_template_verify and len(template_bank) < template_build_count:
            template_bank.append(seg.copy())
            if len(template_bank) == template_build_count:
                envelopes = []
                for tb in template_bank:
                    env = _amplitude_envelope(tb, sr)
                    pk = np.argmax(env)
                    shift = pre_samples - pk
                    envelopes.append(np.roll(env, shift))
                avg_template_env = np.mean(envelopes, axis=0)

        _progress(progress_callback, idx, n_total)

    if verbose:
        print(f"\n  ┌─ Extraction Results ────────────────────────────")
        print(f"  │ Saved            : {stats['saved']}/{n_total}")
        print(f"  │ Rejected silent  : {stats['rejected_silent']}")
        print(f"  │ Rejected low peak: {stats['rejected_low_peak']}")
        print(f"  │ Rejected low SNR : {stats['rejected_low_snr']}")
        print(f"  │ Rejected crest   : {stats['rejected_low_transient']}")
        print(f"  │ Rejected template: {stats['rejected_template']}")
        print(f"  │ Rejected overlap : {stats['rejected_overlap']}")
        print(f"  └────────────────────────────────────────────────────")

    return segments, stats


def _reject(entry, reason, **extra):
    """Build a rejected-segment record."""
    return {
        'audio': None,
        'key': entry['key'],
        'timestamp': entry.get('timestamp', ''),
        'csv_time': entry['relative_time'],
        'peak_sample': extra.get('peak_sample', 0),
        'start_sample': 0,
        'snr_db': extra.get('snr_db', 0.0),
        'peak_amp': extra.get('peak_amp', 0.0),
        'crest_factor': extra.get('crest', 0.0),
        'template_corr': extra.get('template_corr', 0.0),
        'time_diff': 0.0,
        'status': reason,
    }


def _progress(cb, idx, total):
    if cb:
        cb(idx + 1, total)


# ═══════════════════════════════════════════════════════════════════════════════
#  INTERACTIVE VERIFICATION VIEWER
# ═══════════════════════════════════════════════════════════════════════════════

class InteractiveVerificationViewer:
    """Full-audio verification viewer with zoom, pan, and segment navigation."""

    def __init__(self, parent, audio_data, sr, segments):
        self.window = tk.Toplevel(parent)
        self.window.title("Verification Viewer - Interactive Full Audio")
        self.window.geometry("1400x700")
        
        self.audio_data = audio_data
        self.sr = sr
        self.segments = [s for s in segments if s['status'] == 'ok']
        
        self.fig = None
        self.canvas = None
        self.ax = None
        self.line_csv = None
        self.line_peak = None
        self.spans = []
        self.current_zoom = None
        
        self.build_ui()
        self.draw_full_audio()

    def build_ui(self):
        # Controls frame
        ctrl_frame = tk.Frame(self.window)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Zoom controls
        tk.Label(ctrl_frame, text="Zoom:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl_frame, text="Fit All", command=self.zoom_fit).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="Zoom In", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="Zoom Out", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        
        # Navigation
        tk.Label(ctrl_frame, text="| Navigate:", font=("Arial", 9)).pack(side=tk.LEFT, padx=15)
        tk.Button(ctrl_frame, text="← Left", command=self.pan_left).pack(side=tk.LEFT, padx=2)
        tk.Button(ctrl_frame, text="Right →", command=self.pan_right).pack(side=tk.LEFT, padx=2)
        
        # Segment slider
        tk.Label(ctrl_frame, text="| Segment:", font=("Arial", 9)).pack(side=tk.LEFT, padx=15)
        self.seg_var = tk.IntVar(value=0)
        self.seg_slider = tk.Scale(ctrl_frame, from_=0, to=max(0, len(self.segments)-1),
                                    orient=tk.HORIZONTAL, variable=self.seg_var,
                                    command=self.on_segment_selected, length=200)
        self.seg_slider.pack(side=tk.LEFT, padx=5)
        
        self.seg_label = tk.Label(ctrl_frame, text="0/0", font=("Arial", 9))
        self.seg_label.pack(side=tk.LEFT, padx=5)
        
        # Canvas frame
        canvas_frame = tk.Frame(self.window)
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.fig, self.ax = plt.subplots(figsize=(14, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events for pan
        self.canvas.mpl_connect('button_press_event', self.on_press)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        self.canvas.mpl_connect('motion_notify_event', self.on_motion)
        self.press_x = None

    def draw_full_audio(self):
        """Draw full audio with all segments highlighted."""
        self.ax.clear()
        
        if self.audio_data.ndim == 2:
            mono = np.mean(self.audio_data, axis=1)
        else:
            mono = self.audio_data
        
        time_axis = np.arange(len(mono)) / self.sr
        self.ax.plot(time_axis, mono, 'b-', linewidth=0.5, alpha=0.6, label='Full Audio')
        
        # Draw segments
        colors = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, seg in enumerate(self.segments):
            start_time = seg['start_sample'] / self.sr
            end_time = seg['end_sample'] / self.sr
            self.ax.axvspan(start_time, end_time, alpha=0.2, color=colors[i % 20])
        
        self.ax.set_xlabel('Time (s)', fontsize=11)
        self.ax.set_ylabel('Amplitude', fontsize=11)
        self.ax.set_title('Full Recording with Extracted Segments', fontsize=13, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        self.current_zoom = (0, len(mono) / self.sr)
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.seg_slider.config(to=max(0, len(self.segments)-1))
        self.update_seg_label()

    def on_segment_selected(self, val):
        """Highlight selected segment and zoom to it."""
        idx = int(val)
        if 0 <= idx < len(self.segments):
            seg = self.segments[idx]
            start_time = seg['start_sample'] / self.sr
            end_time = seg['end_sample'] / self.sr
            
            margin = (end_time - start_time) * 0.5
            self.ax.set_xlim(max(0, start_time - margin), end_time + margin)
            self.canvas.draw()
            self.update_seg_label()

    def update_seg_label(self):
        """Update segment counter label."""
        total = len(self.segments)
        idx = self.seg_var.get()
        self.seg_label.config(text=f"{idx+1}/{total}" if total > 0 else "0/0")

    def zoom_fit(self):
        """Zoom to fit all audio."""
        self.ax.set_xlim(0, len(self.audio_data) / self.sr)
        self.canvas.draw()

    def zoom_in(self):
        """Zoom in by 50%."""
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        width = (xlim[1] - xlim[0]) / 3
        self.ax.set_xlim(center - width, center + width)
        self.canvas.draw()

    def zoom_out(self):
        """Zoom out by 50%."""
        xlim = self.ax.get_xlim()
        center = (xlim[0] + xlim[1]) / 2
        width = (xlim[1] - xlim[0]) * 1.5
        self.ax.set_xlim(max(0, center - width), min(len(self.audio_data)/self.sr, center + width))
        self.canvas.draw()

    def pan_left(self):
        """Pan left by 20%."""
        xlim = self.ax.get_xlim()
        width = xlim[1] - xlim[0]
        shift = width * 0.2
        self.ax.set_xlim(max(0, xlim[0] - shift), max(width, xlim[1] - shift))
        self.canvas.draw()

    def pan_right(self):
        """Pan right by 20%."""
        xlim = self.ax.get_xlim()
        width = xlim[1] - xlim[0]
        shift = width * 0.2
        max_time = len(self.audio_data) / self.sr
        self.ax.set_xlim(min(max_time - width, xlim[0] + shift), min(max_time, xlim[1] + shift))
        self.canvas.draw()

    def on_press(self, event):
        """Start pan."""
        if event.inaxes == self.ax:
            self.press_x = event.xdata

    def on_release(self, event):
        """End pan."""
        self.press_x = None

    def on_motion(self, event):
        """Pan during drag."""
        if self.press_x is not None and event.inaxes == self.ax:
            dx = event.xdata - self.press_x
            xlim = self.ax.get_xlim()
            shift = -dx
            max_time = len(self.audio_data) / self.sr
            width = xlim[1] - xlim[0]
            self.ax.set_xlim(max(0, xlim[0] + shift), min(max_time, xlim[1] + shift))
            self.canvas.draw()


# ═══════════════════════════════════════════════════════════════════════════════
#  UI TAB
# ═══════════════════════════════════════════════════════════════════════════════

class DataSegmenterTab:
    """Ultimate keystroke segmenter with preview and verification."""

    def __init__(self, parent, config, audio_handler):
        self.parent = parent
        self.config = config
        self.audio = audio_handler

        self.current_session_path = None
        self.batch_sessions = None
        self.audio_data = None
        self.keystroke_log = []
        self.sample_rate = None
        self._cancel_flag = False
        self.last_segments = None

        # Parameters
        self.segment_duration = tk.DoubleVar(value=0.430)
        self.pre_trigger = tk.DoubleVar(value=0.10)
        self.search_radius = tk.DoubleVar(value=0.25)
        self.peak_snr_db = tk.DoubleVar(value=3.0)
        self.min_crest_factor = tk.DoubleVar(value=1.8)
        self.enable_template = tk.BooleanVar(value=True)
        self.template_corr_thresh = tk.DoubleVar(value=0.20)
        self.enable_filtering = tk.BooleanVar(value=False)
        self.filter_low = tk.IntVar(value=50)
        self.filter_high = tk.IntVar(value=5000)
        self.time_offset = tk.DoubleVar(value=0.0)
        self.output_name_var = tk.StringVar(value=f"segmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

        self.preview_canvas_frame = None
        self.preview_canvas = None

        self.build_ui()

    def _safe_ui(self, func):
        try:
            self.parent.after_idle(func)
        except Exception:
            pass

    def build_ui(self):
        canvas = tk.Canvas(self.parent)
        scrollbar = tk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        scrollable = tk.Frame(canvas)
        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        main = tk.Frame(scrollable, padx=20, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        tk.Label(main, text="Ultimate Keystroke Segmenter v2",
                 font=("Arial", 16, "bold"), fg="#1976D2").pack(pady=(0, 20))

        # 1. Load
        load_f = tk.LabelFrame(main, text="1. Load Session",
                               font=("Arial", 10, "bold"), padx=15, pady=15)
        load_f.pack(fill=tk.X, pady=(0, 15))
        tk.Button(load_f, text="Browse Session Folder", command=self.browse_session,
                  bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.session_label = tk.Label(load_f, text="No session loaded", font=("Arial", 9), fg="gray")
        self.session_label.pack(pady=2)
        self.session_info_label = tk.Label(load_f, text="", font=("Arial", 9), fg="#1976D2")
        self.session_info_label.pack(pady=2)

        # 2. Visual Offset Preview
        preview_f = tk.LabelFrame(main, text="2. Visual Offset Adjustment (Preview First Keystroke)",
                                  font=("Arial", 10, "bold"), padx=15, pady=15)
        preview_f.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        # Preview canvas
        self.preview_canvas_frame = tk.Frame(preview_f, bg='white', height=250)
        self.preview_canvas_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        self.preview_canvas_frame.pack_propagate(False)

        # Offset slider
        offset_row = tk.Frame(preview_f)
        offset_row.pack(fill=tk.X, pady=8)
        tk.Label(offset_row, text="Time Offset (s):", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        offset_slider = tk.Scale(offset_row, from_=-1.0, to=1.0, resolution=0.001,
                                orient=tk.HORIZONTAL, variable=self.time_offset,
                                command=self._on_offset_changed, length=400)
        offset_slider.pack(side=tk.LEFT, padx=10, fill=tk.X, expand=True)

        # Preset buttons
        preset_row = tk.Frame(preview_f)
        preset_row.pack(fill=tk.X, pady=5)
        tk.Label(preset_row, text="Presets:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        for label, offset in [("-50ms", -0.050), ("-10ms", -0.010), ("0ms", 0.0), ("+10ms", 0.010), ("+50ms", 0.050)]:
            tk.Button(preset_row, text=label, width=8,
                     command=lambda o=offset: self._set_offset(o)).pack(side=tk.LEFT, padx=3)

        # 3. Parameters
        param_f = tk.LabelFrame(main, text="2. Segmentation Parameters",
                                font=("Arial", 10, "bold"), padx=15, pady=15)
        param_f.pack(fill=tk.X, pady=(0, 15))

        self._slider(param_f, "Segment duration (s)", self.segment_duration, 0.1, 1.0, 0.01)
        self._slider(param_f, "Pre-trigger (s)", self.pre_trigger, 0.0, 0.3, 0.01)
        self._slider(param_f, "Search radius (s)", self.search_radius, 0.05, 0.50, 0.01)

        ttk.Separator(param_f, orient='horizontal').pack(fill=tk.X, pady=8)
        tk.Label(param_f, text="Quality Gates", font=("Arial", 9, "bold"), fg="#666").pack(anchor=tk.W)

        self._slider(param_f, "Peak SNR (dB)", self.peak_snr_db, 0.0, 15.0, 0.5)
        self._slider(param_f, "Min crest factor", self.min_crest_factor, 1.0, 6.0, 0.1)

        ttk.Separator(param_f, orient='horizontal').pack(fill=tk.X, pady=8)
        tk.Checkbutton(param_f, text="Template verification",
                       variable=self.enable_template, font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=3)
        self._slider(param_f, "Template threshold", self.template_corr_thresh, 0.0, 0.8, 0.05)

        ttk.Separator(param_f, orient='horizontal').pack(fill=tk.X, pady=8)
        tk.Checkbutton(param_f, text="Bandpass filter",
                       variable=self.enable_filtering, command=self._toggle_filter,
                       font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=3)
        filt_row = tk.Frame(param_f)
        filt_row.pack(fill=tk.X, pady=3)
        tk.Label(filt_row, text="Low Hz:").pack(side=tk.LEFT, padx=5)
        self.filter_low_entry = tk.Entry(filt_row, textvariable=self.filter_low, width=7)
        self.filter_low_entry.pack(side=tk.LEFT, padx=3)
        tk.Label(filt_row, text="High Hz:").pack(side=tk.LEFT, padx=10)
        self.filter_high_entry = tk.Entry(filt_row, textvariable=self.filter_high, width=7)
        self.filter_high_entry.pack(side=tk.LEFT, padx=3)

        preset_row = tk.Frame(param_f)
        preset_row.pack(fill=tk.X, pady=3)
        for name, lo, hi in [("50–5kHz", 50, 5000), ("50–3kHz", 50, 3000), ("50–8kHz", 50, 8000)]:
            tk.Button(preset_row, text=name,
                      command=lambda l=lo, h=hi: self._set_filter(l, h)).pack(side=tk.LEFT, padx=3)

        # 3. Output
        out_f = tk.LabelFrame(main, text="3. Output",
                              font=("Arial", 10, "bold"), padx=15, pady=15)
        out_f.pack(fill=tk.X, pady=(0, 15))
        out_row = tk.Frame(out_f)
        out_row.pack(fill=tk.X, pady=5)
        tk.Label(out_row, text="Output folder:", width=14, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        tk.Entry(out_row, textvariable=self.output_name_var, width=40).pack(side=tk.LEFT, padx=5)

        # 4. Process
        proc_f = tk.LabelFrame(main, text="4. Process",
                               font=("Arial", 10, "bold"), padx=15, pady=15)
        proc_f.pack(fill=tk.X, pady=(0, 15))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(proc_f, variable=self.progress_var, maximum=100, length=400)
        self.progress_bar.pack(pady=8)
        self.progress_label = tk.Label(proc_f, text="Ready", font=("Arial", 10), fg="#424242")
        self.progress_label.pack(pady=3)

        btn_row = tk.Frame(proc_f)
        btn_row.pack(pady=8)
        self.process_btn = tk.Button(btn_row, text="Start Segmentation",
                                     command=self.start_segmentation,
                                     bg="#4CAF50", fg="white",
                                     font=("Arial", 11, "bold"),
                                     width=18, height=2, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=10)
        tk.Button(btn_row, text="Cancel", command=self._cancel,
                 font=("Arial", 11), width=12, height=2).pack(side=tk.LEFT, padx=10)

        # 5. Results
        res_f = tk.LabelFrame(main, text="5. Results & Verification",
                              font=("Arial", 10, "bold"), padx=15, pady=15)
        res_f.pack(fill=tk.BOTH, expand=True)
        self.results_text = tk.Text(res_f, height=12, font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

        self.verify_btn = tk.Button(res_f, text="🔍 View Interactive Full Audio with Segments",
                                    command=self.view_verification,
                                    bg="#FF9800", fg="white", font=("Arial", 10, "bold"),
                                    state=tk.DISABLED)
        self.verify_btn.pack(pady=8)

    def _slider(self, parent, label, var, lo, hi, res):
        row = tk.Frame(parent)
        row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, width=28, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        tk.Scale(row, from_=lo, to=hi, resolution=res, orient=tk.HORIZONTAL,
                 variable=var, length=200).pack(side=tk.LEFT, padx=5)

    def _toggle_filter(self):
        st = tk.NORMAL if self.enable_filtering.get() else tk.DISABLED
        self.filter_low_entry.config(state=st)
        self.filter_high_entry.config(state=st)

    def _set_filter(self, lo, hi):
        self.filter_low.set(lo)
        self.filter_high.set(hi)
        self.enable_filtering.set(True)
        self._toggle_filter()

    def _set_offset(self, offset):
        """Set offset to preset value."""
        self.time_offset.set(offset)
        self._redraw_preview()

    def _on_offset_changed(self, val):
        """Called when offset slider changes."""
        self._redraw_preview()

    def _redraw_preview(self):
        """Draw waveform preview with live offset."""
        if self.audio_data is None or not self.keystroke_log or not HAS_MATPLOTLIB:
            return

        try:
            first = self.keystroke_log[0]
            csv_time = first['relative_time']
            adjusted_time = csv_time + self.time_offset.get()
            duration = self.segment_duration.get()

            adjusted_sample = int(adjusted_time * self.sample_rate)
            start_sample = adjusted_sample
            end_sample = start_sample + int(duration * self.sample_rate)

            # Context window (±1s around adjusted time)
            context_samples = int(1.0 * self.sample_rate)
            view_start = max(0, adjusted_sample - context_samples)
            view_end = min(len(self.audio_data), adjusted_sample + context_samples)
            audio_view = self.audio_data[view_start:view_end]
            
            if audio_view.ndim == 2:
                audio_view = audio_view.mean(axis=1)

            # Clear previous canvas
            for widget in self.preview_canvas_frame.winfo_children():
                widget.destroy()

            fig, ax = plt.subplots(figsize=(12, 2.5))
            time_axis = (np.arange(len(audio_view)) + view_start) / self.sample_rate
            ax.plot(time_axis, audio_view, 'b-', linewidth=0.7, alpha=0.7, label='Audio')
            
            # Extraction window
            if 0 <= start_sample < end_sample <= len(self.audio_data):
                start_time = start_sample / self.sample_rate
                end_time = end_sample / self.sample_rate
                ax.axvspan(start_time, end_time, alpha=0.3, color='green', label='Extraction Window')
            
            # CSV time vs adjusted time
            ax.axvline(csv_time, color='orange', linestyle='--', linewidth=2, label='CSV Time')
            ax.axvline(adjusted_time, color='red', linestyle='-', linewidth=2.5, label='Adjusted Time')
            
            ax.set_xlabel('Time (s)', fontsize=9)
            ax.set_ylabel('Amplitude', fontsize=9)
            status = '✓ OK' if 0 <= start_sample < end_sample <= len(self.audio_data) else '✗ OUT OF BOUNDS'
            ax.set_title(f"{first['key']} | Offset: {self.time_offset.get():+.3f}s | {status}", 
                        fontsize=10, fontweight='bold')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.tight_layout()

            canvas = FigureCanvasTkAgg(fig, master=self.preview_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.preview_canvas = canvas
        except Exception as e:
            print(f"Preview error: {e}")

    def _cancel(self):
        self._cancel_flag = True

    def browse_session(self):
        folder = filedialog.askdirectory(title="Select Session Folder (or Parent for Batch)")
        if not folder:
            return
        af = os.path.join(folder, 'audio.wav')
        lf = os.path.join(folder, 'keystroke_log.csv')
        if os.path.exists(af) and os.path.exists(lf):
            self._load_single(folder)
        else:
            sessions = self._find_sessions(folder)
            if not sessions:
                messagebox.showerror("Error",
                    "No valid sessions found.\nFolder must contain audio.wav + keystroke_log.csv,\n"
                    "or subfolders that do.")
                return
            msg = f"Found {len(sessions)} session(s):\n\n"
            msg += "\n".join(f"  • {os.path.basename(s)}" for s in sessions[:10])
            if len(sessions) > 10:
                msg += f"\n  … and {len(sessions)-10} more"
            msg += "\n\nProcess all?"
            if messagebox.askyesno("Batch", msg):
                self._load_batch(sessions)

    def _find_sessions(self, root, max_depth=3):
        found = []
        def _walk(d, depth):
            if depth > max_depth:
                return
            try:
                if os.path.exists(os.path.join(d, 'audio.wav')) and \
                   os.path.exists(os.path.join(d, 'keystroke_log.csv')):
                    found.append(d)
                    return
                for item in sorted(os.listdir(d)):
                    p = os.path.join(d, item)
                    if os.path.isdir(p):
                        _walk(p, depth + 1)
            except (PermissionError, OSError):
                pass
        _walk(root, 0)
        return sorted(found)

    def _load_single(self, folder):
        try:
            self.current_session_path = folder
            self.batch_sessions = None
            self.audio_data, self.sample_rate = self.audio.load_audio(
                os.path.join(folder, 'audio.wav'))
            if self.audio_data is None:
                raise RuntimeError("load_audio returned None")
            with open(os.path.join(folder, 'keystroke_log.csv'), 'r') as f:
                self.keystroke_log = []
                for row in csv.DictReader(f):
                    row['relative_time'] = float(row['relative_time'])
                    self.keystroke_log.append(row)
            dur = len(self.audio_data) / self.sample_rate
            self.session_label.config(text=f"Loaded: {os.path.basename(folder)}", fg="green")
            self.session_info_label.config(
                text=f"Audio: {dur:.1f}s @ {self.sample_rate} Hz | Keystrokes: {len(self.keystroke_log)}")
            self.process_btn.config(state=tk.NORMAL)
            self._redraw_preview()
            self.verify_btn.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load: {e}")
            self.current_session_path = None

    def _load_batch(self, sessions):
        self.current_session_path = None
        self.batch_sessions = sessions
        self.audio_data = None
        self.keystroke_log = []
        total = 0
        for s in sessions:
            try:
                total += len(self._read_log(os.path.join(s, 'keystroke_log.csv')))
            except Exception:
                pass
        self.session_label.config(text=f"Batch: {len(sessions)} sessions", fg="green")
        self.session_info_label.config(text=f"≈ {total} keystrokes total")
        self.process_btn.config(state=tk.NORMAL)

    @staticmethod
    def _read_log(path):
        log = []
        with open(path, 'r') as f:
            for row in csv.DictReader(f):
                row['relative_time'] = float(row['relative_time'])
                log.append(row)
        return log

    def _snapshot_params(self) -> Dict:
        return {
            'output_name': self.output_name_var.get(),
            'segment_duration': self.segment_duration.get(),
            'pre_trigger': self.pre_trigger.get(),
            'search_radius': self.search_radius.get(),
            'peak_snr_db': self.peak_snr_db.get(),
            'min_crest_factor': self.min_crest_factor.get(),
            'enable_template': self.enable_template.get(),
            'template_corr': self.template_corr_thresh.get(),
            'enable_filtering': self.enable_filtering.get(),
            'filter_low': self.filter_low.get(),
            'filter_high': self.filter_high.get(),
        }

    def start_segmentation(self):
        if self.batch_sessions is None and (self.audio_data is None or not self.keystroke_log):
            messagebox.showwarning("Warning", "No session loaded")
            return
        self._cancel_flag = False
        params = self._snapshot_params()
        self.process_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)

        if self.batch_sessions:
            threading.Thread(target=self._run_batch, args=(params,), daemon=True).start()
        else:
            threading.Thread(target=self._run_single, args=(params,), daemon=True).start()

    def _run_single(self, params):
        try:
            output_base = os.path.join(os.path.dirname(self.current_session_path),
                                        params['output_name'])
            os.makedirs(output_base, exist_ok=True)

            def _prog(i, n):
                if self._cancel_flag:
                    raise InterruptedError("Cancelled")
                pct = i / n * 100
                self._safe_ui(lambda p=pct: self.progress_var.set(p))
                self._safe_ui(lambda a=i, b=n: self.progress_label.config(text=f"{a}/{b}"))

            segments, stats = extract_keystrokes(
                self.audio_data, self.sample_rate, self.keystroke_log,
                segment_duration=params['segment_duration'],
                pre_trigger=params['pre_trigger'],
                search_radius_sec=params['search_radius'],
                peak_snr_db=params['peak_snr_db'],
                min_crest_factor=params['min_crest_factor'],
                enable_template_verify=params['enable_template'],
                template_corr_threshold=params['template_corr'],
                enable_bandpass=params['enable_filtering'],
                bandpass_low=params['filter_low'],
                bandpass_high=params['filter_high'],
                progress_callback=_prog,
            )

            key_counts, metadata = self._save_segments(segments, output_base)
            self._save_metadata(metadata, output_base)
            self._save_rejection_report(segments, output_base)

            self.last_segments = segments

            results = self._format_results(stats, key_counts, output_base, self.sample_rate)
            self._safe_ui(lambda: self.results_text.insert(1.0, results))
            self._safe_ui(lambda: self.progress_var.set(100))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: self.verify_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: messagebox.showinfo("Done", f"Saved {stats['saved']} segments"))

        except InterruptedError:
            self._safe_ui(lambda: self.progress_label.config(text="Cancelled"))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._safe_ui(lambda: self.progress_label.config(text=f"Error: {e}"))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: messagebox.showerror("Error", str(e)))

    def _run_batch(self, params):
        try:
            first = self.batch_sessions[0]
            current = first
            recordings_root = None
            for _ in range(10):
                parent = os.path.dirname(current)
                if os.path.basename(current) in ('recordings', 'backups'):
                    recordings_root = current
                    break
                if parent == current:
                    break
                current = parent

            if recordings_root:
                rel = os.path.relpath(first, recordings_root)
                parts = rel.split(os.sep)
                device = "_".join(parts[:-1]) if len(parts) >= 2 else ""
                name = f"{device}_{params['output_name']}" if device else params['output_name']
                output_base = os.path.join(recordings_root, 'segmented', name)
            else:
                output_base = os.path.join(os.path.dirname(first), params['output_name'])

            os.makedirs(output_base, exist_ok=True)
            n_sessions = len(self.batch_sessions)

            overall = {
                'total_sessions': n_sessions, 'processed': 0, 'failed': 0,
                'total_csv': 0, 'total_saved': 0, 'key_counts': {},
                'details': [],
            }

            for si, sfolder in enumerate(self.batch_sessions):
                if self._cancel_flag:
                    break

                sname = os.path.basename(sfolder)
                pct = si / n_sessions * 100
                self._safe_ui(lambda p=pct: self.progress_var.set(p))
                self._safe_ui(lambda a=si+1, b=n_sessions, n=sname:
                              self.progress_label.config(text=f"Session {a}/{b}: {n}"))

                try:
                    audio_data, sr = self.audio.load_audio(os.path.join(sfolder, 'audio.wav'))
                    if audio_data is None:
                        raise RuntimeError("load_audio returned None")
                    log = self._read_log(os.path.join(sfolder, 'keystroke_log.csv'))

                    print(f"\n[{si+1}/{n_sessions}] {sname}: {len(log)} keystrokes, {len(audio_data)/sr:.1f}s")

                    segments, stats = extract_keystrokes(
                        audio_data, sr, log,
                        segment_duration=params['segment_duration'],
                        pre_trigger=params['pre_trigger'],
                        search_radius_sec=params['search_radius'],
                        peak_snr_db=params['peak_snr_db'],
                        min_crest_factor=params['min_crest_factor'],
                        enable_template_verify=params['enable_template'],
                        template_corr_threshold=params['template_corr'],
                        enable_bandpass=params['enable_filtering'],
                        bandpass_low=params['filter_low'],
                        bandpass_high=params['filter_high'],
                    )

                    key_counts, _ = self._save_segments(segments, output_base)

                    overall['processed'] += 1
                    overall['total_csv'] += stats['total_csv']
                    overall['total_saved'] += stats['saved']
                    for k, c in key_counts.items():
                        overall['key_counts'][k] = overall['key_counts'].get(k, 0) + c
                    overall['details'].append({'name': sname, 'stats': stats, 'ok': True})

                except Exception as e:
                    import traceback; traceback.print_exc()
                    overall['failed'] += 1
                    overall['details'].append({'name': sname, 'ok': False, 'error': str(e)})

            self._write_batch_summary(overall, output_base)
            results = self._format_batch_results(overall, output_base)
            self._safe_ui(lambda: self.results_text.insert(1.0, results))
            self._safe_ui(lambda: self.progress_var.set(100))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: messagebox.showinfo("Batch Done",
                f"{overall['processed']} sessions, {overall['total_saved']} segments saved"))

        except Exception as e:
            import traceback; traceback.print_exc()
            self._safe_ui(lambda: self.progress_label.config(text=f"Error: {e}"))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))

    def view_verification(self):
        """Show interactive verification viewer."""
        if not HAS_MATPLOTLIB or self.last_segments is None or self.audio_data is None:
            messagebox.showerror("Error", "Matplotlib required or no extraction data available")
            return
        InteractiveVerificationViewer(self.parent, self.audio_data, self.sample_rate, self.last_segments)

    def _save_segments(self, segments, output_base):
        key_counts = {}
        metadata = []
        for seg in segments:
            if seg['status'] != 'ok':
                continue
            key = seg['key']
            key_folder = os.path.join(output_base, key)
            os.makedirs(key_folder, exist_ok=True)

            existing = [f for f in os.listdir(key_folder) if f.endswith('.wav')]
            fnum = len(existing)
            fname = f"{fnum}.wav"
            fpath = os.path.join(key_folder, fname)

            if self.audio.save_audio(fpath, seg['audio']):
                key_counts[key] = key_counts.get(key, 0) + 1
                metadata.append({
                    'key': key, 'filename': fname,
                    'timestamp': seg['timestamp'],
                    'csv_time': f"{seg['csv_time']:.4f}",
                    'peak_sample': seg['peak_sample'],
                    'start_sample': seg['start_sample'],
                    'snr_db': f"{seg['snr_db']:.1f}",
                    'peak_amp': f"{seg['peak_amp']:.6f}",
                    'crest_factor': f"{seg['crest_factor']:.2f}",
                    'template_corr': f"{seg['template_corr']:.3f}",
                })
        return key_counts, metadata

    @staticmethod
    def _save_metadata(metadata, output_base):
        if not metadata:
            return
        path = os.path.join(output_base, 'metadata.csv')
        fields = list(metadata[0].keys())
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(metadata)

    @staticmethod
    def _save_rejection_report(segments, output_base):
        rejected = [s for s in segments if s['status'] != 'ok']
        if not rejected:
            return
        path = os.path.join(output_base, 'rejections.csv')
        fields = ['key', 'csv_time', 'status', 'peak_sample', 'snr_db', 'peak_amp', 'crest_factor', 'template_corr']
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for s in rejected:
                row = {k: s.get(k, '') for k in fields}
                for fk in ['csv_time', 'snr_db', 'peak_amp', 'crest_factor', 'template_corr']:
                    if isinstance(row[fk], float):
                        row[fk] = f"{row[fk]:.4f}"
                w.writerow(row)

    @staticmethod
    def _format_results(stats, key_counts, output_base, sr):
        saved = stats['saved']
        total = stats['total_csv']
        txt = f"""SEGMENTATION COMPLETE
{'='*55}
CSV keystrokes:      {total}
Saved:               {saved}  ({100*saved/max(1,total):.1f}%)

Rejection breakdown:
  Silent / boundary: {stats['rejected_silent']}
  Low peak amplitude:{stats['rejected_low_peak']}
  Low local SNR:     {stats['rejected_low_snr']}
  Low crest factor:  {stats['rejected_low_transient']}
  Template mismatch: {stats['rejected_template']}
  Overlap:           {stats['rejected_overlap']}

Key distribution:
"""
        for k in sorted(key_counts):
            txt += f"  {k:12s}: {key_counts[k]:4d}\n"
        txt += f"\nOutput: {output_base}\n"
        return txt

    @staticmethod
    def _format_batch_results(overall, output_base):
        txt = f"""BATCH SEGMENTATION COMPLETE
{'='*55}
Sessions:  {overall['processed']}/{overall['total_sessions']}  (failed: {overall['failed']})
Saved:     {overall['total_saved']}/{overall['total_csv']}  ({100*overall['total_saved']/max(1,overall['total_csv']):.1f}%)

Key distribution:
"""
        for k in sorted(overall['key_counts']):
            txt += f"  {k:12s}: {overall['key_counts'][k]:4d}\n"
        txt += f"\nOutput: {output_base}\n"
        return txt

    @staticmethod
    def _write_batch_summary(overall, output_base):
        path = os.path.join(output_base, 'batch_summary.txt')
        with open(path, 'w') as f:
            f.write(f"Batch: {overall['processed']}/{overall['total_sessions']} sessions\n")
            f.write(f"Saved: {overall['total_saved']}/{overall['total_csv']}\n\n")
            for d in overall['details']:
                if d['ok']:
                    s = d['stats']
                    f.write(f"{d['name']}: {s['saved']}/{s['total_csv']} saved\n")
                else:
                    f.write(f"{d['name']}: FAILED — {d['error']}\n")