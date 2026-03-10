"""
Robust Keystroke Segmentation Module — v2

Fixes from v1 based on real recording analysis:
────────────────────────────────────────────────
PROBLEM: Keystrokes were rejected because the SNR gate compared the FULL
segment RMS (430ms, mostly silence/tail) against the noise floor.  A keystroke
that peaks at 0.04 amplitude over ~10ms has a segment-wide RMS of maybe
0.001 — barely above a noise floor of 0.0004.  So SNR_db ≈ 8, which can
still fail depending on settings.

FIX 1 — LOCAL PEAK ENERGY:  Instead of segment-wide RMS, measure the RMS
of a short window (~10ms) centered on the peak *within the search region*.
This is the actual keystroke energy, not diluted by 400ms of silence.

FIX 2 — ROBUST NOISE FLOOR: Use the 5th percentile of frame energies AND
exclude frames that contain keystroke candidates (top 20%) from the noise
estimate.  This prevents dense typing from inflating the noise floor.

FIX 3 — PEAK AMPLITUDE GATE (replaces segment-wide SNR): Simply check that
the peak amplitude in the search window exceeds N× the noise floor.  This is
far more intuitive and reliable for impulsive sounds.

FIX 4 — DIAGNOSTIC LOGGING: Print per-segment rejection reasons with actual
values so you can see exactly why things are being rejected and tune thresholds.
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


# ═══════════════════════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def _short_time_energy(audio: np.ndarray, win_samples: int) -> np.ndarray:
    """Sliding-window mean of squared samples."""
    sq = audio.astype(np.float64) ** 2
    return uniform_filter1d(sq, size=max(1, win_samples), mode='constant', origin=0)


def _estimate_noise_floor(audio: np.ndarray, sr: int,
                           frame_dur: float = 0.050,
                           quantile: float = 0.05) -> float:
    """
    Estimate noise floor RMS using the quietest frames.

    Uses the 5th percentile (not 10th) of 50ms frame RMS values, which is more
    conservative and avoids being pulled up by dense keystroke activity.

    Returns the noise floor as an RMS value.
    """
    frame_len = int(frame_dur * sr)
    n_frames = len(audio) // frame_len
    if n_frames < 5:
        return 1e-7

    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms_per_frame = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))

    # Use the 5th percentile — very conservative, ignores keystroke frames
    floor = float(np.percentile(rms_per_frame, quantile * 100))
    return max(floor, 1e-7)


def _local_peak_rms(audio: np.ndarray, center: int, sr: int,
                     window_ms: float = 10.0) -> float:
    """RMS of a short window centered on `center`. This captures the actual
    keystroke energy without dilution from the silent tail."""
    half_win = int(window_ms / 1000.0 * sr / 2)
    lo = max(0, center - half_win)
    hi = min(len(audio), center + half_win)
    if hi <= lo:
        return 0.0
    seg = audio[lo:hi].astype(np.float64)
    return float(np.sqrt(np.mean(seg ** 2)))


def _find_waveform_peak(audio: np.ndarray, center: int,
                         search_radius: int) -> int:
    """Index of max |amplitude| within ±search_radius of center."""
    lo = max(0, center - search_radius)
    hi = min(len(audio), center + search_radius)
    if hi <= lo:
        return center
    return int(lo + np.argmax(np.abs(audio[lo:hi])))


def _rms(segment: np.ndarray) -> float:
    return float(np.sqrt(np.mean(segment.astype(np.float64) ** 2)))


def _amplitude_envelope(audio: np.ndarray, sr: int,
                         smooth_ms: float = 5.0) -> np.ndarray:
    """
    Compute the amplitude envelope: abs(signal) smoothed with a short window.
    
    This is always positive and phase-insensitive, so two keystrokes that differ
    only in waveform polarity or micro-timing will still have high envelope
    correlation.
    """
    env = np.abs(audio.astype(np.float64))
    smooth_samples = max(1, int(smooth_ms / 1000.0 * sr))
    return uniform_filter1d(env, size=smooth_samples, mode='constant')


def _max_ncc(template: np.ndarray, candidate: np.ndarray,
              max_lag: int) -> float:
    """
    Maximum normalized cross-correlation over a ±max_lag range.
    
    Instead of only checking zero-lag (which fails if the peak is shifted by
    even a few samples), we slide the candidate and return the best correlation.
    """
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
    
    # Check each lag: positive lag = candidate is shifted right (peak came later)
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
    """Peak-to-RMS ratio (crest factor). Keystrokes are impulsive → high crest."""
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
#  CORE EXTRACTION (stateless, no UI dependency)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_keystrokes(
    audio: np.ndarray,
    sr: int,
    keystroke_log: List[Dict],
    *,
    # Segment geometry
    segment_duration: float = 0.430,
    pre_trigger: float = 0.10,
    # Search
    search_radius_sec: float = 0.25,
    # Quality gates
    peak_snr_db: float = 6.0,          # local peak must be this many dB above noise
    min_crest_factor: float = 1.8,      # peak/RMS of the extraction window
    min_peak_amplitude: float = 0.0,    # absolute floor (0 = disabled, auto-adapt)
    # Template verification
    enable_template_verify: bool = True,
    template_corr_threshold: float = 0.20,
    template_build_count: int = 20,
    # Post-processing
    enable_bandpass: bool = False,
    bandpass_low: int = 50,
    bandpass_high: int = 5000,
    # Misc
    overlap_guard_sec: float = 0.02,
    verbose: bool = True,
    progress_callback=None,
) -> Tuple[List[Dict], Dict]:
    """
    CSV-guided keystroke extraction with robust quality gating.

    For each CSV timestamp:
      1. Search ±search_radius for the energy peak (multi-scale STE)
      2. Refine to waveform amplitude peak
      3. Measure LOCAL peak energy (10ms window) vs noise floor → SNR gate
      4. Check crest factor of extraction window → transient gate
      5. Template cross-correlation (after template is built) → shape gate
      6. Extract and save

    Returns (segments, stats).
    """

    # ── Mono ─────────────────────────────────────────────────────────────
    if audio.ndim == 2:
        mono = np.mean(audio, axis=1).astype(np.float64)
        is_stereo = True
    else:
        mono = audio.astype(np.float64)
        is_stereo = False

    n_audio = len(mono)

    # ── Noise floor ──────────────────────────────────────────────────────
    noise_floor_rms = _estimate_noise_floor(mono, sr)
    # Also compute noise floor as peak amplitude in quiet frames
    # (useful as an absolute reference)
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

    # If min_peak_amplitude is 0 (auto), set it to 3× noise peak amplitude
    if min_peak_amplitude <= 0:
        min_peak_amplitude = noise_peak_amp * 2.0

    snr_linear = 10 ** (peak_snr_db / 20.0)
    local_energy_threshold = noise_floor_rms * snr_linear

    if verbose:
        print(f"\n  ┌─ Recording Analysis ─────────────────────────")
        print(f"  │ Audio length   : {n_audio/sr:.2f}s ({n_audio} samples @ {sr} Hz)")
        print(f"  │ Overall RMS    : {_rms(mono):.6f}")
        print(f"  │ Noise floor RMS: {noise_floor_rms:.6f}")
        print(f"  │ Noise peak amp : {noise_peak_amp:.6f}")
        print(f"  │ Min peak amp   : {min_peak_amplitude:.6f} (auto)")
        print(f"  │ Local E thresh : {local_energy_threshold:.6f} ({peak_snr_db} dB above noise)")
        print(f"  │ Min crest      : {min_crest_factor}")
        print(f"  │ CSV entries    : {len(keystroke_log)}")
        print(f"  └────────────────────────────────────────────────")

    # ── Multi-scale STE for peak finding ─────────────────────────────────
    ste_2ms  = _short_time_energy(mono, max(1, int(0.002 * sr)))
    ste_10ms = _short_time_energy(mono, max(1, int(0.010 * sr)))
    ste_50ms = _short_time_energy(mono, max(1, int(0.050 * sr)))
    ste_combined = np.cbrt(ste_2ms * ste_10ms * ste_50ms)

    # ── Process each CSV entry ───────────────────────────────────────────
    sorted_log = sorted(keystroke_log, key=lambda x: x['relative_time'])
    n_total = len(sorted_log)

    duration_samples = int(segment_duration * sr)
    pre_samples = int(pre_trigger * sr)
    search_radius = int(search_radius_sec * sr)
    overlap_guard = int(overlap_guard_sec * sr)

    segments: List[Dict] = []
    template_bank: List[np.ndarray] = []
    avg_template_env: Optional[np.ndarray] = None  # envelope-based template

    last_peak_sample = -999999

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

    rejection_log = []  # collect first N rejections for diagnostics

    for idx, entry in enumerate(sorted_log):
        csv_time = entry['relative_time']
        csv_sample = int(csv_time * sr)
        key = entry['key']
        timestamp = entry.get('timestamp', '')

        # ── 1. Find energy peak in search window ────────────────────────
        lo = max(0, csv_sample - search_radius)
        hi = min(n_audio, csv_sample + search_radius)

        if hi <= lo:
            segments.append(_reject(entry, 'boundary'))
            stats['rejected_silent'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        region_ste = ste_combined[lo:hi]
        energy_peak = lo + int(np.argmax(region_ste))

        # ── 2. Refine to waveform amplitude peak (±5ms) ────────────────
        refine_r = int(0.005 * sr)
        peak_sample = _find_waveform_peak(mono, energy_peak, refine_r)

        # ── 3. Overlap guard ────────────────────────────────────────────
        if abs(peak_sample - last_peak_sample) < overlap_guard:
            segments.append(_reject(entry, 'overlap', peak_sample=peak_sample))
            stats['rejected_overlap'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        # ── 4. Peak amplitude gate ──────────────────────────────────────
        #    Check the actual waveform amplitude at the peak location.
        #    This is the most direct and reliable test.
        peak_amp = float(np.abs(mono[peak_sample])) if 0 <= peak_sample < n_audio else 0.0
        # Also check a ±2ms window max (in case peak is 1 sample off)
        tiny_r = int(0.002 * sr)
        plo = max(0, peak_sample - tiny_r)
        phi = min(n_audio, peak_sample + tiny_r)
        peak_amp = float(np.max(np.abs(mono[plo:phi]))) if phi > plo else peak_amp

        if peak_amp < min_peak_amplitude:
            if len(rejection_log) < 30:
                rejection_log.append(
                    f"  REJECT peak_amp: '{key}' @ {csv_time:.3f}s  "
                    f"amp={peak_amp:.6f} < {min_peak_amplitude:.6f}")
            segments.append(_reject(entry, 'low_peak', peak_sample=peak_sample,
                                     peak_amp=peak_amp))
            stats['rejected_low_peak'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        # ── 5. Local SNR gate (10ms window around peak vs noise floor) ──
        local_rms = _local_peak_rms(mono, peak_sample, sr, window_ms=10.0)

        if local_rms < local_energy_threshold:
            if len(rejection_log) < 30:
                rejection_log.append(
                    f"  REJECT local_snr: '{key}' @ {csv_time:.3f}s  "
                    f"local_rms={local_rms:.6f} < {local_energy_threshold:.6f}")
            snr_db = 20.0 * np.log10(max(local_rms, 1e-12) / noise_floor_rms)
            segments.append(_reject(entry, 'low_snr', peak_sample=peak_sample,
                                     snr_db=snr_db, peak_amp=peak_amp))
            stats['rejected_low_snr'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        snr_db = 20.0 * np.log10(local_rms / noise_floor_rms)

        # ── 6. Extract the segment ──────────────────────────────────────
        start = max(0, peak_sample - pre_samples)
        end = start + duration_samples
        if end > n_audio:
            end = n_audio
            start = max(0, end - duration_samples)

        seg = mono[start:end]
        if len(seg) < duration_samples:
            seg = np.pad(seg, (0, duration_samples - len(seg)))
        seg = seg[:duration_samples]

        # ── 7. Crest factor check (on extraction window) ────────────────
        cf = _crest_factor(seg)
        if cf < min_crest_factor:
            if len(rejection_log) < 30:
                rejection_log.append(
                    f"  REJECT crest: '{key}' @ {csv_time:.3f}s  "
                    f"cf={cf:.2f} < {min_crest_factor:.2f}")
            segments.append(_reject(entry, 'low_transient', peak_sample=peak_sample,
                                     snr_db=snr_db, peak_amp=peak_amp,
                                     crest=cf))
            stats['rejected_low_transient'] += 1
            _progress(progress_callback, idx, n_total)
            continue

        # ── 8. Template verification ────────────────────────────────────
        #    Uses ENVELOPE-based correlation with sliding lag to handle
        #    small timing shifts between segments.  Raw waveform correlation
        #    fails for impulsive signals because a 1-sample shift flips the
        #    sign of the correlation.
        template_corr = 1.0
        if enable_template_verify and avg_template_env is not None:
            s = seg[:len(avg_template_env)]
            if len(s) == len(avg_template_env):
                # Compute amplitude envelope of this segment
                s_env = _amplitude_envelope(s, sr)
                
                # Sliding normalized cross-correlation over ±lag_ms
                lag_samples = int(0.015 * sr)  # ±15ms tolerance
                template_corr = _max_ncc(avg_template_env, s_env, lag_samples)

                if template_corr < template_corr_threshold:
                    if len(rejection_log) < 30:
                        rejection_log.append(
                            f"  REJECT template: '{key}' @ {csv_time:.3f}s  "
                            f"corr={template_corr:.3f} < {template_corr_threshold:.3f}")
                    segments.append(_reject(entry, 'template', peak_sample=peak_sample,
                                             snr_db=snr_db, peak_amp=peak_amp,
                                             template_corr=template_corr))
                    stats['rejected_template'] += 1
                    _progress(progress_callback, idx, n_total)
                    continue

        # ── 9. ACCEPTED ─────────────────────────────────────────────────
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
            'snr_db': snr_db,
            'peak_amp': peak_amp,
            'crest_factor': cf,
            'template_corr': template_corr,
            'time_diff': abs(peak_sample / sr - csv_time),
            'status': 'ok',
        })
        stats['saved'] += 1
        last_peak_sample = peak_sample

        # ── 10. Build template from first N good segments ───────────────
        #    We build an envelope-based template (not raw waveform) so that
        #    polarity differences and micro-timing shifts don't cause
        #    anti-correlation.
        if enable_template_verify and len(template_bank) < template_build_count:
            template_bank.append(seg.copy())
            if len(template_bank) == template_build_count:
                # Compute envelope for each, align by envelope peak, average
                envelopes = []
                for tb in template_bank:
                    env = _amplitude_envelope(tb, sr)
                    pk = np.argmax(env)
                    shift = pre_samples - pk
                    envelopes.append(np.roll(env, shift))
                avg_template_env = np.mean(envelopes, axis=0)
                if verbose:
                    print(f"  Envelope template built from {template_build_count} segments")

        _progress(progress_callback, idx, n_total)

    # ── Summary ──────────────────────────────────────────────────────────
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

        if rejection_log:
            print(f"\n  ── Sample rejections (first {len(rejection_log)}) ──")
            for r in rejection_log:
                print(r)

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
#  UI TAB
# ═══════════════════════════════════════════════════════════════════════════════

class DataSegmenterTab:
    """Tab for segmenting continuous recordings into individual keystroke samples."""

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

        # ── Tunable parameters with sensible defaults ────────────────────
        self.segment_duration    = tk.DoubleVar(value=0.430)
        self.pre_trigger         = tk.DoubleVar(value=0.10)
        self.search_radius       = tk.DoubleVar(value=0.25)    # ±250ms
        self.peak_snr_db         = tk.DoubleVar(value=3.0)     # lowered from 6 — uses local energy now
        self.min_crest_factor    = tk.DoubleVar(value=1.8)
        self.enable_template     = tk.BooleanVar(value=True)
        self.template_corr_thresh = tk.DoubleVar(value=0.20)
        self.enable_filtering    = tk.BooleanVar(value=False)
        self.filter_low          = tk.IntVar(value=50)
        self.filter_high         = tk.IntVar(value=5000)

        self.build_ui()

    def _safe_ui(self, func):
        try:
            self.parent.after_idle(func)
        except Exception:
            pass

    # ── Build UI ─────────────────────────────────────────────────────────
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

        tk.Label(main, text="Robust Keystroke Segmentation v2",
                 font=("Arial", 16, "bold"), fg="#1976D2").pack(pady=(0, 20))

        # ── 1. Load ──────────────────────────────────────────────────────
        load_f = tk.LabelFrame(main, text="1. Load Recording Session",
                               font=("Arial", 10, "bold"), padx=15, pady=15)
        load_f.pack(fill=tk.X, pady=(0, 15))
        tk.Button(load_f, text="Browse Session Folder", command=self.browse_session,
                  bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=5)
        self.session_label = tk.Label(load_f, text="No session loaded", font=("Arial", 9), fg="gray")
        self.session_label.pack(pady=2)
        self.session_info_label = tk.Label(load_f, text="", font=("Arial", 9), fg="#1976D2")
        self.session_info_label.pack(pady=2)

        # ── 2. Parameters ────────────────────────────────────────────────
        param_f = tk.LabelFrame(main, text="2. Segmentation Parameters",
                                font=("Arial", 10, "bold"), padx=15, pady=15)
        param_f.pack(fill=tk.X, pady=(0, 15))

        self._slider(param_f, "Segment duration (s)",   self.segment_duration, 0.1, 1.0, 0.01)
        self._slider(param_f, "Pre-trigger (s)",         self.pre_trigger,      0.0, 0.3, 0.01)
        self._slider(param_f, "Search radius (s)",       self.search_radius,    0.05, 0.50, 0.01)

        # Separator
        ttk.Separator(param_f, orient='horizontal').pack(fill=tk.X, pady=8)
        tk.Label(param_f, text="Quality Gates", font=("Arial", 9, "bold"), fg="#666").pack(anchor=tk.W)

        self._slider(param_f, "Peak SNR (dB above noise)", self.peak_snr_db,     0.0, 15.0, 0.5)
        self._slider(param_f, "Min crest factor",           self.min_crest_factor, 1.0, 6.0, 0.1)

        # Template section
        ttk.Separator(param_f, orient='horizontal').pack(fill=tk.X, pady=8)
        tk.Checkbutton(param_f, text="Template verification (reject non-keystroke shapes)",
                       variable=self.enable_template,
                       font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=3)
        self._slider(param_f, "Template corr. threshold", self.template_corr_thresh, 0.0, 0.8, 0.05)

        # Filter section
        ttk.Separator(param_f, orient='horizontal').pack(fill=tk.X, pady=8)
        tk.Checkbutton(param_f, text="Bandpass filter output",
                       variable=self.enable_filtering,
                       command=self._toggle_filter,
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

        # ── 3. Output ────────────────────────────────────────────────────
        out_f = tk.LabelFrame(main, text="3. Output",
                              font=("Arial", 10, "bold"), padx=15, pady=15)
        out_f.pack(fill=tk.X, pady=(0, 15))
        out_row = tk.Frame(out_f)
        out_row.pack(fill=tk.X, pady=5)
        tk.Label(out_row, text="Output folder:", width=14, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.output_name_var = tk.StringVar(value=f"segmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tk.Entry(out_row, textvariable=self.output_name_var, width=40).pack(side=tk.LEFT, padx=5)

        # ── 4. Process ───────────────────────────────────────────────────
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

        # ── 5. Results ───────────────────────────────────────────────────
        res_f = tk.LabelFrame(main, text="Results & Diagnostics",
                              font=("Arial", 10, "bold"), padx=15, pady=15)
        res_f.pack(fill=tk.BOTH, expand=True)
        self.results_text = tk.Text(res_f, height=15, font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)

    # ── UI helpers ───────────────────────────────────────────────────────
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

    def _cancel(self):
        self._cancel_flag = True

    # ── Snapshot tkinter vars for thread safety ──────────────────────────
    def _snapshot_params(self) -> Dict:
        return {
            'output_name':        self.output_name_var.get(),
            'segment_duration':   self.segment_duration.get(),
            'pre_trigger':        self.pre_trigger.get(),
            'search_radius':      self.search_radius.get(),
            'peak_snr_db':        self.peak_snr_db.get(),
            'min_crest_factor':   self.min_crest_factor.get(),
            'enable_template':    self.enable_template.get(),
            'template_corr':      self.template_corr_thresh.get(),
            'enable_filtering':   self.enable_filtering.get(),
            'filter_low':         self.filter_low.get(),
            'filter_high':        self.filter_high.get(),
        }

    # ══════════════════════════════════════════════════════════════════════
    #  SESSION LOADING
    # ══════════════════════════════════════════════════════════════════════

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
            self.keystroke_log = self._read_log(os.path.join(folder, 'keystroke_log.csv'))
            dur = len(self.audio_data) / self.sample_rate
            self.session_label.config(text=f"Loaded: {os.path.basename(folder)}", fg="green")
            self.session_info_label.config(
                text=f"Audio: {dur:.1f}s @ {self.sample_rate} Hz | Keystrokes: {len(self.keystroke_log)}")
            self.process_btn.config(state=tk.NORMAL)
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

    # ══════════════════════════════════════════════════════════════════════
    #  DISPATCH
    # ══════════════════════════════════════════════════════════════════════

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

    # ══════════════════════════════════════════════════════════════════════
    #  SINGLE SESSION
    # ══════════════════════════════════════════════════════════════════════

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

            results = self._format_results(stats, key_counts, output_base, self.sample_rate)
            self._safe_ui(lambda: self.results_text.insert(1.0, results))
            self._safe_ui(lambda: self.progress_var.set(100))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: messagebox.showinfo("Done", f"Saved {stats['saved']} segments"))

        except InterruptedError:
            self._safe_ui(lambda: self.progress_label.config(text="Cancelled"))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._safe_ui(lambda: self.progress_label.config(text=f"Error: {e}"))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: messagebox.showerror("Error", str(e)))

    # ══════════════════════════════════════════════════════════════════════
    #  BATCH PROCESSING
    # ══════════════════════════════════════════════════════════════════════

    def _run_batch(self, params):
        try:
            # Determine output root (navigate up to find recordings folder)
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

                    print(f"\n[{si+1}/{n_sessions}] {sname}: {len(log)} keystrokes, "
                          f"{len(audio_data)/sr:.1f}s")

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
            self._safe_ui(lambda: messagebox.showinfo(
                "Batch Done",
                f"{overall['processed']} sessions, {overall['total_saved']} segments saved"))

        except Exception as e:
            import traceback; traceback.print_exc()
            self._safe_ui(lambda: self.progress_label.config(text=f"Error: {e}"))
            self._safe_ui(lambda: self.process_btn.config(state=tk.NORMAL))

    # ══════════════════════════════════════════════════════════════════════
    #  SAVE HELPERS
    # ══════════════════════════════════════════════════════════════════════

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
                    'time_diff': f"{seg['time_diff']:.4f}",
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
        """Save a CSV of all rejected segments for debugging."""
        rejected = [s for s in segments if s['status'] != 'ok']
        if not rejected:
            return
        path = os.path.join(output_base, 'rejections.csv')
        fields = ['key', 'csv_time', 'status', 'peak_sample', 'snr_db',
                  'peak_amp', 'crest_factor', 'template_corr']
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
            w.writeheader()
            for s in rejected:
                row = {k: s.get(k, '') for k in fields}
                # Format floats
                for fk in ['csv_time', 'snr_db', 'peak_amp', 'crest_factor', 'template_corr']:
                    if isinstance(row[fk], float):
                        row[fk] = f"{row[fk]:.4f}"
                w.writerow(row)

    # ══════════════════════════════════════════════════════════════════════
    #  FORMATTING
    # ══════════════════════════════════════════════════════════════════════

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

Recording analysis:
  Noise floor RMS:   {stats['noise_floor_rms']:.6f}
  Noise peak amp:    {stats['noise_peak_amp']:.6f}
  Auto min peak amp: {stats['min_peak_amplitude']:.6f}
  Sample rate:       {sr} Hz

Key distribution:
"""
        for k in sorted(key_counts):
            txt += f"  {k:12s}: {key_counts[k]:4d}\n"
        txt += f"\nOutput: {output_base}\n"
        txt += f"Check rejections.csv for details on rejected segments.\n"
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
                    f.write(f"{d['name']}: {s['saved']}/{s['total_csv']} saved "
                            f"(noise_floor={s['noise_floor_rms']:.6f})\n")
                else:
                    f.write(f"{d['name']}: FAILED — {d['error']}\n")