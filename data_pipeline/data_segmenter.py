"""
Keystroke Segmentation Module — v5

PHILOSOPHY: The CSV log IS the ground truth. Every CSV entry = exactly one
keystroke = exactly one output segment.  The segment is long enough (430 ms
default) to capture both the press transient and the release transient in a
single clip.

Key design decisions:
  1. ONE segment per CSV entry.  No separate press/release files.
     Output count == CSV entry count (minus quality rejections).
  2. Auto-offset via histogram cross-correlation between conservative onset
     detections and CSV times.
  3. Peak finding is CSV-guided: for each CSV time + offset, find the
     significant peak CLOSEST to the adjusted time (not the loudest).
     This ensures the press transient is picked over a louder release.
  4. Quality gates (SNR, crest factor, template) can reject truly silent or
     noise-only segments, but defaults are lenient.
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
from scipy.signal import find_peaks
import threading

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ═══════════════════════════════════════════════════════════════════════════════
#  SIGNAL UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 2:
        return np.mean(audio, axis=1).astype(np.float64)
    return audio.astype(np.float64)


def _amplitude_envelope(audio: np.ndarray, sr: int, ms: float = 3.0) -> np.ndarray:
    env = np.abs(audio.astype(np.float64))
    return uniform_filter1d(env, size=max(1, int(ms / 1000.0 * sr)), mode='constant')


def _estimate_noise_floor(audio: np.ndarray, sr: int) -> float:
    frame_len = int(0.050 * sr)
    n_frames = len(audio) // frame_len
    if n_frames < 5:
        return 1e-7
    frames = audio[:n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    return max(float(np.percentile(rms, 5)), 1e-7)


def _local_rms(audio: np.ndarray, center: int, sr: int, ms: float = 10.0) -> float:
    hw = int(ms / 1000.0 * sr / 2)
    lo, hi = max(0, center - hw), min(len(audio), center + hw)
    if hi <= lo:
        return 0.0
    return float(np.sqrt(np.mean(audio[lo:hi].astype(np.float64) ** 2)))


def _rms(seg: np.ndarray) -> float:
    return float(np.sqrt(np.mean(seg.astype(np.float64) ** 2)))


def _crest_factor(seg: np.ndarray) -> float:
    r = _rms(seg)
    return float(np.max(np.abs(seg))) / r if r > 1e-12 else 0.0


def _max_ncc(template: np.ndarray, candidate: np.ndarray, max_lag: int) -> float:
    n = min(len(template), len(candidate))
    if n == 0:
        return 0.0
    t = template[:n].astype(np.float64) - np.mean(template[:n])
    tn = np.linalg.norm(t)
    if tn < 1e-12:
        return 0.0
    best = -1.0
    for lag in range(-max_lag, max_lag + 1):
        t_sl = t[:n - lag] if lag >= 0 else t[-lag:]
        c_sl = (candidate[lag:n] if lag >= 0 else candidate[:n + lag]).astype(np.float64)
        if len(t_sl) == 0:
            continue
        c_sl -= np.mean(c_sl)
        cn = np.linalg.norm(c_sl)
        tn2 = np.linalg.norm(t_sl)
        if cn < 1e-12 or tn2 < 1e-12:
            continue
        corr = float(np.dot(t_sl / tn2, c_sl / cn))
        if corr > best:
            best = corr
    return best


def _bandpass(seg: np.ndarray, sr: int, low: int, high: int) -> np.ndarray:
    nyq = sr / 2.0
    lo, hi = max(low, 1) / nyq, min(high, nyq - 1) / nyq
    if lo >= hi:
        return seg
    sos = signal.butter(4, [lo, hi], btype='band', output='sos')
    return signal.sosfiltfilt(sos, seg).astype(seg.dtype)


# ═══════════════════════════════════════════════════════════════════════════════
#  CONSERVATIVE ONSET DETECTION  (only for offset calculation)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_prominent_onsets(mono: np.ndarray, sr: int) -> np.ndarray:
    """
    Detect only PROMINENT transients using amplitude envelope + high prominence.
    Conservative: better to miss some than to over-detect.
    Used ONLY for auto-offset, NOT for extraction.
    """
    amp_env = _amplitude_envelope(mono, sr, ms=3.0)
    min_dist = max(1, int(0.040 * sr))   # 40 ms min gap

    # Prominence = peak must stand out from local baseline by this much.
    # Use percentile gap: p90 - p50 ensures only the top transients pass.
    p90 = float(np.percentile(amp_env, 90))
    p50 = float(np.percentile(amp_env, 50))
    prom = max(p90 - p50, p50 * 3.0, 1e-6)

    peaks, _ = find_peaks(amp_env, distance=min_dist, prominence=prom)
    return peaks


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTO OFFSET  — histogram-based cross-correlation
# ═══════════════════════════════════════════════════════════════════════════════

def auto_calculate_offset(audio: np.ndarray, sr: int,
                          keystroke_log: List[Dict],
                          max_offset_sec: float = 2.0) -> float:
    """
    Histogram-of-differences offset estimation:
      1. Detect prominent onsets globally (conservative — few false positives).
      2. For every (onset, csv_time) pair, compute diff = onset − csv_time.
      3. Histogram with 10 ms bins → tallest bin = true offset.
      4. Refine with median of nearby diffs.
    """
    if not keystroke_log or len(keystroke_log) < 2:
        return 0.0

    mono = _to_mono(audio)
    csv_times = sorted([float(e['relative_time']) for e in keystroke_log])
    csv_arr = np.array(csv_times)

    onset_samples = _detect_prominent_onsets(mono, sr)
    if len(onset_samples) < 3:
        print(f"  Auto-offset: only {len(onset_samples)} onsets, returning 0")
        return 0.0

    onset_times = onset_samples.astype(np.float64) / sr
    print(f"  Auto-offset: {len(onset_times)} onsets, {len(csv_times)} CSV entries")

    # All pairwise diffs within range
    diffs = []
    for ot in onset_times:
        d = ot - csv_arr
        mask = np.abs(d) <= max_offset_sec
        diffs.extend(d[mask].tolist())

    if len(diffs) < 3:
        print(f"  Auto-offset: too few diffs ({len(diffs)}), returning 0")
        return 0.0
    diffs = np.array(diffs)

    # Histogram with 10 ms bins
    bin_w = 0.010
    lo_edge = float(np.floor(diffs.min() / bin_w) * bin_w)
    hi_edge = float(np.ceil(diffs.max() / bin_w) * bin_w) + bin_w
    bins = np.arange(lo_edge, hi_edge + bin_w / 2, bin_w)
    hist, edges = np.histogram(diffs, bins=bins)

    best_idx = int(np.argmax(hist))
    rough = (edges[best_idx] + edges[best_idx + 1]) / 2.0

    # Refine: median of diffs within ±30 ms of rough
    nearby = diffs[np.abs(diffs - rough) <= 0.030]
    result = float(np.median(nearby)) if len(nearby) >= 3 else rough

    # Report match quality
    shifted = csv_arr + result
    matched = sum(1 for sc in shifted if np.any(np.abs(onset_times - sc) < 0.060))
    pct = 100.0 * matched / len(csv_times) if csv_times else 0
    print(f"  Auto-offset: {result:+.4f} s  "
          f"(matches {matched}/{len(csv_times)} = {pct:.0f}%)")
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE EXTRACTION  — strictly CSV-guided, one segment per entry
# ═══════════════════════════════════════════════════════════════════════════════

def extract_keystrokes(
    audio: np.ndarray,
    sr: int,
    keystroke_log: List[Dict],
    *,
    offset: float = 0.0,
    segment_duration: float = 0.430,
    pre_trigger: float = 0.10,
    search_radius_sec: float = 0.25,
    peak_snr_db: float = 3.0,
    min_crest_factor: float = 1.2,
    min_peak_amplitude: float = 0.0,
    enable_template_verify: bool = True,
    template_corr_threshold: float = 0.20,
    template_build_count: int = 20,
    enable_bandpass: bool = False,
    bandpass_low: int = 50,
    bandpass_high: int = 5000,
    overlap_guard_sec: float = 0.015,
    verbose: bool = True,
    progress_callback=None,
) -> Tuple[List[Dict], Dict]:
    """
    CSV-guided extraction.  For each CSV entry:
      1. Compute adjusted_time = csv_time + offset
      2. Search ±search_radius for the significant peak closest to adj_time
      3. Cut one segment of segment_duration centred pre_trigger before peak
      4. Apply quality gates; reject if truly silent/noisy

    Output: exactly 0 or 1 segment per CSV entry.  The 430 ms window
    naturally includes both press and release transients.
    """
    is_stereo = audio.ndim == 2
    mono = _to_mono(audio)
    n_audio = len(mono)

    noise_rms = _estimate_noise_floor(mono, sr)

    # Estimate noise peak for amplitude gate
    frame_len = int(0.05 * sr)
    n_frames = n_audio // frame_len
    if n_frames > 5:
        frames = mono[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        quiet = frames[frame_rms <= np.percentile(frame_rms, 15)].ravel()
        noise_peak = float(np.percentile(np.abs(quiet), 99)) if len(quiet) else noise_rms * 3
    else:
        noise_peak = noise_rms * 3

    if min_peak_amplitude <= 0:
        min_peak_amplitude = noise_peak * 1.2

    local_snr_thresh = noise_rms * (10 ** (peak_snr_db / 20.0))

    if verbose:
        print(f"\n  ┌─ Extraction ─────────────────────────────────")
        print(f"  │ Audio  : {n_audio / sr:.2f} s  offset: {offset:+.4f} s")
        print(f"  │ Noise  : {noise_rms:.6f}  min_peak: {min_peak_amplitude:.6f}")
        print(f"  │ CSV    : {len(keystroke_log)}")
        print(f"  └──────────────────────────────────────────────")

    sorted_log = sorted(keystroke_log, key=lambda x: x['relative_time'])
    n_total = len(sorted_log)

    dur_s = int(segment_duration * sr)
    pre_s = int(pre_trigger * sr)
    rad_s = int(search_radius_sec * sr)
    ovlp_s = int(overlap_guard_sec * sr)

    segments: List[Dict] = []
    template_bank: List[np.ndarray] = []
    avg_tpl_env = None
    last_peak = -999999

    stats = dict(total_csv=n_total, saved=0,
                 rejected_silent=0, rejected_low_snr=0,
                 rejected_low_peak=0, rejected_low_transient=0,
                 rejected_template=0, rejected_overlap=0)

    for idx, entry in enumerate(sorted_log):
        csv_t = entry['relative_time']
        adj_t = max(0.0, csv_t + offset)
        adj_s = int(adj_t * sr)
        key = entry['key']

        # ── Find peak: closest significant peak to adjusted CSV time ─────
        #
        # The offset was calibrated to align CSV times with PRESS onsets.
        # So the press peak is near adj_s; the release is 60–150 ms later.
        # Picking argmax would grab a louder release → press gets cut off.
        # Instead: find all significant peaks, pick the one nearest adj_s.
        #
        lo = max(0, adj_s - rad_s)
        hi = min(n_audio, adj_s + rad_s)

        if hi <= lo:
            segments.append(_rej(entry, 'boundary'))
            stats['rejected_silent'] += 1
            _prog(progress_callback, idx, n_total)
            continue

        region = np.abs(mono[lo:hi])

        # Find significant peaks (prominence > 30% of region max)
        reg_max = float(region.max())
        min_prom = max(reg_max * 0.15, noise_peak * 0.5, 1e-7)
        min_d = max(1, int(0.015 * sr))
        cand_peaks, _ = find_peaks(region, distance=min_d,
                                    prominence=min_prom)

        if len(cand_peaks) == 0:
            # No discrete peaks found — use argmax as fallback
            pk = lo + int(np.argmax(region))
        elif len(cand_peaks) == 1:
            pk = lo + int(cand_peaks[0])
        else:
            # Pick the peak closest to adj_s (the calibrated press time)
            abs_peaks = lo + cand_peaks
            dists = np.abs(abs_peaks - adj_s)
            pk = int(abs_peaks[np.argmin(dists)])

        # Overlap guard
        if abs(pk - last_peak) < ovlp_s:
            segments.append(_rej(entry, 'overlap', peak_sample=pk))
            stats['rejected_overlap'] += 1
            _prog(progress_callback, idx, n_total)
            continue

        # Peak amplitude
        tiny = int(0.003 * sr)
        pa = float(np.max(np.abs(
            mono[max(0, pk - tiny):min(n_audio, pk + tiny)])))
        if pa < min_peak_amplitude:
            segments.append(_rej(entry, 'low_peak', peak_sample=pk, peak_amp=pa))
            stats['rejected_low_peak'] += 1
            _prog(progress_callback, idx, n_total)
            continue

        # SNR gate
        lrms = _local_rms(mono, pk, sr)
        if lrms < local_snr_thresh:
            snr = 20.0 * np.log10(max(lrms, 1e-12) / noise_rms)
            segments.append(_rej(entry, 'low_snr', peak_sample=pk,
                                 snr_db=snr, peak_amp=pa))
            stats['rejected_low_snr'] += 1
            _prog(progress_callback, idx, n_total)
            continue

        snr = 20.0 * np.log10(lrms / noise_rms)

        # ── Cut segment ──────────────────────────────────────────────────
        start = max(0, pk - pre_s)
        end = start + dur_s
        if end > n_audio:
            end = n_audio
            start = max(0, end - dur_s)
        seg = mono[start:end]
        if len(seg) < dur_s:
            seg = np.pad(seg, (0, dur_s - len(seg)))
        seg = seg[:dur_s]

        # Crest factor
        cf = _crest_factor(seg)
        if cf < min_crest_factor:
            segments.append(_rej(entry, 'low_transient', peak_sample=pk,
                                 snr_db=snr, peak_amp=pa, crest=cf))
            stats['rejected_low_transient'] += 1
            _prog(progress_callback, idx, n_total)
            continue

        # Template verify
        tpl_corr = 1.0
        if enable_template_verify and avg_tpl_env is not None:
            s = seg[:len(avg_tpl_env)]
            if len(s) == len(avg_tpl_env):
                tpl_corr = _max_ncc(avg_tpl_env, _amplitude_envelope(s, sr),
                                    int(0.015 * sr))
                if tpl_corr < template_corr_threshold:
                    segments.append(_rej(entry, 'template', peak_sample=pk,
                                         snr_db=snr, peak_amp=pa, tpl=tpl_corr))
                    stats['rejected_template'] += 1
                    _prog(progress_callback, idx, n_total)
                    continue

        # ── ACCEPTED ─────────────────────────────────────────────────────
        fseg = seg.copy()
        if enable_bandpass:
            fseg = _bandpass(fseg, sr, bandpass_low, bandpass_high)
        if is_stereo:
            fseg = np.column_stack([fseg, fseg])

        segments.append(dict(
            audio=fseg, key=key,
            timestamp=entry.get('timestamp', ''),
            csv_time=entry['relative_time'],
            peak_sample=pk, start_sample=start, end_sample=end,
            snr_db=snr, peak_amp=pa, crest_factor=cf,
            template_corr=tpl_corr,
            time_diff=abs(pk / sr - adj_t),
            status='ok',
        ))
        stats['saved'] += 1
        last_peak = pk

        # Build template bank
        if enable_template_verify and len(template_bank) < template_build_count:
            template_bank.append(seg.copy())
            if len(template_bank) == template_build_count:
                envs = []
                for tb in template_bank:
                    e = _amplitude_envelope(tb, sr)
                    envs.append(np.roll(e, pre_s - int(np.argmax(e))))
                avg_tpl_env = np.mean(envs, axis=0)

        _prog(progress_callback, idx, n_total)

    if verbose:
        s = stats
        print(f"  Saved {s['saved']}/{n_total}  "
              f"(boundary={s['rejected_silent']} peak={s['rejected_low_peak']} "
              f"snr={s['rejected_low_snr']} crest={s['rejected_low_transient']} "
              f"tpl={s['rejected_template']} ovlp={s['rejected_overlap']})")

    return segments, stats


def _rej(entry, reason, **kw):
    return dict(audio=None, key=entry['key'],
                timestamp=entry.get('timestamp', ''),
                csv_time=entry['relative_time'],
                peak_sample=kw.get('peak_sample', 0),
                start_sample=0, end_sample=0,
                snr_db=kw.get('snr_db', 0.0),
                peak_amp=kw.get('peak_amp', 0.0),
                crest_factor=kw.get('crest', 0.0),
                template_corr=kw.get('tpl', 0.0),
                time_diff=0.0, status=reason)


def _prog(cb, idx, total):
    if cb:
        cb(idx + 1, total)


# ═══════════════════════════════════════════════════════════════════════════════
#  UI — two-panel layout
# ═══════════════════════════════════════════════════════════════════════════════

class DataSegmenterTab:
    """
    Left  — scrollable controls
    Right — matplotlib canvas (offset-preview OR full-audio viewer)
    """

    def __init__(self, parent, config, audio_handler):
        self.parent = parent
        self.config = config
        self.audio  = audio_handler

        # Session state
        self.current_session_path: Optional[str] = None
        self.batch_sessions: Optional[List[str]] = None
        self.session_cache: Dict[str, tuple] = {}

        # Preview state
        self.prev_audio = None
        self.prev_log:  List[Dict] = []
        self.prev_sr:   Optional[int] = None
        self.prev_path: Optional[str] = None
        self.prev_offset: float = 0.0

        # Offset token (guards stale background results)
        self._offset_token: int = 0

        # Viewer state
        self.viewer_sessions: Dict[str, tuple] = {}
        self.viewer_audio    = None
        self.viewer_sr:      Optional[int] = None
        self.viewer_segments: Optional[List[Dict]] = None
        self._viewer_ok_segs: List[Dict] = []
        self.viewer_session_var = tk.StringVar()

        self._cancel_flag = False
        self._drag_x: Optional[float] = None

        # Tk vars
        self.segment_duration     = tk.DoubleVar(value=0.430)
        self.pre_trigger          = tk.DoubleVar(value=0.10)
        self.search_radius        = tk.DoubleVar(value=0.25)
        self.peak_snr_db          = tk.DoubleVar(value=3.0)
        self.min_crest_factor     = tk.DoubleVar(value=1.2)
        self.enable_template      = tk.BooleanVar(value=True)
        self.template_corr_thresh = tk.DoubleVar(value=0.20)
        self.enable_filtering     = tk.BooleanVar(value=False)
        self.filter_low           = tk.IntVar(value=50)
        self.filter_high          = tk.IntVar(value=5000)
        self.output_name_var      = tk.StringVar(
            value=f"segmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.fine_tune            = tk.DoubleVar(value=0.0)
        self.preview_session_var  = tk.StringVar()
        self.seg_var              = tk.IntVar(value=0)

        self.right_mode = 'preview'
        self._build_ui()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build_ui(self):
        self.paned = tk.PanedWindow(self.parent, orient=tk.HORIZONTAL,
                                    sashwidth=6, sashrelief=tk.RAISED)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # Left pane (scrollable)
        lo = tk.Frame(self.paned)
        self.paned.add(lo, minsize=540)
        lc = tk.Canvas(lo, width=535, highlightthickness=0)
        ls = tk.Scrollbar(lo, orient='vertical', command=lc.yview)
        inner = tk.Frame(lc)
        inner.bind('<Configure>',
                   lambda e: lc.configure(scrollregion=lc.bbox('all')))
        lc.create_window((0, 0), window=inner, anchor='nw')
        lc.configure(yscrollcommand=ls.set)
        ls.pack(side='right', fill='y')
        lc.pack(side='left', fill='both', expand=True)
        lc.bind_all('<MouseWheel>',
                    lambda e: lc.yview_scroll(int(-e.delta / 120), 'units'))
        self._build_left(inner)

        # Right pane
        ro = tk.Frame(self.paned, bg='#f5f5f5')
        self.paned.add(ro, minsize=460)
        ro.grid_rowconfigure(2, weight=1)
        ro.grid_columnconfigure(0, weight=1)
        self._build_right(ro)

    # ── left controls ─────────────────────────────────────────────────────────

    def _build_left(self, main):
        tk.Label(main, text='Keystroke Segmenter v5',
                 font=('Arial', 15, 'bold'), fg='#1565C0').pack(pady=(10, 16))

        # 1. Load
        f1 = tk.LabelFrame(main, text='1. Load Session',
                           font=('Arial', 10, 'bold'), padx=12, pady=10)
        f1.pack(fill=tk.X, padx=10, pady=(0, 10))
        tk.Button(f1, text='Browse Session / Batch Folder',
                  command=self._browse,
                  bg='#1976D2', fg='white',
                  font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=4)
        self.session_label = tk.Label(f1, text='No session loaded',
                                      font=('Arial', 9), fg='gray')
        self.session_label.pack(anchor=tk.W)
        self.session_info = tk.Label(f1, text='', font=('Arial', 9), fg='#1565C0')
        self.session_info.pack(anchor=tk.W, pady=(2, 0))

        # Batch selector
        self.batch_sel_frame = tk.Frame(f1)
        self.batch_sel_frame.pack(fill=tk.X, pady=(6, 0))
        tk.Label(self.batch_sel_frame, text='Preview:',
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=4)
        self.session_combo = ttk.Combobox(
            self.batch_sel_frame, textvariable=self.preview_session_var,
            state='readonly', width=38)
        self.session_combo.pack(side=tk.LEFT, padx=4)
        self.session_combo.bind('<<ComboboxSelected>>', self._on_combo_select)
        self.batch_sel_frame.pack_forget()

        # 2. Parameters
        f2 = tk.LabelFrame(main, text='2. Segmentation Parameters',
                           font=('Arial', 10, 'bold'), padx=12, pady=10)
        f2.pack(fill=tk.X, padx=10, pady=(0, 10))
        self._slider(f2, 'Segment duration (s)', self.segment_duration, 0.1, 1.0, 0.01)
        self._slider(f2, 'Pre-trigger (s)',       self.pre_trigger,      0.0, 0.3, 0.01)
        self._slider(f2, 'Search radius (s)',     self.search_radius,    0.05, 0.50, 0.01)
        ttk.Separator(f2, orient='horizontal').pack(fill=tk.X, pady=6)
        tk.Label(f2, text='Quality Gates', font=('Arial', 9, 'bold'),
                 fg='#555').pack(anchor=tk.W)
        self._slider(f2, 'Peak SNR (dB)',    self.peak_snr_db,      0.0, 15.0, 0.5)
        self._slider(f2, 'Min crest factor', self.min_crest_factor, 1.0, 6.0,  0.1)
        ttk.Separator(f2, orient='horizontal').pack(fill=tk.X, pady=6)
        tk.Checkbutton(f2, text='Template verification',
                       variable=self.enable_template,
                       font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=2)
        self._slider(f2, 'Template threshold', self.template_corr_thresh, 0.0, 0.8, 0.05)
        ttk.Separator(f2, orient='horizontal').pack(fill=tk.X, pady=6)
        tk.Checkbutton(f2, text='Bandpass filter',
                       variable=self.enable_filtering,
                       command=self._toggle_filter,
                       font=('Arial', 9, 'bold')).pack(anchor=tk.W, pady=2)
        fr = tk.Frame(f2); fr.pack(fill=tk.X, pady=2)
        tk.Label(fr, text='Low Hz:').pack(side=tk.LEFT, padx=4)
        self.flt_lo_entry = tk.Entry(fr, textvariable=self.filter_low, width=7)
        self.flt_lo_entry.pack(side=tk.LEFT, padx=2)
        tk.Label(fr, text='High Hz:').pack(side=tk.LEFT, padx=8)
        self.flt_hi_entry = tk.Entry(fr, textvariable=self.filter_high, width=7)
        self.flt_hi_entry.pack(side=tk.LEFT, padx=2)
        pr = tk.Frame(f2); pr.pack(fill=tk.X, pady=2)
        for nm, lo, hi in [('50–5kHz', 50, 5000), ('50–3kHz', 50, 3000),
                            ('50–8kHz', 50, 8000)]:
            tk.Button(pr, text=nm,
                      command=lambda l=lo, h=hi: self._set_filter(l, h)
                      ).pack(side=tk.LEFT, padx=2)
        self._toggle_filter()

        # 3. Output
        f3 = tk.LabelFrame(main, text='3. Output',
                           font=('Arial', 10, 'bold'), padx=12, pady=10)
        f3.pack(fill=tk.X, padx=10, pady=(0, 10))
        or_ = tk.Frame(f3); or_.pack(fill=tk.X)
        tk.Label(or_, text='Output folder:', width=14,
                 anchor=tk.W).pack(side=tk.LEFT, padx=4)
        tk.Entry(or_, textvariable=self.output_name_var,
                 width=34).pack(side=tk.LEFT, padx=4)

        # 4. Process
        f4 = tk.LabelFrame(main, text='4. Process',
                           font=('Arial', 10, 'bold'), padx=12, pady=10)
        f4.pack(fill=tk.X, padx=10, pady=(0, 10))
        self.prog_var = tk.DoubleVar()
        ttk.Progressbar(f4, variable=self.prog_var, maximum=100,
                        length=440).pack(pady=6)
        self.prog_label = tk.Label(f4, text='Ready',
                                   font=('Arial', 10), fg='#424242')
        self.prog_label.pack(pady=2)
        br = tk.Frame(f4); br.pack(pady=6)
        self.process_btn = tk.Button(
            br, text='▶  Start Segmentation',
            command=self._start_segmentation,
            bg='#43A047', fg='white', font=('Arial', 11, 'bold'),
            width=20, height=2, state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=8)
        tk.Button(br, text='Cancel',
                  command=lambda: setattr(self, '_cancel_flag', True),
                  font=('Arial', 11), width=10, height=2).pack(side=tk.LEFT, padx=8)

        # 5. Results
        f5 = tk.LabelFrame(main, text='5. Results',
                           font=('Arial', 10, 'bold'), padx=12, pady=10)
        f5.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        self.results_text = tk.Text(f5, height=12, font=('Courier', 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
        self.show_viewer_btn = tk.Button(
            f5, text='🎵  Switch to Full-Audio Viewer',
            command=self._activate_viewer,
            bg='#EF6C00', fg='white', font=('Arial', 10, 'bold'),
            state=tk.DISABLED)
        self.show_viewer_btn.pack(pady=6)

    # ── right panel ───────────────────────────────────────────────────────────

    def _build_right(self, parent):
        # Row 0 — toolbar
        tb = tk.Frame(parent, bg='#e8eaf6', pady=4)
        tb.grid(row=0, column=0, sticky='ew')
        tk.Label(tb, text='Panel:', bg='#e8eaf6',
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=8)
        self.btn_prev = tk.Button(tb, text='📊 Offset Preview',
                                  command=self._activate_preview,
                                  bg='#90CAF9', relief=tk.SUNKEN,
                                  font=('Arial', 9))
        self.btn_prev.pack(side=tk.LEFT, padx=3)
        self.btn_view = tk.Button(tb, text='🎵 Full-Audio Viewer',
                                  command=self._activate_viewer,
                                  font=('Arial', 9))
        self.btn_view.pack(side=tk.LEFT, padx=3)

        # Row 1a — offset strip
        self.offset_strip = tk.Frame(parent, bg='#f1f8e9', pady=4)
        self.offset_strip.grid(row=1, column=0, sticky='ew')
        tk.Label(self.offset_strip, text='Auto offset:',
                 bg='#f1f8e9', font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=8)
        self.auto_offset_lbl = tk.Label(
            self.offset_strip, text='—',
            bg='#f1f8e9', fg='#1565C0', font=('Arial', 9, 'bold'), width=12)
        self.auto_offset_lbl.pack(side=tk.LEFT)
        tk.Label(self.offset_strip, text='Fine-tune (±0.15 s):',
                 bg='#f1f8e9', font=('Arial', 9)).pack(side=tk.LEFT, padx=(10, 4))
        tk.Scale(self.offset_strip, from_=-0.15, to=0.15, resolution=0.005,
                 orient=tk.HORIZONTAL, variable=self.fine_tune,
                 command=lambda _: self._redraw_preview(),
                 bg='#f1f8e9', length=170, showvalue=True).pack(side=tk.LEFT)
        tk.Button(self.offset_strip, text='Reset',
                  command=lambda: (self.fine_tune.set(0.0),
                                   self._redraw_preview()),
                  font=('Arial', 8)).pack(side=tk.LEFT, padx=6)

        # Row 1b — viewer nav (hidden initially)
        self.viewer_nav = tk.Frame(parent, bg='#fff3e0', pady=4)
        self.viewer_nav.grid(row=1, column=0, sticky='ew')
        self.viewer_nav.grid_remove()

        vn_top = tk.Frame(self.viewer_nav, bg='#fff3e0')
        vn_top.pack(fill=tk.X, pady=(0, 2))
        tk.Label(vn_top, text='Session:', bg='#fff3e0',
                 font=('Arial', 9, 'bold')).pack(side=tk.LEFT, padx=(6, 3))
        self.viewer_session_combo = ttk.Combobox(
            vn_top, textvariable=self.viewer_session_var,
            state='readonly', width=36)
        self.viewer_session_combo.pack(side=tk.LEFT, padx=2)
        self.viewer_session_combo.bind(
            '<<ComboboxSelected>>', self._on_viewer_session_select)

        vn_bot = tk.Frame(self.viewer_nav, bg='#fff3e0')
        vn_bot.pack(fill=tk.X)
        for txt, cmd in [('Fit', self._v_fit), ('Zoom+', self._v_zin),
                          ('Zoom−', self._v_zout), ('◀', self._v_pleft),
                          ('▶', self._v_pright)]:
            tk.Button(vn_bot, text=txt, command=cmd,
                      font=('Arial', 8), bg='#FFE0B2').pack(side=tk.LEFT, padx=2)
        tk.Label(vn_bot, text=' Seg:', bg='#fff3e0',
                 font=('Arial', 9)).pack(side=tk.LEFT)
        self.seg_slider = tk.Scale(
            vn_bot, from_=0, to=0, orient=tk.HORIZONTAL,
            variable=self.seg_var, command=self._v_goto_seg,
            bg='#fff3e0', length=155)
        self.seg_slider.pack(side=tk.LEFT)
        self.seg_label = tk.Label(vn_bot, text='0/0',
                                  bg='#fff3e0', font=('Arial', 9))
        self.seg_label.pack(side=tk.LEFT, padx=4)

        # Row 2 — canvas
        cf = tk.Frame(parent, bg='white')
        cf.grid(row=2, column=0, sticky='nsew')

        if HAS_MATPLOTLIB:
            self.fig, self.ax = plt.subplots(figsize=(7, 5))
            self.fig.patch.set_facecolor('white')
            self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=cf)
            self.mpl_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self._placeholder('Load a session to see the offset preview')
            self.mpl_canvas.mpl_connect('button_press_event',   self._on_press)
            self.mpl_canvas.mpl_connect('button_release_event', self._on_release)
            self.mpl_canvas.mpl_connect('motion_notify_event',  self._on_motion)
        else:
            tk.Label(cf, text='pip install matplotlib',
                     fg='red', font=('Arial', 12)).pack(expand=True)

    # ── mode switching ────────────────────────────────────────────────────────

    def _activate_preview(self):
        self.right_mode = 'preview'
        self.btn_prev.config(relief=tk.SUNKEN, bg='#90CAF9')
        self.btn_view.config(relief=tk.RAISED, bg='#d9d9d9')
        self.viewer_nav.grid_remove()
        self.offset_strip.grid()
        self._redraw_preview()

    def _activate_viewer(self):
        self.right_mode = 'viewer'
        self.btn_view.config(relief=tk.SUNKEN, bg='#90CAF9')
        self.btn_prev.config(relief=tk.RAISED, bg='#d9d9d9')
        self.offset_strip.grid_remove()
        self.viewer_nav.grid()
        self._redraw_viewer()

    def _register_viewer_session(self, name, audio, sr, segs):
        self.viewer_sessions[name] = (audio, sr, segs)
        names = list(self.viewer_sessions.keys())
        self.viewer_session_combo.config(values=names)
        self.viewer_session_combo.set(name)
        self.viewer_audio    = audio
        self.viewer_sr       = sr
        self.viewer_segments = segs

    def _on_viewer_session_select(self, _=None):
        name = self.viewer_session_var.get()
        if name in self.viewer_sessions:
            audio, sr, segs = self.viewer_sessions[name]
            self.viewer_audio    = audio
            self.viewer_sr       = sr
            self.viewer_segments = segs
            self._redraw_viewer()

    # ── session loading ───────────────────────────────────────────────────────

    def _browse(self):
        folder = filedialog.askdirectory(
            title='Select Session Folder (or batch parent)')
        if not folder:
            return
        if (os.path.exists(os.path.join(folder, 'audio.wav')) and
                os.path.exists(os.path.join(folder, 'keystroke_log.csv'))):
            self._load_single(folder)
        else:
            sessions = self._find_sessions(folder)
            if not sessions:
                messagebox.showerror(
                    'Error',
                    'No valid sessions found.\n'
                    'Needs audio.wav + keystroke_log.csv.')
                return
            msg = (f'Found {len(sessions)} session(s).\n\n'
                   + '\n'.join(f'  • {os.path.basename(s)}'
                               for s in sessions[:10])
                   + (f'\n  … and {len(sessions)-10} more'
                      if len(sessions) > 10 else '')
                   + '\n\nProcess all in batch?')
            if messagebox.askyesno('Batch', msg):
                self._load_batch(sessions)

    @staticmethod
    def _find_sessions(root, max_depth=3):
        found = []
        def _walk(d, depth):
            if depth > max_depth:
                return
            try:
                if (os.path.exists(os.path.join(d, 'audio.wav')) and
                        os.path.exists(os.path.join(d, 'keystroke_log.csv'))):
                    found.append(d); return
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
            audio, sr = self.audio.load_audio(os.path.join(folder, 'audio.wav'))
            if audio is None:
                raise RuntimeError('load_audio returned None')
            log = self._read_log(os.path.join(folder, 'keystroke_log.csv'))
            self.session_cache[folder] = (audio, sr, log)
        except Exception as e:
            messagebox.showerror('Error', f'Failed to load:\n{e}')
            return

        self.current_session_path = folder
        self.batch_sessions = None
        self.batch_sel_frame.pack_forget()

        dur = len(audio) / sr
        self.session_label.config(
            text=f'Loaded: {os.path.basename(folder)}', fg='green')
        self.session_info.config(
            text=f'Audio: {dur:.1f} s @ {sr} Hz  |  {len(log)} keystrokes')
        self.process_btn.config(state=tk.NORMAL)
        self.show_viewer_btn.config(state=tk.DISABLED)

        self._apply_preview(folder, audio, sr, log, offset=0.0)
        self._kick_offset_computation(folder, audio, sr, log)

    def _load_batch(self, sessions):
        self.current_session_path = None
        self.batch_sessions = sessions
        self.session_cache = {}

        names = [os.path.basename(s) for s in sessions]
        self.session_combo.config(values=names)
        self.session_combo.current(0)
        self.batch_sel_frame.pack(fill=tk.X, pady=(6, 0))

        total = 0
        for s in sessions:
            try:
                total += len(self._read_log(
                    os.path.join(s, 'keystroke_log.csv')))
            except Exception:
                pass
        self.session_label.config(
            text=f'Batch: {len(sessions)} sessions', fg='green')
        self.session_info.config(text=f'≈ {total} keystrokes total')
        self.process_btn.config(state=tk.NORMAL)
        self.show_viewer_btn.config(state=tk.DISABLED)
        self._load_preview_async(sessions[0])

    def _on_combo_select(self, _=None):
        idx = self.session_combo.current()
        if self.batch_sessions and 0 <= idx < len(self.batch_sessions):
            self._load_preview_async(self.batch_sessions[idx])

    def _load_preview_async(self, folder):
        self._placeholder('Loading session…')
        self.auto_offset_lbl.config(text='loading…', fg='#888')

        def _run():
            try:
                if folder in self.session_cache:
                    audio, sr, log = self.session_cache[folder]
                else:
                    audio, sr = self.audio.load_audio(
                        os.path.join(folder, 'audio.wav'))
                    if audio is None:
                        raise RuntimeError('load_audio returned None')
                    log = self._read_log(
                        os.path.join(folder, 'keystroke_log.csv'))
                    self.session_cache[folder] = (audio, sr, log)

                self._safe_ui(lambda: self._apply_preview(
                    folder, audio, sr, log, offset=0.0))
                self._kick_offset_computation(folder, audio, sr, log)
            except Exception as e:
                self._safe_ui(
                    lambda: self._placeholder(f'Load error: {e}'))

        threading.Thread(target=_run, daemon=True).start()

    def _apply_preview(self, folder, audio, sr, log, *, offset):
        self.prev_audio  = audio
        self.prev_log    = log
        self.prev_sr     = sr
        self.prev_path   = folder
        self.prev_offset = offset
        self.fine_tune.set(0.0)
        self.auto_offset_lbl.config(
            text=f'{offset:+.4f} s' if offset != 0.0 else 'computing…',
            fg='#1565C0' if offset != 0.0 else '#888')
        if self.right_mode == 'preview':
            self._redraw_preview()

    def _kick_offset_computation(self, folder, audio, sr, log):
        self._offset_token += 1
        token = self._offset_token
        self.auto_offset_lbl.config(text='computing…', fg='#888')

        def _run():
            try:
                off = auto_calculate_offset(audio, sr, log)
            except Exception as e:
                print(f'Offset error: {e}')
                import traceback; traceback.print_exc()
                off = 0.0

            def _apply():
                if self._offset_token != token:
                    return
                self.prev_offset = off
                self.auto_offset_lbl.config(
                    text=f'{off:+.4f} s', fg='#1565C0')
                if self.right_mode == 'preview':
                    self._redraw_preview()

            self._safe_ui(_apply)

        threading.Thread(target=_run, daemon=True).start()

    # ── preview drawing ───────────────────────────────────────────────────────

    def _total_offset(self) -> float:
        return self.prev_offset + self.fine_tune.get()

    def _redraw_preview(self):
        if not HAS_MATPLOTLIB:
            return
        if self.prev_audio is None or not self.prev_log or self.prev_sr is None:
            self._placeholder('Load a session to see the offset preview')
            return
        try:
            sr   = self.prev_sr
            mono = _to_mono(self.prev_audio)
            n_a  = len(mono)
            toff = self._total_offset()

            first = self.prev_log[0]
            csv_t = float(first['relative_time'])
            adj_t = max(0.0, csv_t + toff)
            pre   = self.pre_trigger.get()
            dur   = self.segment_duration.get()
            seg_s = adj_t - pre
            seg_e = seg_s + dur

            ctx = 1.5
            v0 = max(0.0, adj_t - ctx)
            v1 = min(n_a / sr, adj_t + ctx)
            s0, s1 = int(v0 * sr), int(v1 * sr)
            snippet = mono[s0:s1].astype(np.float32)
            t_ax = np.linspace(v0, v1, len(snippet))

            self.ax.clear()
            self.ax.plot(t_ax, snippet, color='#1565C0', lw=0.7, alpha=0.75,
                         label='Audio')

            ew0 = max(v0, seg_s)
            ew1 = min(v1, seg_e)
            if ew1 > ew0:
                self.ax.axvspan(ew0, ew1, alpha=0.22, color='#4CAF50',
                                label='Extraction window')

            self.ax.axvline(csv_t, color='#FF9800', ls='--', lw=1.8,
                            label=f'CSV time ({csv_t:.3f} s)')
            self.ax.axvline(adj_t, color='#E53935', ls='-', lw=2.2,
                            label=f'Adjusted ({adj_t:.3f} s)')

            # Show a few more keystrokes for context
            for entry in self.prev_log[1:8]:
                ct2 = float(entry['relative_time'])
                at2 = max(0.0, ct2 + toff)
                if v0 <= at2 <= v1:
                    self.ax.axvline(at2, color='#E53935', ls=':', lw=0.8,
                                    alpha=0.4)
                if v0 <= ct2 <= v1:
                    self.ax.axvline(ct2, color='#FF9800', ls=':', lw=0.8,
                                    alpha=0.3)

            ok = (0 <= int(seg_s * sr) and int(seg_e * sr) <= n_a)
            self.ax.set_title(
                f"Key: '{first['key']}' | "
                f"auto {self.prev_offset:+.4f} s  "
                f"fine {self.fine_tune.get():+.3f} s"
                f" = total {toff:+.4f} s | "
                f"{'✓ OK' if ok else '✗ OOB'}",
                fontsize=9, fontweight='bold',
                color='#1B5E20' if ok else '#B71C1C')
            self.ax.set_xlabel('Time (s)', fontsize=9)
            self.ax.set_ylabel('Amplitude', fontsize=9)
            self.ax.legend(loc='upper right', fontsize=8)
            self.ax.grid(True, alpha=0.2)
            self.ax.set_xlim(v0, v1)
            self.fig.tight_layout(pad=0.6)
            self.mpl_canvas.draw_idle()
        except Exception as e:
            print(f'Preview redraw error: {e}')
            import traceback; traceback.print_exc()

    # ── viewer drawing ────────────────────────────────────────────────────────

    def _redraw_viewer(self):
        if not HAS_MATPLOTLIB:
            return
        if (self.viewer_segments is None or
                self.viewer_audio is None or not self.viewer_sr):
            self._placeholder(
                'Run segmentation first, then switch to this view')
            return

        ok = [s for s in self.viewer_segments if s['status'] == 'ok']
        if not ok:
            self._placeholder('No accepted segments to display')
            return

        mono = _to_mono(self.viewer_audio).astype(np.float32)
        t_ax = np.arange(len(mono), dtype=np.float32) / self.viewer_sr

        self.ax.clear()
        self.ax.plot(t_ax, mono, color='#1565C0', lw=0.4, alpha=0.5)

        clrs = plt.cm.tab20(np.linspace(0, 1, 20))
        for i, seg in enumerate(ok):
            self.ax.axvspan(seg['start_sample'] / self.viewer_sr,
                            seg['end_sample'] / self.viewer_sr,
                            alpha=0.28, color=clrs[i % 20])

        self.ax.set_xlabel('Time (s)', fontsize=9)
        self.ax.set_ylabel('Amplitude', fontsize=9)
        self.ax.set_title(
            f'{len(ok)} segments — drag to pan, buttons to zoom',
            fontsize=9, fontweight='bold')
        self.ax.grid(True, alpha=0.2)
        self.ax.set_xlim(0, len(mono) / self.viewer_sr)

        self._viewer_ok_segs = ok
        self.seg_slider.config(to=max(0, len(ok) - 1))
        self.seg_var.set(0)
        self.seg_label.config(text=f'1/{len(ok)}')
        self.fig.tight_layout(pad=0.6)
        self.mpl_canvas.draw_idle()

    # ── viewer navigation ─────────────────────────────────────────────────────

    def _v_fit(self):
        if self.viewer_audio is not None and self.viewer_sr:
            self.ax.set_xlim(
                0, len(_to_mono(self.viewer_audio)) / self.viewer_sr)
            self.mpl_canvas.draw_idle()

    def _v_zin(self):
        xl = self.ax.get_xlim()
        c = (xl[0] + xl[1]) / 2; w = (xl[1] - xl[0]) / 3
        self.ax.set_xlim(c - w, c + w)
        self.mpl_canvas.draw_idle()

    def _v_zout(self):
        if self.viewer_audio is not None and self.viewer_sr:
            xl = self.ax.get_xlim()
            c = (xl[0] + xl[1]) / 2; w = (xl[1] - xl[0]) * 1.6
            tmax = len(_to_mono(self.viewer_audio)) / self.viewer_sr
            self.ax.set_xlim(max(0, c - w), min(tmax, c + w))
            self.mpl_canvas.draw_idle()

    def _v_pleft(self):
        xl = self.ax.get_xlim()
        sh = (xl[1] - xl[0]) * 0.25; w = xl[1] - xl[0]
        self.ax.set_xlim(max(0, xl[0] - sh), max(w, xl[1] - sh))
        self.mpl_canvas.draw_idle()

    def _v_pright(self):
        if self.viewer_audio is None or not self.viewer_sr:
            return
        xl = self.ax.get_xlim()
        sh = (xl[1] - xl[0]) * 0.25; w = xl[1] - xl[0]
        tmax = len(_to_mono(self.viewer_audio)) / self.viewer_sr
        self.ax.set_xlim(min(tmax - w, xl[0] + sh), min(tmax, xl[1] + sh))
        self.mpl_canvas.draw_idle()

    def _v_goto_seg(self, _=None):
        ok = getattr(self, '_viewer_ok_segs', [])
        idx = self.seg_var.get()
        if ok and 0 <= idx < len(ok) and self.viewer_sr:
            seg = ok[idx]
            st = seg['start_sample'] / self.viewer_sr
            en = seg['end_sample'] / self.viewer_sr
            mg = (en - st) * 0.8
            self.ax.set_xlim(max(0, st - mg), en + mg)
            self.mpl_canvas.draw_idle()
        self.seg_label.config(
            text=f'{idx + 1}/{len(ok)}' if ok else '0/0')

    # ── mouse drag-pan ────────────────────────────────────────────────────────

    def _on_press(self, e):
        if self.right_mode == 'viewer' and e.inaxes == self.ax:
            self._drag_x = e.xdata

    def _on_release(self, _):
        self._drag_x = None

    def _on_motion(self, e):
        if (self.right_mode == 'viewer' and self._drag_x is not None
                and e.inaxes == self.ax and e.xdata is not None
                and self.viewer_audio is not None and self.viewer_sr):
            dx = e.xdata - self._drag_x
            xl = self.ax.get_xlim()
            tmax = len(_to_mono(self.viewer_audio)) / self.viewer_sr
            w = xl[1] - xl[0]
            new0 = max(0, xl[0] - dx)
            new1 = new0 + w
            if new1 > tmax:
                new1 = tmax; new0 = tmax - w
            self.ax.set_xlim(new0, new1)
            self.mpl_canvas.draw_idle()

    # ── placeholder ───────────────────────────────────────────────────────────

    def _placeholder(self, msg):
        if not HAS_MATPLOTLIB:
            return
        self.ax.clear()
        self.ax.text(0.5, 0.5, msg, ha='center', va='center',
                     fontsize=11, color='#9E9E9E',
                     transform=self.ax.transAxes)
        self.ax.set_axis_off()
        self.fig.tight_layout(pad=0.6)
        self.mpl_canvas.draw_idle()

    # ── segmentation ──────────────────────────────────────────────────────────

    def _start_segmentation(self):
        if self.prev_audio is None:
            messagebox.showwarning(
                'Warning', 'No session loaded for segmentation')
            return
        self._cancel_flag = False
        params = self._params()
        self.process_btn.config(state=tk.DISABLED)
        self.prog_var.set(0)
        self.results_text.delete(1.0, tk.END)
        if self.batch_sessions:
            threading.Thread(target=self._run_batch, args=(params,),
                             daemon=True).start()
        else:
            threading.Thread(target=self._run_single, args=(params,),
                             daemon=True).start()

    def _run_single(self, params):
        try:
            audio  = self.prev_audio
            sr     = self.prev_sr
            log    = self.prev_log
            folder = self.prev_path

            out = os.path.join(os.path.dirname(folder),
                               params['output_name'])
            os.makedirs(out, exist_ok=True)

            def _prog(i, n):
                if self._cancel_flag:
                    raise InterruptedError
                self._safe_ui(lambda p=i / n * 100: self.prog_var.set(p))
                self._safe_ui(lambda a=i, b=n:
                              self.prog_label.config(
                                  text=f'Processing {a}/{b}…'))

            segs, stats = extract_keystrokes(
                audio, sr, log,
                offset=self._total_offset(),
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

            kc, meta = self._save_segments(segs, out)
            self._save_metadata(meta, out)
            self._save_rejections(segs, out)

            vname = os.path.basename(folder)
            self._safe_ui(lambda a=audio, s=sr, sg=segs, n=vname:
                          self._register_viewer_session(n, a, s, sg))

            res = self._fmt_results(stats, kc, out)
            self._safe_ui(lambda: self.results_text.insert(1.0, res))
            self._safe_ui(lambda: self.prog_var.set(100))
            self._safe_ui(lambda: self.prog_label.config(text='Done ✓'))
            self._safe_ui(
                lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(
                lambda: self.show_viewer_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: messagebox.showinfo(
                'Done',
                f"Saved {stats['saved']} / {stats['total_csv']}"))

        except InterruptedError:
            self._safe_ui(
                lambda: self.prog_label.config(text='Cancelled'))
            self._safe_ui(
                lambda: self.process_btn.config(state=tk.NORMAL))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._safe_ui(
                lambda: self.prog_label.config(text=f'Error: {e}'))
            self._safe_ui(
                lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(
                lambda: messagebox.showerror('Error', str(e)))

    def _run_batch(self, params):
        try:
            first = self.batch_sessions[0]
            out = os.path.join(os.path.dirname(first),
                               params['output_name'])
            os.makedirs(out, exist_ok=True)
            n = len(self.batch_sessions)
            ov = dict(total_sessions=n, processed=0, failed=0,
                      total_csv=0, total_saved=0,
                      key_counts={}, details=[])
            cumulative_counts: Dict[str, int] = {}  # shared across sessions
            all_meta = []   # accumulated metadata
            all_segs = []   # accumulated segments for rejections

            for si, folder in enumerate(self.batch_sessions):
                if self._cancel_flag:
                    break
                nm = os.path.basename(folder)
                self._safe_ui(
                    lambda p=si / n * 100: self.prog_var.set(p))
                self._safe_ui(
                    lambda a=si + 1, b=n, nm=nm:
                    self.prog_label.config(
                        text=f'Session {a}/{b}: {nm}'))
                try:
                    if folder in self.session_cache:
                        audio, sr, log = self.session_cache[folder]
                    else:
                        audio, sr = self.audio.load_audio(
                            os.path.join(folder, 'audio.wav'))
                        if audio is None:
                            raise RuntimeError('load_audio returned None')
                        log = self._read_log(
                            os.path.join(folder, 'keystroke_log.csv'))
                        self.session_cache[folder] = (audio, sr, log)

                    print(f'\n[{si + 1}/{n}] {nm}: '
                          f'{len(log)} keys, {len(audio) / sr:.1f} s')
                    off = auto_calculate_offset(audio, sr, log)

                    segs, stats = extract_keystrokes(
                        audio, sr, log,
                        offset=off,
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
                    kc, meta = self._save_segments(segs, out, cumulative_counts)
                    all_meta.extend(meta)
                    all_segs.extend(segs)
                    ov['processed'] += 1
                    ov['total_csv'] += stats['total_csv']
                    ov['total_saved'] += stats['saved']
                    for k, c in kc.items():
                        ov['key_counts'][k] = \
                            ov['key_counts'].get(k, 0) + c
                    ov['details'].append(
                        {'name': nm, 'stats': stats, 'ok': True})
                    self._safe_ui(
                        lambda a=audio, s=sr, sg=segs, n=nm:
                        self._register_viewer_session(n, a, s, sg))

                except Exception as e:
                    import traceback; traceback.print_exc()
                    ov['failed'] += 1
                    ov['details'].append(
                        {'name': nm, 'ok': False, 'error': str(e)})

            self._write_batch_summary(ov, out)
            self._save_metadata(all_meta, out)
            self._save_rejections(all_segs, out)
            res = self._fmt_batch(ov, out)
            self._safe_ui(lambda: self.results_text.insert(1.0, res))
            self._safe_ui(lambda: self.prog_var.set(100))
            self._safe_ui(
                lambda: self.prog_label.config(text='Batch done ✓'))
            self._safe_ui(
                lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui(
                lambda: self.show_viewer_btn.config(state=tk.NORMAL))
            self._safe_ui(lambda: messagebox.showinfo(
                'Batch Done',
                f"{ov['processed']}/{n} sessions\n"
                f"{ov['total_saved']}/{ov['total_csv']} saved"))
        except Exception as e:
            import traceback; traceback.print_exc()
            self._safe_ui(
                lambda: self.prog_label.config(text=f'Error: {e}'))
            self._safe_ui(
                lambda: self.process_btn.config(state=tk.NORMAL))

    # ── helpers ───────────────────────────────────────────────────────────────

    def _params(self) -> Dict:
        return dict(
            output_name=self.output_name_var.get(),
            segment_duration=self.segment_duration.get(),
            pre_trigger=self.pre_trigger.get(),
            search_radius=self.search_radius.get(),
            peak_snr_db=self.peak_snr_db.get(),
            min_crest_factor=self.min_crest_factor.get(),
            enable_template=self.enable_template.get(),
            template_corr=self.template_corr_thresh.get(),
            enable_filtering=self.enable_filtering.get(),
            filter_low=self.filter_low.get(),
            filter_high=self.filter_high.get(),
        )

    def _slider(self, parent, label, var, lo, hi, res):
        row = tk.Frame(parent); row.pack(fill=tk.X, pady=2)
        tk.Label(row, text=label, width=26,
                 anchor=tk.W).pack(side=tk.LEFT, padx=4)
        tk.Scale(row, from_=lo, to=hi, resolution=res,
                 orient=tk.HORIZONTAL, variable=var,
                 length=200).pack(side=tk.LEFT, padx=4)

    def _toggle_filter(self):
        st = tk.NORMAL if self.enable_filtering.get() else tk.DISABLED
        self.flt_lo_entry.config(state=st)
        self.flt_hi_entry.config(state=st)

    def _set_filter(self, lo, hi):
        self.filter_low.set(lo); self.filter_high.set(hi)
        self.enable_filtering.set(True); self._toggle_filter()

    def _safe_ui(self, fn):
        try:
            self.parent.after_idle(fn)
        except Exception:
            pass

    # ── save: one file per keystroke under output/<key>/N.wav ─────────────────

    def _save_segments(self, segments, out, cumulative_counts=None):
        """
        Save segments to out/<key>/N.wav.

        cumulative_counts: optional Dict[str, int] that persists across
        batch calls so file numbering continues (session 2 starts where
        session 1 left off instead of overwriting from 0).
        If None (single mode), a fresh dict is created.
        Returns (session_kc, meta) where session_kc counts THIS call only.
        """
        if cumulative_counts is None:
            cumulative_counts = {}
        session_kc: Dict[str, int] = {}   # counts for THIS call only
        meta = []

        for seg in segments:
            if seg['status'] != 'ok':
                continue
            key = seg['key']
            kf  = os.path.join(out, key)
            os.makedirs(kf, exist_ok=True)

            n  = cumulative_counts.get(key, 0)
            fp = os.path.join(kf, f'{n}.wav')

            if self.audio.save_audio(fp, seg['audio']):
                cumulative_counts[key] = n + 1
                session_kc[key] = session_kc.get(key, 0) + 1
                meta.append(dict(
                    key=key,
                    filename=f'{n}.wav',
                    timestamp=seg['timestamp'],
                    csv_time=f"{seg['csv_time']:.4f}",
                    peak_sample=seg['peak_sample'],
                    start_sample=seg['start_sample'],
                    snr_db=f"{seg['snr_db']:.1f}",
                    peak_amp=f"{seg['peak_amp']:.6f}",
                    crest_factor=f"{seg['crest_factor']:.2f}",
                    template_corr=f"{seg['template_corr']:.3f}",
                ))
        return session_kc, meta

    @staticmethod
    def _save_metadata(meta, out):
        if not meta:
            return
        with open(os.path.join(out, 'metadata.csv'), 'w',
                  newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(meta[0].keys()))
            w.writeheader(); w.writerows(meta)

    @staticmethod
    def _save_rejections(segs, out):
        rej = [s for s in segs if s['status'] != 'ok']
        if not rej:
            return
        fields = ['key', 'csv_time', 'status', 'peak_sample',
                  'snr_db', 'peak_amp', 'crest_factor', 'template_corr']
        with open(os.path.join(out, 'rejections.csv'), 'w',
                  newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields,
                               extrasaction='ignore')
            w.writeheader()
            for s in rej:
                row = {k: s.get(k, '') for k in fields}
                for fk in ('csv_time', 'snr_db', 'peak_amp',
                           'crest_factor', 'template_corr'):
                    if isinstance(row[fk], float):
                        row[fk] = f'{row[fk]:.4f}'
                w.writerow(row)

    @staticmethod
    def _read_log(path) -> List[Dict]:
        log = []
        with open(path, 'r') as f:
            for row in csv.DictReader(f):
                row['relative_time'] = float(row['relative_time'])
                log.append(row)
        return log

    @staticmethod
    def _fmt_results(stats, kc, out) -> str:
        s = stats; tot = s['total_csv']
        saved = s['saved']
        t = (f"SEGMENTATION COMPLETE\n{'=' * 50}\n"
             f"CSV entries : {tot}\n"
             f"Saved       : {saved}  ({100 * saved / max(1, tot):.1f}%)\n"
             f"Rejected    : {tot - saved}\n\n"
             f"Rejection breakdown:\n"
             f"  Boundary   : {s['rejected_silent']}\n"
             f"  Low peak   : {s['rejected_low_peak']}\n"
             f"  Low SNR    : {s['rejected_low_snr']}\n"
             f"  Low crest  : {s['rejected_low_transient']}\n"
             f"  Template   : {s['rejected_template']}\n"
             f"  Overlap    : {s['rejected_overlap']}\n\n"
             f"Per-key counts:\n")
        for k in sorted(kc):
            t += f"  {k:14s}: {kc[k]:4d}\n"
        t += f"\nOutput: {out}/<key>/N.wav\n"
        return t

    @staticmethod
    def _fmt_batch(ov, out) -> str:
        t = (f"BATCH COMPLETE\n{'=' * 50}\n"
             f"Sessions : {ov['processed']}/{ov['total_sessions']}"
             f"  (failed: {ov['failed']})\n"
             f"Saved    : {ov['total_saved']}/{ov['total_csv']}"
             f"  ({100 * ov['total_saved'] / max(1, ov['total_csv']):.1f}%)"
             f"\n\nKeys:\n")
        for k in sorted(ov['key_counts']):
            t += f"  {k:14s}: {ov['key_counts'][k]:4d}\n"
        t += f"\nOutput: {out}\n"
        return t

    @staticmethod
    def _write_batch_summary(ov, out):
        with open(os.path.join(out, 'batch_summary.txt'), 'w') as f:
            f.write(f"Sessions: {ov['processed']}/{ov['total_sessions']}\n")
            f.write(f"Saved: {ov['total_saved']}/{ov['total_csv']}\n\n")
            for d in ov['details']:
                if d['ok']:
                    st = d['stats']
                    f.write(f"{d['name']}: {st['saved']}/{st['total_csv']}\n")
                else:
                    f.write(f"{d['name']}: FAILED — {d['error']}\n")