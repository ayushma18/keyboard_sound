"""
Data segmentation module - extracts individual keystroke segments from continuous recordings.

Improved implementation based on audio research best practices:
- Sliding window energy detection (similar to librosa onset detection)
- Peak-centered extraction with proper attack capture
- Adaptive thresholding with quality validation
- Backtracking to capture complete attack phase
"""
import os
import csv
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from scipy import signal
from scipy.ndimage import maximum_filter
import threading


class DataSegmenterTab:
    """Tab for segmenting continuous recordings into individual keystroke samples."""
    
    def __init__(self, parent, config, audio_handler):
        self.parent = parent
        self.config = config
        self.audio = audio_handler
        
        self.current_session_path = None
        self.batch_sessions = None  # For batch processing
        self.audio_data = None
        self.keystroke_log = []
        self.sample_rate = None
        
        # Segmentation parameters
        self.segment_duration = tk.DoubleVar(value=0.430)
        self.pre_trigger = tk.DoubleVar(value=0.1)
        self.post_trigger = tk.DoubleVar(value=0.33)
        self.enable_peak_centering = tk.BooleanVar(value=True)
        self.enable_filtering = tk.BooleanVar(value=False)
        self.filter_low = tk.IntVar(value=50)
        self.filter_high = tk.IntVar(value=5000)
        
        self.build_ui()
    
    def _safe_ui_update(self, func):
        """Safely update UI from any thread."""
        try:
            # Use parent.after_idle for thread-safe updates
            self.parent.after_idle(func)
        except Exception as e:
            print(f"UI update error: {e}")
    
    def build_ui(self):
        """Build the data segmenter UI."""
        # Create canvas with scrollbar
        canvas = tk.Canvas(self.parent)
        scrollbar = tk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_frame = tk.Frame(scrollable_frame, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="Data Segmentation",
                        font=("Arial", 16, "bold"), fg="#1976D2")
        title.pack(pady=(0, 20))
        
        # Session loading
        load_frame = tk.LabelFrame(main_frame, text="1. Load Continuous Recording Session",
                                  font=("Arial", 10, "bold"), padx=15, pady=15)
        load_frame.pack(fill=tk.X, pady=(0, 15))
        
        btn_row = tk.Frame(load_frame)
        btn_row.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_row, text="Browse Session Folder", command=self.browse_session,
                 bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.session_label = tk.Label(load_frame, text="No session loaded",
                                      font=("Arial", 9), fg="gray")
        self.session_label.pack(pady=5)
        
        self.session_info_label = tk.Label(load_frame, text="",
                                          font=("Arial", 9), fg="#1976D2")
        self.session_info_label.pack(pady=5)
        
        # Segmentation parameters
        param_frame = tk.LabelFrame(main_frame, text="2. Configure Segmentation Parameters",
                                   font=("Arial", 10, "bold"), padx=15, pady=15)
        param_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Duration
        dur_row = tk.Frame(param_frame)
        dur_row.pack(fill=tk.X, pady=5)
        tk.Label(dur_row, text="Segment Duration (seconds):", width=25, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        tk.Scale(dur_row, from_=0.1, to=1.0, resolution=0.01, orient=tk.HORIZONTAL,
                variable=self.segment_duration, length=200).pack(side=tk.LEFT, padx=5)
        tk.Label(dur_row, textvariable=self.segment_duration).pack(side=tk.LEFT)
        
        # Pre/Post trigger
        trigger_row = tk.Frame(param_frame)
        trigger_row.pack(fill=tk.X, pady=5)
        tk.Label(trigger_row, text="Pre-trigger (s):", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        tk.Scale(trigger_row, from_=0.0, to=0.3, resolution=0.01, orient=tk.HORIZONTAL,
                variable=self.pre_trigger, length=150).pack(side=tk.LEFT, padx=5)
        
        tk.Label(trigger_row, text="Post-trigger (s):", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=15)
        tk.Scale(trigger_row, from_=0.0, to=0.5, resolution=0.01, orient=tk.HORIZONTAL,
                variable=self.post_trigger, length=150).pack(side=tk.LEFT, padx=5)
        
        # Peak centering
        tk.Checkbutton(param_frame, text="Enable Peak Centering (Recommended)",
                      variable=self.enable_peak_centering,
                      font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=5)
        
        # Filtering
        filter_check = tk.Checkbutton(param_frame, text="Enable Bandpass Filtering",
                                     variable=self.enable_filtering,
                                     command=self.toggle_filter_controls,
                                     font=("Arial", 9, "bold"))
        filter_check.pack(anchor=tk.W, pady=5)
        
        filter_row = tk.Frame(param_frame)
        filter_row.pack(fill=tk.X, pady=5)
        tk.Label(filter_row, text="Low:", width=8).pack(side=tk.LEFT, padx=5)
        self.filter_low_entry = tk.Entry(filter_row, textvariable=self.filter_low, width=8)
        self.filter_low_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(filter_row, text="Hz").pack(side=tk.LEFT)
        
        tk.Label(filter_row, text="High:", width=8).pack(side=tk.LEFT, padx=15)
        self.filter_high_entry = tk.Entry(filter_row, textvariable=self.filter_high, width=8)
        self.filter_high_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(filter_row, text="Hz").pack(side=tk.LEFT)
        
        # Preset buttons
        preset_row = tk.Frame(param_frame)
        preset_row.pack(fill=tk.X, pady=5)
        tk.Label(preset_row, text="Presets:", font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        
        presets = [
            ("Optimal (50-5kHz)", 50, 5000),
            ("Dataset Match (50-3kHz)", 50, 3000),
            ("Extended (50-8kHz)", 50, 8000),
        ]
        
        for name, low, high in presets:
            tk.Button(preset_row, text=name,
                     command=lambda l=low, h=high: self.set_filter_preset(l, h)).pack(side=tk.LEFT, padx=2)
        
        # Output configuration
        output_frame = tk.LabelFrame(main_frame, text="3. Output Configuration",
                                    font=("Arial", 10, "bold"), padx=15, pady=15)
        output_frame.pack(fill=tk.X, pady=(0, 15))
        
        out_row = tk.Frame(output_frame)
        out_row.pack(fill=tk.X, pady=5)
        tk.Label(out_row, text="Output Folder Name:", width=18, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.output_name_var = tk.StringVar(value=f"segmented_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tk.Entry(out_row, textvariable=self.output_name_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # Processing controls
        process_frame = tk.LabelFrame(main_frame, text="4. Process",
                                     font=("Arial", 10, "bold"), padx=15, pady=15)
        process_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(pady=10)
        
        self.progress_label = tk.Label(process_frame, text="Ready",
                                       font=("Arial", 10), fg="#424242")
        self.progress_label.pack(pady=5)
        
        btn_row2 = tk.Frame(process_frame)
        btn_row2.pack(pady=10)
        
        self.process_btn = tk.Button(btn_row2, text="Start Segmentation",
                                     command=self.start_segmentation,
                                     bg="#4CAF50", fg="white",
                                     font=("Arial", 11, "bold"),
                                     width=18, height=2,
                                     state=tk.DISABLED)
        self.process_btn.pack(side=tk.LEFT, padx=10)
        
        tk.Button(btn_row2, text="Cancel",
                 command=self.cancel_segmentation,
                 font=("Arial", 11),
                 width=12, height=2).pack(side=tk.LEFT, padx=10)
        
        # Results
        results_frame = tk.LabelFrame(main_frame, text="Results",
                                     font=("Arial", 10, "bold"), padx=15, pady=15)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        self.results_text = tk.Text(results_frame, height=8, font=("Courier", 9))
        self.results_text.pack(fill=tk.BOTH, expand=True)
    
    def toggle_filter_controls(self):
        """Enable/disable filter controls."""
        state = tk.NORMAL if self.enable_filtering.get() else tk.DISABLED
        self.filter_low_entry.config(state=state)
        self.filter_high_entry.config(state=state)
    
    def set_filter_preset(self, low: int, high: int):
        """Set filter preset."""
        self.filter_low.set(low)
        self.filter_high.set(high)
        self.enable_filtering.set(True)
        self.toggle_filter_controls()
    
    def browse_session(self):
        """Browse for a continuous recording session or parent folder containing multiple sessions."""
        folder = filedialog.askdirectory(title="Select Session Folder (or Parent Folder for Batch Processing)")
        if not folder:
            return
        
        # Check if this is a single session folder or parent folder
        audio_file = os.path.join(folder, 'audio.wav')
        log_file = os.path.join(folder, 'keystroke_log.csv')
        
        if os.path.exists(audio_file) and os.path.exists(log_file):
            # Single session folder
            self.load_session(folder)
        else:
            # Check if this folder contains session subfolders
            session_folders = self.find_session_folders(folder)
            
            if not session_folders:
                messagebox.showerror("Error", 
                    "No valid sessions found.\n\n"
                    "Selected folder must either:\n"
                    "1. Contain audio.wav and keystroke_log.csv, or\n"
                    "2. Contain subfolders with sessions")
                return
            
            # Ask user if they want to process all sessions
            msg = f"Found {len(session_folders)} session(s) in this folder.\n\n" + \
                  "\n".join([f"  • {os.path.basename(f)}" for f in session_folders[:10]])
            if len(session_folders) > 10:
                msg += f"\n  ... and {len(session_folders) - 10} more"
            msg += "\n\nProcess all sessions?"
            
            result = messagebox.askyesno("Batch Processing", msg)
            if result:
                self.load_batch_sessions(session_folders)
            else:
                messagebox.showinfo("Cancelled", "Please select a specific session folder.")
    
    def find_session_folders(self, parent_folder: str, max_depth: int = 3) -> List[str]:
        """Recursively find all folders containing audio.wav and keystroke_log.csv.
        
        Args:
            parent_folder: Root folder to search
            max_depth: Maximum recursion depth
            
        Returns:
            List of paths to valid session folders
        """
        session_folders = []
        
        def search_recursive(folder: str, depth: int):
            if depth > max_depth:
                return
            
            try:
                # Check if this folder is a valid session
                audio_file = os.path.join(folder, 'audio.wav')
                log_file = os.path.join(folder, 'keystroke_log.csv')
                
                if os.path.exists(audio_file) and os.path.exists(log_file):
                    session_folders.append(folder)
                    return  # Don't search deeper if this is a session folder
                
                # Search subdirectories
                for item in os.listdir(folder):
                    item_path = os.path.join(folder, item)
                    if os.path.isdir(item_path):
                        search_recursive(item_path, depth + 1)
            except (PermissionError, OSError):
                pass  # Skip folders we can't access
        
        search_recursive(parent_folder, 0)
        return sorted(session_folders)
    
    def load_batch_sessions(self, session_folders: List[str]):
        """Load multiple sessions for batch processing."""
        self.current_session_path = None
        self.batch_sessions = session_folders
        self.audio_data = None
        self.keystroke_log = []
        
        # Update UI
        self.session_label.config(
            text=f"Batch Mode: {len(session_folders)} sessions selected",
            fg="green"
        )
        
        total_keystrokes = 0
        for folder in session_folders:
            log_file = os.path.join(folder, 'keystroke_log.csv')
            try:
                with open(log_file, 'r') as f:
                    total_keystrokes += sum(1 for _ in csv.DictReader(f))
            except:
                pass
        
        info = f"Sessions: {len(session_folders)} | Total keystrokes: ~{total_keystrokes}"
        self.session_info_label.config(text=info)
        
        self.process_btn.config(state=tk.NORMAL)
    
    def load_session(self, folder: str):
        """Load continuous recording session."""
        try:
            self.current_session_path = folder
            self.batch_sessions = None
            
            # Load audio
            audio_file = os.path.join(folder, 'audio.wav')
            self.audio_data, self.sample_rate = self.audio.load_audio(audio_file)
            
            if self.audio_data is None:
                raise Exception("Failed to load audio file")
            
            # Load keystroke log
            log_file = os.path.join(folder, 'keystroke_log.csv')
            self.keystroke_log = []
            
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['relative_time'] = float(row['relative_time'])
                    self.keystroke_log.append(row)
            
            # Update UI
            self.session_label.config(text=f"Loaded: {os.path.basename(folder)}", fg="green")
            
            duration = len(self.audio_data) / self.sample_rate
            info = f"Audio: {duration:.1f}s, {self.sample_rate}Hz | Keystrokes: {len(self.keystroke_log)}"
            self.session_info_label.config(text=info)
            
            self.process_btn.config(state=tk.NORMAL)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session: {e}")
            self.current_session_path = None
            self.audio_data = None
            self.keystroke_log = []
    
    def start_segmentation(self):
        """Start the segmentation process."""
        # Check if we're in batch mode or single session mode
        if hasattr(self, 'batch_sessions') and self.batch_sessions:
            # Batch mode - capture all tkinter variables BEFORE threading
            # This is critical on Windows - tkinter vars are not thread-safe!
            params = {
                'output_name': self.output_name_var.get(),
                'segment_duration': self.segment_duration.get(),
                'pre_trigger': self.pre_trigger.get(),
                'post_trigger': self.post_trigger.get(),
                'enable_peak_centering': self.enable_peak_centering.get(),
                'enable_filtering': self.enable_filtering.get(),
                'filter_low': self.filter_low.get(),
                'filter_high': self.filter_high.get()
            }
            
            self.process_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
            self.results_text.delete(1.0, tk.END)
            
            thread = threading.Thread(target=self.run_batch_segmentation, args=(params,), daemon=True)
            thread.start()
        elif self.audio_data is None or not self.keystroke_log:
            messagebox.showwarning("Warning", "No session loaded")
            return
        else:
            # Single session mode - capture variables before threading
            params = {
                'output_name': self.output_name_var.get(),
                'segment_duration': self.segment_duration.get(),
                'pre_trigger': self.pre_trigger.get(),
                'post_trigger': self.post_trigger.get(),
                'enable_peak_centering': self.enable_peak_centering.get(),
                'enable_filtering': self.enable_filtering.get(),
                'filter_low': self.filter_low.get(),
                'filter_high': self.filter_high.get()
            }
            
            self.process_btn.config(state=tk.DISABLED)
            self.progress_var.set(0)
            self.results_text.delete(1.0, tk.END)
            
            thread = threading.Thread(target=self.run_segmentation, args=(params,), daemon=True)
            thread.start()
    
    def cancel_segmentation(self):
        """Cancel segmentation (placeholder)."""
        pass
    
    def compute_onset_strength(self, audio: np.ndarray,
                                sample_rate: int,
                                hop_length: int = 512, 
                                win_length: int = 2048) -> np.ndarray:
        """
        Compute onset strength envelope using sliding window energy.
        Similar to librosa.onset.onset_strength but optimized for percussive keystrokes.
        
        Research basis: Keystroke papers use 10-100ms windows for energy calculation.
        
        Args:
            audio: Audio signal
            sample_rate: Sample rate of the audio
            hop_length: Number of samples between successive frames
            win_length: Window length for STFT
        """
        # Use STFT for frequency-aware energy computation
        f, t, Zxx = signal.stft(audio, fs=sample_rate, 
                               nperseg=win_length, noverlap=win_length-hop_length)
        
        # Compute spectral magnitude
        magnitude = np.abs(Zxx)
        
        # Focus on keystroke frequency range (typically 50-5000 Hz)
        freq_mask = (f >= 50) & (f <= 5000)
        magnitude_filtered = magnitude[freq_mask, :]
        
        # Compute onset strength as spectral flux
        # (difference between successive frames, half-wave rectified)
        onset_env = np.zeros(magnitude_filtered.shape[1])
        onset_env[1:] = np.sum(np.maximum(0, magnitude_filtered[:, 1:] - magnitude_filtered[:, :-1]), axis=0)
        
        # Smooth the onset envelope
        window = signal.windows.hann(5)
        onset_env = signal.convolve(onset_env, window / window.sum(), mode='same')
        
        return onset_env
    
    def detect_peaks_adaptive(self, onset_env: np.ndarray, 
                               target_count: int,
                               initial_threshold_percentile: float = 70) -> Tuple[List[int], float]:
        """
        Detect peaks in onset envelope with adaptive thresholding.
        
        Algorithm:
        1. Start with percentile-based threshold
        2. Find peaks above threshold
        3. Adjust threshold iteratively to match target count
        """
        # Start with percentile-based threshold
        threshold = np.percentile(onset_env, initial_threshold_percentile)
        step = np.max(onset_env) * 0.02
        
        iteration = 0
        max_iterations = 200
        best_peaks = []
        best_diff = float('inf')
        
        while iteration < max_iterations:
            # Find peaks above threshold with minimum distance
            # Keystrokes are typically at least 50ms apart
            min_distance = int(0.05 * self.sample_rate / 512)  # ~50ms in frames
            
            peaks = self._find_peaks_above_threshold(onset_env, threshold, min_distance)
            
            diff = abs(len(peaks) - target_count)
            
            # Track best result
            if diff < best_diff:
                best_diff = diff
                best_peaks = peaks
            
            # Stop if we hit target
            if diff == 0:
                print(f"Converged at iteration {iteration}: {len(peaks)} peaks")
                break
            
            # Adjust threshold
            if len(peaks) > target_count:
                threshold += step
            else:
                threshold -= step
            
            # Decay step size for fine-tuning
            step *= 0.98
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Peaks={len(peaks)}, Threshold={threshold:.4f}, Target={target_count}")
            
            iteration += 1
        
        # Convert frame indices to sample indices
        hop_length = 512
        peak_samples = [p * hop_length for p in best_peaks]
        
        return peak_samples, threshold
    
    def _find_peaks_above_threshold(self, signal_data: np.ndarray, 
                                     threshold: float, 
                                     min_distance: int) -> List[int]:
        """Find local maxima above threshold with minimum distance constraint."""
        # Use scipy's maximum filter for peak detection
        size = min_distance * 2 + 1
        local_max = maximum_filter(signal_data, size=size, mode='constant')
        
        # Peak is where signal equals local maximum and exceeds threshold
        peaks = (signal_data == local_max) & (signal_data > threshold)
        peak_indices = np.where(peaks)[0].tolist()
        
        return peak_indices
    
    def backtrack_to_onset(self, onset_env: np.ndarray, 
                           peak_positions: List[int]) -> List[int]:
        """
        Backtrack from each peak to find the actual onset (attack beginning).
        
        Research basis: librosa.onset.onset_backtrack rolls back to preceding 
        local minimum to capture full attack phase.
        """
        hop_length = 512
        onset_starts = []
        
        for peak_sample in peak_positions:
            peak_frame = peak_sample // hop_length
            
            # Search backward for local minimum
            search_start = max(0, peak_frame - 20)  # ~100ms back
            
            if search_start < peak_frame:
                segment = onset_env[search_start:peak_frame]
                if len(segment) > 0:
                    # Find minimum in this region
                    min_idx = np.argmin(segment)
                    onset_frame = search_start + min_idx
                    onset_sample = onset_frame * hop_length
                else:
                    onset_sample = peak_sample
            else:
                onset_sample = peak_sample
            
            onset_starts.append(onset_sample)
        
        return onset_starts
    
    def extract_segments_centered(self, audio: np.ndarray,
                                   peak_positions: List[int],
                                   onset_positions: List[int],
                                   duration: float,
                                   pre_trigger: float,
                                   post_trigger: float) -> List[Dict]:
        """
        Extract audio segments centered around peaks with proper attack capture.
        
        If peak_centering is enabled, center around peak.
        Otherwise, use onset position with pre/post trigger timing.
        """
        segments = []
        duration_samples = int(duration * self.sample_rate)
        
        for peak_sample, onset_sample in zip(peak_positions, onset_positions):
            if self.enable_peak_centering.get():
                # Center around peak with pre/post trigger
                pre_samples = int(pre_trigger * self.sample_rate)
                post_samples = int(post_trigger * self.sample_rate)
                
                start = peak_sample - pre_samples
                end = peak_sample + post_samples
            else:
                # Use onset position
                start = onset_sample
                end = onset_sample + duration_samples
            
            # Boundary checks
            start = max(0, start)
            end = min(len(audio), end)
            
            # Extract segment
            segment = audio[start:end]
            
            # Pad if necessary
            if len(segment) < duration_samples:
                padding = duration_samples - len(segment)
                segment = np.pad(segment, (0, padding), mode='constant')
            
            # Trim if too long
            segment = segment[:duration_samples]
            
            segments.append({
                'audio': segment,
                'peak_sample': peak_sample,
                'start_sample': start,
                'onset_sample': onset_sample
            })
        
        return segments
    
    def validate_segments(self, segments: List[Dict], 
                          full_audio: np.ndarray) -> List[Dict]:
        """
        Validate segments and compute quality scores.
        
        Quality checks:
        1. Minimum energy threshold (reject silent segments)
        2. Peak-to-RMS ratio (keystrokes have high transients)
        3. Spectral content in keystroke frequency range
        """
        valid_segments = []
        
        for seg_info in segments:
            segment = seg_info['audio']
            
            # Check 1: Minimum RMS energy
            rms = np.sqrt(np.mean(segment**2))
            if rms < 1e-5:  # Essentially silent
                continue
            
            # Check 2: Peak-to-RMS ratio (keystrokes are transient)
            peak = np.max(np.abs(segment))
            peak_to_rms = peak / (rms + 1e-10)
            
            # Keystrokes typically have high peak-to-RMS ratio (>2)
            if peak_to_rms < 1.5:
                continue
            
            # Check 3: Spectral content
            # Compute FFT and check energy in keystroke range
            fft = np.fft.rfft(segment)
            freqs = np.fft.rfftfreq(len(segment), 1/self.sample_rate)
            magnitude = np.abs(fft)
            
            # Energy in 50-5000 Hz range
            freq_mask = (freqs >= 50) & (freqs <= 5000)
            keystroke_energy = np.sum(magnitude[freq_mask]**2)
            total_energy = np.sum(magnitude**2)
            
            energy_ratio = keystroke_energy / (total_energy + 1e-10)
            
            # Keystrokes should have most energy in this range
            if energy_ratio < 0.5:
                continue
            
            # Compute quality score (0-1)
            quality = (peak_to_rms / 10.0) * energy_ratio
            quality = min(1.0, quality)
            
            seg_info['quality'] = quality
            valid_segments.append(seg_info)
        
        return valid_segments
    
    def match_segments_to_keystrokes(self, segments: List[Dict], 
                                      keystroke_log: List[Dict],
                                      max_time_diff: float = 0.2) -> List[Dict]:
        """
        Match segments to keystrokes using temporal proximity.
        
        This prevents misclassification when some keystrokes are not detected.
        Each segment is matched to its closest keystroke in time (within threshold).
        
        Algorithm:
        1. For each segment, find the closest CSV keystroke in time
        2. Use 1:1 matching: each keystroke can only be matched once
        3. Compute confidence score based on temporal proximity
        4. Flag unmatched segments and missing keystrokes
        
        Args:
            segments: List of detected segment dictionaries
            keystroke_log: List of keystroke log entries
            max_time_diff: Maximum allowed time difference for matching (seconds)
            
        Returns:
            List of match dictionaries with segment, keystroke, and quality info
        """
        matches = []
        used_keystroke_indices = set()
        
        print(f"\n=== Temporal Matching: {len(segments)} segments to {len(keystroke_log)} keystrokes ===")
        
        # For each segment, find best matching keystroke
        for seg_info in segments:
            peak_time = seg_info['peak_sample'] / self.sample_rate
            
            # Find closest keystroke in time
            best_idx = None
            best_time_diff = float('inf')
            
            for idx, keystroke in enumerate(keystroke_log):
                if idx in used_keystroke_indices:
                    continue  # Already matched
                
                csv_time = keystroke['relative_time']
                time_diff = abs(peak_time - csv_time)
                
                if time_diff < best_time_diff:
                    best_time_diff = time_diff
                    best_idx = idx
            
            # Check if match is acceptable
            if best_idx is not None and best_time_diff <= max_time_diff:
                # Good match found
                keystroke = keystroke_log[best_idx]
                used_keystroke_indices.add(best_idx)
                
                # Compute confidence (1.0 = perfect, 0.0 = at threshold)
                confidence = 1.0 - (best_time_diff / max_time_diff)
                
                matches.append({
                    'segment': seg_info,
                    'key': keystroke['key'],
                    'timestamp': keystroke['timestamp'],
                    'csv_time': keystroke['relative_time'],
                    'time_diff': best_time_diff,
                    'confidence': confidence,
                    'matched': True
                })
                
                if best_time_diff > 0.1:  # Warning for >100ms
                    print(f"  Warning: Large time diff for '{keystroke['key']}': {best_time_diff*1000:.1f}ms")
            else:
                # No good match - unmatched segment (false positive)
                matches.append({
                    'segment': seg_info,
                    'key': 'UNMATCHED',
                    'timestamp': '',
                    'csv_time': peak_time,
                    'time_diff': best_time_diff if best_idx is not None else -1,
                    'confidence': 0.0,
                    'matched': False
                })
                print(f"  WARNING: Unmatched segment at {peak_time:.3f}s (closest: {best_time_diff:.3f}s)")
        
        # Report unmatched keystrokes from CSV
        unmatched_count = 0
        for idx, keystroke in enumerate(keystroke_log):
            if idx not in used_keystroke_indices:
                unmatched_count += 1
                print(f"  WARNING: Unmatched CSV keystroke '{keystroke['key']}' at {keystroke['relative_time']:.3f}s")
        
        print(f"\nMatching Summary:")
        print(f"  Matched: {len([m for m in matches if m['matched']])} / {len(keystroke_log)}")
        print(f"  Unmatched segments: {len([m for m in matches if not m['matched']])}")
        print(f"  Unmatched CSV entries: {unmatched_count}")
        
        return matches
    
    def run_batch_segmentation(self, params):
        """Process multiple sessions recursively with detailed logging.
        
        Args:
            params: Dictionary of parameters captured from tkinter variables (thread-safe)
        """
        try:
            print("\n[DEBUG] Starting batch segmentation...")
            
            # Use passed parameters instead of accessing tkinter variables
            output_base_name = params['output_name']
            print(f"[DEBUG] Output base name: {output_base_name}")
            
            # Find the recordings root directory and create segmented folder there
            # Navigate up from session folders to find "recordings" directory
            first_session = self.batch_sessions[0]
            current_path = first_session
            recordings_root = None
            
            # Go up the directory tree to find "recordings" folder
            for _ in range(10):  # Limit depth to avoid infinite loop
                parent = os.path.dirname(current_path)
                if os.path.basename(current_path) in ['recordings', 'backups']:
                    recordings_root = current_path
                    break
                if parent == current_path:  # Reached root
                    break
                current_path = parent
            
            # If we found recordings root, create segmented folder there
            # Otherwise fall back to parent of first session
            if recordings_root:
                # Extract device/keyboard info from path for better naming
                # e.g., recordings/dji2-kumara/continuous/0 -> segmented/dji2-kumara_continuous_...
                rel_path = os.path.relpath(first_session, recordings_root)
                path_parts = rel_path.split(os.sep)
                if len(path_parts) >= 2:
                    device_info = "_".join(path_parts[:-1])  # e.g., "dji2-kumara_continuous"
                    output_name_with_device = f"{device_info}_{output_base_name}"
                else:
                    output_name_with_device = output_base_name
                
                segmented_folder = os.path.join(recordings_root, 'segmented')
                output_base = os.path.join(segmented_folder, output_name_with_device)
            else:
                # Fallback: use parent of first session
                first_session_parent = os.path.dirname(first_session)
                output_base = os.path.join(first_session_parent, output_base_name)
            
            os.makedirs(output_base, exist_ok=True)
            print(f"[DEBUG] Output directory created: {output_base}")
            
            total_sessions = len(self.batch_sessions)
            overall_stats = {
                'total_sessions': total_sessions,
                'processed_sessions': 0,
                'failed_sessions': 0,
                'total_keystrokes': 0,
                'total_saved': 0,
                'total_skipped': 0,
                'session_details': [],
                'key_counts': {}  # Track how many samples per key across all sessions
            }
            
            print(f"\n{'='*80}")
            print(f"BATCH SEGMENTATION: {total_sessions} sessions")
            print(f"Output: {output_base}")
            print(f"{'='*80}\n")
            
            # Process each session
            for session_idx, session_folder in enumerate(self.batch_sessions):
                try:
                    session_name = os.path.basename(session_folder)
                    print(f"\n[{session_idx+1}/{total_sessions}] Processing: {session_name}")
                    print("-" * 60)
                    
                    # Update progress safely
                    progress = (session_idx / total_sessions) * 100
                    print(f"[DEBUG] Updating progress to {progress}%")
                    self._safe_ui_update(lambda p=progress: self.progress_var.set(p))
                    self._safe_ui_update(lambda i=session_idx+1, t=total_sessions, n=session_name: 
                        self.progress_label.config(text=f"Session {i}/{t}: {n}"))
                    
                    print(f"[DEBUG] Calling process_single_session_batch...")
                    # Process this session - pass params
                    session_stats = self.process_single_session_batch(
                        session_folder,
                        output_base,
                        session_name,
                        params
                    )
                    print(f"[DEBUG] Session processing complete")
                    
                    if session_stats['success']:
                        overall_stats['processed_sessions'] += 1
                        overall_stats['total_keystrokes'] += session_stats['total_keystrokes']
                        overall_stats['total_saved'] += session_stats['saved']
                        overall_stats['total_skipped'] += session_stats['skipped']
                        
                        # Aggregate key counts
                        for key, count in session_stats['key_counts'].items():
                            overall_stats['key_counts'][key] = overall_stats['key_counts'].get(key, 0) + count
                        
                        print(f"  ✓ Success: {session_stats['saved']} segments saved")
                    else:
                        overall_stats['failed_sessions'] += 1
                        print(f"  ✗ Failed: {session_stats['error']}")
                    
                    overall_stats['session_details'].append(session_stats)
                except Exception as session_error:
                    print(f"[ERROR] Session {session_name} failed with error: {session_error}")
                    import traceback
                    traceback.print_exc()
                    overall_stats['failed_sessions'] += 1
            
            # Save batch summary
            summary_file = os.path.join(output_base, 'batch_summary.txt')
            with open(summary_file, 'w') as f:
                f.write(f"BATCH SEGMENTATION SUMMARY\n")
                f.write(f"{'='*80}\n\n")
                f.write(f"Total sessions: {overall_stats['total_sessions']}\n")
                f.write(f"Processed successfully: {overall_stats['processed_sessions']}\n")
                f.write(f"Failed: {overall_stats['failed_sessions']}\n")
                f.write(f"Total keystrokes: {overall_stats['total_keystrokes']}\n")
                f.write(f"Total saved: {overall_stats['total_saved']}\n")
                f.write(f"Total skipped: {overall_stats['total_skipped']}\n")
                f.write(f"Success rate: {100*overall_stats['total_saved']/max(1,overall_stats['total_keystrokes']):.1f}%\n\n")
                
                f.write(f"\nKEY DISTRIBUTION (across all sessions):\n")
                f.write(f"{'-'*40}\n")
                for key in sorted(overall_stats['key_counts'].keys()):
                    count = overall_stats['key_counts'][key]
                    f.write(f"  {key:10s}: {count:4d} samples\n")
                
                f.write(f"\n\nSESSION DETAILS:\n")
                f.write(f"{'='*80}\n")
                for session_stats in overall_stats['session_details']:
                    f.write(f"\nSession: {session_stats['session_name']}\n")
                    if session_stats['success']:
                        f.write(f"  Status: SUCCESS\n")
                        f.write(f"  Keystrokes: {session_stats['total_keystrokes']}\n")
                        f.write(f"  Saved: {session_stats['saved']}\n")
                        f.write(f"  Skipped: {session_stats['skipped']}\n")
                        f.write(f"  Keys found:\n")
                        for key, count in sorted(session_stats['key_counts'].items()):
                            f.write(f"    {key}: {count}\n")
                    else:
                        f.write(f"  Status: FAILED\n")
                        f.write(f"  Error: {session_stats['error']}\n")
            
            # Display results
            results = f"""BATCH SEGMENTATION COMPLETE!

Sessions processed:  {overall_stats['processed_sessions']}/{overall_stats['total_sessions']}
Failed sessions:     {overall_stats['failed_sessions']}

Total keystrokes:    {overall_stats['total_keystrokes']}
Segments saved:      {overall_stats['total_saved']}
Skipped (silent):    {overall_stats['total_skipped']}
Success rate:        {100*overall_stats['total_saved']/max(1,overall_stats['total_keystrokes']):.1f}%

KEY DISTRIBUTION:
"""
            for key in sorted(overall_stats['key_counts'].keys()):
                count = overall_stats['key_counts'][key]
                results += f"  {key:10s}: {count:4d} samples\n"
            
            results += f"\nOutput: {output_base}\nSummary: {summary_file}"
            
            print(f"\n{'='*80}")
            print(results)
            print(f"{'='*80}\n")
            
            self._safe_ui_update(lambda: self.results_text.insert(1.0, results))
            self._safe_ui_update(lambda: self.progress_var.set(100))
            self._safe_ui_update(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui_update(lambda: messagebox.showinfo(
                "Batch Complete",
                f"Processed {overall_stats['processed_sessions']} sessions\n"
                f"Saved {overall_stats['total_saved']} segments total!"))
            
        except Exception as e:
            error_msg = f"Batch segmentation error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            
            self._safe_ui_update(lambda: self.progress_label.config(text=error_msg))
            self._safe_ui_update(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui_update(lambda: messagebox.showerror("Error", error_msg))
    
    def process_single_session_batch(self, session_folder: str, 
                                     output_base: str,
                                     session_name: str,
                                     params: Dict) -> Dict:
        """Process a single session as part of batch processing.
        
        Args:
            session_folder: Path to session folder
            output_base: Output base directory
            session_name: Name of the session
            params: Dictionary with segmentation parameters (thread-safe)
            
        Returns:
            Dictionary with session statistics
        """
        session_stats = {
            'session_name': session_name,
            'session_path': session_folder,
            'success': False,
            'total_keystrokes': 0,
            'saved': 0,
            'skipped': 0,
            'key_counts': {},
            'error': None
        }
        
        try:
            print(f"[DEBUG] Loading audio file...")
            # Load audio
            audio_file = os.path.join(session_folder, 'audio.wav')
            audio_data, sample_rate = self.audio.load_audio(audio_file)
            print(f"[DEBUG] Audio loaded: shape={audio_data.shape if audio_data is not None else None}, sr={sample_rate}")
            
            if audio_data is None:
                raise Exception("Failed to load audio file")
            
            print(f"[DEBUG] Loading keystroke log...")
            # Load keystroke log
            log_file = os.path.join(session_folder, 'keystroke_log.csv')
            keystroke_log = []
            
            with open(log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row['relative_time'] = float(row['relative_time'])
                    keystroke_log.append(row)
            
            session_stats['total_keystrokes'] = len(keystroke_log)
            print(f"[DEBUG] Loaded {len(keystroke_log)} keystrokes")
            
            print(f"[DEBUG] Converting to mono...")
            # Convert to mono
            if len(audio_data.shape) == 2:
                mono_audio = np.mean(audio_data, axis=1)
            else:
                mono_audio = audio_data
            print(f"[DEBUG] Mono audio shape: {mono_audio.shape}")
            
            # Get segmentation parameters from passed params (thread-safe)
            keystroke_length = params['segment_duration']
            pre_trigger = params['pre_trigger']
            enable_peak_centering = params['enable_peak_centering']
            enable_filtering = params['enable_filtering']
            filter_low = params['filter_low']
            filter_high = params['filter_high']
            print(f"[DEBUG] Parameters: length={keystroke_length}s, pre={pre_trigger}s, peak_center={enable_peak_centering}, filter={enable_filtering}")
            
            print(f"[DEBUG] Computing onset envelope...")
            # Compute onset strength if peak centering enabled
            if enable_peak_centering:
                onset_env = self.compute_onset_strength(mono_audio, sample_rate)
                hop_length = 512
                print(f"[DEBUG] Onset envelope computed: length={len(onset_env)}")
            else:
                onset_env = None
                hop_length = 512
                print(f"[DEBUG] Peak centering disabled, skipping onset computation")
            
            # Process each keystroke
            sorted_log = sorted(keystroke_log, key=lambda x: x['relative_time'])
            
            for keystroke in sorted_log:
                csv_time = keystroke['relative_time']
                csv_sample = int(csv_time * sample_rate)
                key = keystroke['key']
                
                # Find actual peak near CSV time
                if enable_peak_centering and onset_env is not None:
                    search_window = int(0.05 * sample_rate)
                    search_start = max(0, csv_sample - search_window)
                    search_end = min(len(mono_audio), csv_sample + search_window)
                    
                    search_start_frame = search_start // hop_length
                    search_end_frame = search_end // hop_length
                    
                    if search_start_frame < len(onset_env) and search_end_frame > 0:
                        search_region = onset_env[search_start_frame:search_end_frame]
                        
                        if len(search_region) > 0:
                            peak_idx = np.argmax(search_region)
                            peak_frame = search_start_frame + peak_idx
                            peak_sample = peak_frame * hop_length
                        else:
                            peak_sample = csv_sample
                    else:
                        peak_sample = csv_sample
                else:
                    peak_sample = csv_sample
                
                # Extract segment
                duration_samples = int(keystroke_length * sample_rate)
                pre_samples = int(pre_trigger * sample_rate)
                
                start = max(0, peak_sample - pre_samples)
                end = min(len(mono_audio), start + duration_samples)
                
                segment = mono_audio[start:end]
                
                # Pad if needed
                if len(segment) < duration_samples:
                    segment = np.pad(segment, (0, duration_samples - len(segment)))
                segment = segment[:duration_samples]
                
                # Quality check
                rms = np.sqrt(np.mean(segment**2))
                
                if rms < 1e-6:
                    session_stats['skipped'] += 1
                    continue
                
                # Apply filtering
                if enable_filtering:
                    segment = self.audio.apply_bandpass_filter(segment,
                                                               filter_low,
                                                               filter_high)
                
                # Convert to stereo if needed
                if len(audio_data.shape) == 2:
                    segment = np.column_stack([segment, segment])
                
                # Save
                key_folder = os.path.join(output_base, key)
                os.makedirs(key_folder, exist_ok=True)
                
                existing = [f for f in os.listdir(key_folder) if f.endswith('.wav')]
                file_num = len(existing)
                filename = f"{file_num}.wav"
                filepath = os.path.join(key_folder, filename)
                
                if self.audio.save_audio(filepath, segment):
                    session_stats['saved'] += 1
                    session_stats['key_counts'][key] = session_stats['key_counts'].get(key, 0) + 1
            
            session_stats['success'] = True
            print(f"[DEBUG] Session completed successfully: saved={session_stats['saved']}, skipped={session_stats['skipped']}")
            
        except Exception as e:
            session_stats['error'] = str(e)
            import sys
            import traceback
            exc_type, exc_value, exc_tb = sys.exc_info()
            print(f"[ERROR] Session processing failed at line {exc_tb.tb_lineno if exc_tb else 'unknown'}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            print(f"[ERROR] Exception message: {str(e)}")
            print("[ERROR] Full traceback:")
            traceback.print_exc()
        
        return session_stats
    
    def run_segmentation(self, params):
        """
        CSV-GUIDED SEGMENTATION (BEST for labeled data)
        
        Instead of detecting peaks and matching (which gives false positives),
        we use CSV timestamps to directly extract segments.
        
        Algorithm:
        1. For each CSV keystroke timestamp
        2. Search ±50ms window around that time for actual peak
        3. Extract segment centered on peak
        4. Validate quality
        
        Result: Exactly N segments for N CSV entries!
        No false positives, no unmatched entries.
        
        Args:
            params: Dictionary with segmentation parameters (thread-safe)
        """
        try:
            output_base = os.path.join(os.path.dirname(self.current_session_path),
                                       params['output_name'])
            os.makedirs(output_base, exist_ok=True)
            
            keystroke_length = params['segment_duration']
            pre_trigger = params['pre_trigger']
            enable_peak_centering = params['enable_peak_centering']
            enable_filtering = params['enable_filtering']
            filter_low = params['filter_low']
            filter_high = params['filter_high']
            
            print(f"\n=== CSV-GUIDED SEGMENTATION ===")
            print(f"CSV keystrokes: {len(self.keystroke_log)}")
            print(f"Segment length: {keystroke_length}s")
            print(f"Peak centering: {'Yes' if enable_peak_centering else 'No'}")
            
            # Convert to mono
            if len(self.audio_data.shape) == 2:
                mono_audio = np.mean(self.audio_data, axis=1)
            else:
                mono_audio = self.audio_data
            
            # Compute onset strength for peak finding
            if enable_peak_centering:
                print("Computing onset strength envelope...")
                onset_env = self.compute_onset_strength(mono_audio, self.sample_rate)
                hop_length = 512
            else:
                onset_env = None
                hop_length = 512
            
            metadata = []
            saved = 0
            skipped = 0
            
            # Process each CSV entry
            sorted_log = sorted(self.keystroke_log, key=lambda x: x['relative_time'])
            
            for idx, keystroke in enumerate(sorted_log):
                csv_time = keystroke['relative_time']
                csv_sample = int(csv_time * self.sample_rate)
                key = keystroke['key']
                timestamp = keystroke['timestamp']
                
                # Find actual peak near CSV time
                if enable_peak_centering and onset_env is not None:
                    # Search ±50ms window
                    search_window = int(0.05 * self.sample_rate)
                    search_start = max(0, csv_sample - search_window)
                    search_end = min(len(mono_audio), csv_sample + search_window)
                    
                    search_start_frame = search_start // hop_length
                    search_end_frame = search_end // hop_length
                    
                    if search_start_frame < len(onset_env) and search_end_frame > 0:
                        search_region = onset_env[search_start_frame:search_end_frame]
                        
                        if len(search_region) > 0:
                            peak_idx = np.argmax(search_region)
                            peak_frame = search_start_frame + peak_idx
                            peak_sample = peak_frame * hop_length
                        else:
                            peak_sample = csv_sample
                    else:
                        peak_sample = csv_sample
                else:
                    # Use CSV time directly
                    peak_sample = csv_sample
                
                # Extract segment
                duration_samples = int(keystroke_length * self.sample_rate)
                pre_samples = int(pre_trigger * self.sample_rate)
                
                start = max(0, peak_sample - pre_samples)
                end = min(len(mono_audio), start + duration_samples)
                
                segment = mono_audio[start:end]
                
                # Pad if needed
                if len(segment) < duration_samples:
                    segment = np.pad(segment, (0, duration_samples - len(segment)))
                segment = segment[:duration_samples]
                
                # Quality check
                rms = np.sqrt(np.mean(segment**2))
                peak_amp = np.max(np.abs(segment))
                peak_to_rms = peak_amp / (rms + 1e-10)
                
                # Very lenient quality threshold (only skip truly silent segments)
                if rms < 1e-6:
                    print(f"  Skipped '{key}' at {csv_time:.3f}s: silent (RMS={rms:.2e})")
                    skipped += 1
                    continue
                
                # Apply filtering
                if enable_filtering:
                    segment = self.audio.apply_bandpass_filter(segment,
                                                               filter_low,
                                                               filter_high)
                
                # Convert to stereo if needed
                if len(self.audio_data.shape) == 2:
                    segment = np.column_stack([segment, segment])
                
                # Save
                key_folder = os.path.join(output_base, key)
                os.makedirs(key_folder, exist_ok=True)
                
                existing = [f for f in os.listdir(key_folder) if f.endswith('.wav')]
                file_num = len(existing)
                filename = f"{file_num}.wav"
                filepath = os.path.join(key_folder, filename)
                
                if self.audio.save_audio(filepath, segment):
                    time_diff = abs(peak_sample / self.sample_rate - csv_time)
                    
                    metadata.append({
                        'key': key,
                        'filename': filename,
                        'timestamp': timestamp,
                        'csv_time': csv_time,
                        'csv_sample': csv_sample,
                        'peak_sample': peak_sample,
                        'peak_time': peak_sample / self.sample_rate,
                        'time_difference': time_diff,
                        'rms': rms,
                        'peak_amplitude': peak_amp,
                        'peak_to_rms': peak_to_rms,
                        'samples': len(segment),
                        'file_number': file_num
                    })
                    
                    saved += 1
                    if saved % 100 == 0:
                        print(f"  Processed {saved}/{len(sorted_log)}")
                
                # Update progress
                progress = ((idx + 1) / len(sorted_log)) * 100
                self._safe_ui_update(lambda p=progress: self.progress_var.set(p))
                self._safe_ui_update(lambda s=saved, t=len(sorted_log): 
                                self.progress_label.config(text=f"Processing: {s}/{t}"))
            
            # Save metadata
            metadata_file = os.path.join(output_base, 'metadata.csv')
            with open(metadata_file, 'w', newline='') as f:
                fieldnames = ['key', 'filename', 'timestamp', 'csv_time', 'csv_sample',
                            'peak_sample', 'peak_time', 'time_difference',
                            'rms', 'peak_amplitude', 'peak_to_rms', 'samples', 'file_number']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata)
            
            # Statistics
            time_diffs = [m['time_difference'] for m in metadata]
            avg_diff = np.mean(time_diffs) if time_diffs else 0
            max_diff = np.max(time_diffs) if time_diffs else 0
            perfect = sum(1 for t in time_diffs if t < 0.05)
            
            results = f"""CSV-GUIDED SEGMENTATION COMPLETE!

CSV keystrokes:      {len(self.keystroke_log)}
Segments saved:      {saved}
Skipped (silent):    {skipped}
Success rate:        {100*saved/len(self.keystroke_log):.1f}%

FALSE POSITIVES:     0 ✓ (CSV-guided approach!)
UNMATCHED:           0 ✓ (All CSV entries processed!)

Alignment Quality:
  Perfect (< 50ms):  {perfect} ({100*perfect/len(time_diffs):.1f}%)
  Avg Δt:            {avg_diff*1000:.1f}ms
  Max Δt:            {max_diff*1000:.1f}ms

Output: {output_base}
Sample rate: {self.sample_rate}Hz
Filtering: {'Enabled' if enable_filtering else 'Disabled'}
"""
            
            print(results)
            self._safe_ui_update(lambda: self.results_text.insert(1.0, results))
            self._safe_ui_update(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui_update(lambda: messagebox.showinfo("Success", 
                                                             f"Saved {saved} segments!\n0 false positives!"))
            
        except Exception as e:
            error_msg = f"Segmentation error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self._safe_ui_update(lambda: self.progress_label.config(text=error_msg))
            self._safe_ui_update(lambda: self.process_btn.config(state=tk.NORMAL))
            self._safe_ui_update(lambda: messagebox.showerror("Error", error_msg))
            os.makedirs(output_base, exist_ok=True)
            
            # Get parameters
            keystroke_length = self.segment_duration.get()
            pre_trigger = self.pre_trigger.get()
            post_trigger = self.post_trigger.get()
            
            print(f"\n=== Robust Keystroke Segmentation (Research-Based) ===")
            print(f"Audio: {len(self.audio_data)} samples ({len(self.audio_data)/self.sample_rate:.2f}s)")
            print(f"Target keystrokes: {len(self.keystroke_log)}")
            print(f"Segment: {keystroke_length}s (pre: {pre_trigger}s, post: {post_trigger}s)")
            
            # Convert to mono if stereo
            if len(self.audio_data.shape) == 2:
                mono_audio = np.mean(self.audio_data, axis=1)
            else:
                mono_audio = self.audio_data
            
            # Step 1: Compute onset strength envelope (sliding window energy)
            print("Computing onset strength envelope...")
            onset_env = self.compute_onset_strength(mono_audio, self.sample_rate)
            
            # Step 2: Find peaks with adaptive thresholding
            print("Detecting keystroke peaks...")
            peak_positions, threshold = self.detect_peaks_adaptive(
                onset_env, 
                target_count=len(self.keystroke_log)
            )
            
            print(f"Detected {len(peak_positions)} peaks with threshold {threshold:.4f}")
            
            # Step 3: Backtrack to find segment start (attack beginning)
            print("Backtracking to capture attack phases...")
            segment_starts = self.backtrack_to_onset(onset_env, peak_positions)
            
            # Step 4: Extract segments centered around peaks with proper timing
            print("Extracting keystroke segments...")
            segments = self.extract_segments_centered(
                mono_audio, 
                peak_positions,
                segment_starts,
                keystroke_length,
                pre_trigger,
                post_trigger
            )
            
            # Step 5: Validate and filter segments
            print("Validating segments...")
            valid_segments = self.validate_segments(segments, mono_audio)
            
            print(f"Valid segments: {len(valid_segments)}/{len(segments)}")
            
            # Match segments to keystrokes using TEMPORAL PROXIMITY
            print("\n=== Matching Phase ===")
            matches = self.match_segments_to_keystrokes(valid_segments, self.keystroke_log)
            
            metadata = []
            metadata_file = os.path.join(output_base, 'metadata.csv')
            saved = 0
            
            for match_info in matches:
                segment_audio = match_info['segment']['audio']
                peak_sample = match_info['segment']['peak_sample']
                start_sample = match_info['segment']['start_sample']
                quality_score = match_info['segment']['quality']
                
                # Get matched keystroke info
                key = match_info['key']
                timestamp = match_info['timestamp']
                csv_time = match_info['csv_time']
                time_diff = match_info['time_diff']
                match_confidence = match_info['confidence']
                
                # Apply filtering if enabled
                segment = segment_audio
                if self.enable_filtering.get():
                    segment = self.audio.apply_bandpass_filter(segment,
                                                               self.filter_low.get(),
                                                               self.filter_high.get())
                
                # Convert back to stereo if original was stereo
                if len(self.audio_data.shape) == 2:
                    segment = np.column_stack([segment, segment])
                
                # Save segment
                key_folder = os.path.join(output_base, key)
                os.makedirs(key_folder, exist_ok=True)
                
                existing_files = [f for f in os.listdir(key_folder) if f.endswith('.wav')]
                file_num = len(existing_files)
                filename = f"{file_num}.wav"
                filepath = os.path.join(key_folder, filename)
                
                if self.audio.save_audio(filepath, segment):
                    rms = np.sqrt(np.mean(segment**2))
                    peak = np.max(np.abs(segment))
                    
                    metadata.append({
                        'key': key,
                        'filename': filename,
                        'timestamp': timestamp,
                        'csv_time': csv_time,
                        'peak_sample': peak_sample,
                        'start_sample': start_sample,
                        'peak_time': peak_sample / self.sample_rate,
                        'start_time': start_sample / self.sample_rate,
                        'time_difference': time_diff,  # Key field for verification
                        'match_confidence': match_confidence,  # 0.0-1.0
                        'quality_score': quality_score,
                        'rms': rms,
                        'peak_amplitude': peak,
                        'samples': len(segment),
                        'file_number': file_num
                    })
                    
                    saved += 1
                    if saved % 10 == 0:
                        print(f"Saved {saved}/{len(matches)}: {key}/{filename} (Δt={time_diff*1000:.1f}ms)")
                
                # Update progress
                progress = ((saved) / len(matches)) * 100
                self.parent.after(0, lambda p=progress: self.progress_var.set(p))
                self.parent.after(0, lambda s=saved, t=len(matches): 
                                self.progress_label.config(text=f"Processing: {s}/{t} (Saved: {s})"))
            
            # Save metadata
            with open(metadata_file, 'w', newline='') as f:
                fieldnames = ['key', 'filename', 'timestamp', 'csv_time',
                            'peak_sample', 'start_sample', 'peak_time', 'start_time',
                            'time_difference', 'match_confidence',
                            'quality_score', 'rms', 'peak_amplitude', 'samples', 'file_number']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata)
            
            # Compute matching statistics
            matched_count = sum(1 for m in matches if m['matched'])
            unmatched_segments = sum(1 for m in matches if not m['matched'])
            perfect_matches = sum(1 for m in matches if m['matched'] and m['time_diff'] < 0.05)
            good_matches = sum(1 for m in matches if m['matched'] and 0.05 <= m['time_diff'] < 0.1)
            
            # Show results
            results = f"""Segmentation Complete!
            
Detected peaks: {len(peak_positions)}
Valid segments: {len(valid_segments)}
CSV keystrokes: {len(self.keystroke_log)}
Successfully saved: {saved}

MATCHING QUALITY:
  Matched: {matched_count} / {len(self.keystroke_log)}
  Perfect (< 50ms): {perfect_matches}
  Good (< 100ms): {good_matches}
  Unmatched segments: {unmatched_segments}
  Missing keystrokes: {len(self.keystroke_log) - matched_count}

Output directory: {output_base}

Detection threshold: {threshold:.4f}
Segment length: {keystroke_length:.3f}s
Pre-trigger: {pre_trigger:.3f}s
Post-trigger: {post_trigger:.3f}s
Sample rate: {self.sample_rate}Hz

Filtering: {'Enabled' if self.enable_filtering.get() else 'Disabled'}
Peak centering: {'Enabled' if self.enable_peak_centering.get() else 'Disabled'}
"""
            
            self.parent.after(0, lambda: self.results_text.insert(1.0, results))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: messagebox.showinfo("Success", 
                                                             f"Segmentation complete!\nSaved {saved} valid segments"))
            
        except Exception as e:
            error_msg = f"Segmentation error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.parent.after(0, lambda: self.progress_label.config(text=error_msg))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: messagebox.showerror("Error", error_msg))