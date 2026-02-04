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
        self.audio_data = None
        self.keystroke_log = []
        self.sample_rate = None
        
        # Segmentation parameters
        self.segment_duration = tk.DoubleVar(value=0.430)
        self.pre_trigger = tk.DoubleVar(value=0.1)
        self.post_trigger = tk.DoubleVar(value=0.33)
        self.enable_peak_centering = tk.BooleanVar(value=True)
        self.enable_filtering = tk.BooleanVar(value=True)
        self.filter_low = tk.IntVar(value=50)
        self.filter_high = tk.IntVar(value=5000)
        
        self.build_ui()
    
    def build_ui(self):
        """Build the data segmenter UI."""
        main_frame = tk.Frame(self.parent, padx=20, pady=20)
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
        """Browse for a continuous recording session."""
        folder = filedialog.askdirectory(title="Select Continuous Recording Session Folder")
        if not folder:
            return
        
        # Check for required files
        audio_file = os.path.join(folder, 'audio.wav')
        log_file = os.path.join(folder, 'keystroke_log.csv')
        
        if not os.path.exists(audio_file):
            messagebox.showerror("Error", "audio.wav not found in selected folder")
            return
        
        if not os.path.exists(log_file):
            messagebox.showerror("Error", "keystroke_log.csv not found in selected folder")
            return
        
        # Load session
        self.load_session(folder)
    
    def load_session(self, folder: str):
        """Load continuous recording session."""
        try:
            self.current_session_path = folder
            
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
        if self.audio_data is None or not self.keystroke_log:
            messagebox.showwarning("Warning", "No session loaded")
            return
        
        # Disable button
        self.process_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        
        # Run in thread
        thread = threading.Thread(target=self.run_segmentation, daemon=True)
        thread.start()
    
    def cancel_segmentation(self):
        """Cancel segmentation (placeholder)."""
        pass
    
    def compute_onset_strength(self, audio: np.ndarray, 
                                hop_length: int = 512, 
                                win_length: int = 2048) -> np.ndarray:
        """
        Compute onset strength envelope using sliding window energy.
        Similar to librosa.onset.onset_strength but optimized for percussive keystrokes.
        
        Research basis: Keystroke papers use 10-100ms windows for energy calculation.
        """
        # Use STFT for frequency-aware energy computation
        f, t, Zxx = signal.stft(audio, fs=self.sample_rate, 
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
    
    def run_segmentation(self):
        """
        Robust keystroke segmentation based on audio research best practices:
        1. Sliding window energy detection (like librosa onset detection)
        2. Peak-centered extraction with proper attack capture
        3. Adaptive thresholding with validation
        4. Quality checks to reject invalid segments
        """
        try:
            # Create output directory
            output_base = os.path.join(os.path.dirname(self.current_session_path),
                                       self.output_name_var.get())
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
            onset_env = self.compute_onset_strength(mono_audio)
            
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
            
            # Match to CSV labels (by temporal proximity)
            metadata = []
            metadata_file = os.path.join(output_base, 'metadata.csv')
            saved = 0
            
            sorted_log = sorted(self.keystroke_log, key=lambda x: x['relative_time'])
            
            for idx, seg_info in enumerate(valid_segments):
                segment_audio = seg_info['audio']
                peak_sample = seg_info['peak_sample']
                start_sample = seg_info['start_sample']
                quality_score = seg_info['quality']
                
                # Match to CSV by index (or could use temporal matching)
                if idx < len(sorted_log):
                    key = sorted_log[idx]['key']
                    timestamp = sorted_log[idx]['timestamp']
                    csv_time = sorted_log[idx]['relative_time']
                else:
                    key = 'unknown'
                    timestamp = ''
                    csv_time = peak_sample / self.sample_rate
                
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
                        'quality_score': quality_score,
                        'rms': rms,
                        'peak_amplitude': peak,
                        'samples': len(segment),
                        'file_number': file_num
                    })
                    
                    saved += 1
                    if saved % 10 == 0:
                        print(f"Saved {saved}/{len(valid_segments)}: {key}/{filename}")
                
                # Update progress
                progress = ((idx + 1) / len(valid_segments)) * 100
                self.parent.after(0, lambda p=progress: self.progress_var.set(p))
                self.parent.after(0, lambda i=idx+1, t=len(valid_segments), s=saved: 
                                self.progress_label.config(text=f"Processing: {i}/{t} (Saved: {s})"))
            
            # Save metadata
            with open(metadata_file, 'w', newline='') as f:
                fieldnames = ['key', 'filename', 'timestamp', 'csv_time',
                            'peak_sample', 'start_sample', 'peak_time', 'start_time',
                            'quality_score', 'rms', 'peak_amplitude', 'samples', 'file_number']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(metadata)
            
            # Show results
            results = f"""Segmentation Complete!
            
Detected peaks: {len(peak_positions)}
Valid segments: {len(valid_segments)}
CSV keystrokes: {len(self.keystroke_log)}
Successfully saved: {saved}
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