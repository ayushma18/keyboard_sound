"""
Data segmentation module - extracts individual keystroke segments from continuous recordings.
"""
import os
import csv
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from .audio_handler import AudioHandler
from .config import Config
import threading


class DataSegmenterTab:
    """Tab for segmenting continuous recordings into individual keystroke samples."""
    
    def __init__(self, parent, config: Config, audio_handler: AudioHandler):
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
    
    def run_segmentation(self):
        """Run HYBRID segmentation: CSV timestamps + energy validation."""
        try:
            # Create output directory
            output_base = os.path.join(os.path.dirname(self.current_session_path),
                                       self.output_name_var.get())
            os.makedirs(output_base, exist_ok=True)
            
            # Get parameters
            keystroke_length = self.segment_duration.get()
            begin_offset = self.pre_trigger.get()
            
            keystroke_samples = int(keystroke_length * self.sample_rate)
            offset_samples = int(begin_offset * self.sample_rate)
            
            print(f"\n=== Starting Hybrid Segmentation (CSV + Energy) ===")
            print(f"Audio length: {len(self.audio_data)} samples ({len(self.audio_data)/self.sample_rate:.2f}s)")
            print(f"Expected keystrokes: {len(self.keystroke_log)}")
            print(f"Keystroke length: {keystroke_length}s, Begin offset: {begin_offset}s")
            
            # Convert to mono for analysis
            if len(self.audio_data.shape) == 2:
                mono_audio = np.mean(self.audio_data, axis=1)
            else:
                mono_audio = self.audio_data
            
            # Calculate fine-grained energy envelope for peak detection
            print("Calculating energy envelope for peak detection...")
            window_size = int(0.005 * self.sample_rate)  # 5ms windows for fine detail
            hop_size = int(0.001 * self.sample_rate)  # 1ms hop for precision
            if hop_size < 1:
                hop_size = 1
            
            energy = []
            for i in range(0, len(mono_audio) - window_size, hop_size):
                window = mono_audio[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                energy.append(rms)
            
            energy = np.array(energy)
            energy_threshold = np.mean(energy) + np.std(energy)  # Mean + 1 std dev
            
            print(f"Energy: Max={np.max(energy):.4f}, Mean={np.mean(energy):.4f}, Threshold={energy_threshold:.4f}")
            
            # Process each keystroke from CSV
            metadata = []
            metadata_file = os.path.join(output_base, 'metadata.csv')
            saved = 0
            skipped = 0
            
            sorted_log = sorted(self.keystroke_log, key=lambda x: x['relative_time'])
            
            for idx, keystroke_info in enumerate(sorted_log):
                key = keystroke_info['key']
                rel_time = keystroke_info['relative_time']
                timestamp = keystroke_info['timestamp']
                
                # Get approximate position from CSV
                approx_sample = int(rel_time * self.sample_rate)
                
                # Find actual peak within ±100ms window
                search_window_ms = 0.1
                search_radius_samples = int(search_window_ms * self.sample_rate)
                search_start = max(0, approx_sample - search_radius_samples)
                search_end = min(len(mono_audio), approx_sample + search_radius_samples)
                
                # Convert to energy indices
                search_start_energy = search_start // hop_size
                search_end_energy = min(len(energy), search_end // hop_size)
                
                if search_end_energy > search_start_energy:
                    # Find peak in energy within search window
                    search_energy = energy[search_start_energy:search_end_energy]
                    if len(search_energy) > 0 and np.max(search_energy) > energy_threshold * 0.3:
                        peak_idx_in_window = np.argmax(search_energy)
                        peak_energy_idx = search_start_energy + peak_idx_in_window
                        peak_sample = peak_energy_idx * hop_size
                    else:
                        # No strong peak found, use CSV timestamp
                        peak_sample = approx_sample
                else:
                    peak_sample = approx_sample
                
                # Extract segment
                begin = max(0, peak_sample - offset_samples)
                end = min(len(mono_audio), peak_sample + keystroke_samples - offset_samples)
                
                segment = mono_audio[begin:end]
                
                # Always pad to target length (no skipping for short segments)
                if len(segment) < keystroke_samples:
                    # Pad with silence to reach target length
                    padding_needed = keystroke_samples - len(segment)
                    segment = np.pad(segment, (0, padding_needed), mode='constant')
                    print(f"Padded {key} at {rel_time:.3f}s with {padding_needed} samples")
                elif len(segment) > keystroke_samples:
                    # Trim to exact length
                    segment = segment[:keystroke_samples]
                
                # Apply filtering if enabled
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
                    # Calculate stats
                    rms = np.sqrt(np.mean(segment**2))
                    peak = np.max(np.abs(segment))
                    
                    metadata.append({
                        'key': key,
                        'filename': filename,
                        'timestamp': timestamp,
                        'relative_time': rel_time,
                        'rms': rms,
                        'peak': peak,
                        'samples': len(segment),
                        'file_number': file_num,
                        'detected_position': peak_sample / self.sample_rate
                    })
                    
                    saved += 1
                    print(f"Saved {key}/{filename} - CSV: {rel_time:.2f}s, Detected: {peak_sample/self.sample_rate:.2f}s, RMS: {rms:.4f}")
                
                # Update progress
                progress = ((idx + 1) / len(sorted_log)) * 100
                self.parent.after(0, lambda p=progress: self.progress_var.set(p))
                self.parent.after(0, lambda i=idx+1, t=len(sorted_log), s=saved: 
                                self.progress_label.config(text=f"Processing: {i}/{t} (Saved: {s})"))
            
            # Save metadata
            with open(metadata_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['key', 'filename', 'timestamp',
                                                        'relative_time', 'rms', 'peak',
                                                        'samples', 'file_number', 'detected_position'])
                writer.writeheader()
                writer.writerows(metadata)
            
            # Show results
            results = f"""Segmentation Complete!
            
Total keystrokes: {len(sorted_log)}
Successfully saved: {saved}
Skipped: {skipped}
Output directory: {output_base}

Segment duration: {keystroke_length:.3f}s
Begin offset: {begin_offset:.3f}s
Sample rate: {self.sample_rate}Hz

Filtering: {'Enabled' if self.enable_filtering.get() else 'Disabled'}
Filter range: {self.filter_low.get()}-{self.filter_high.get()} Hz

Success rate: {saved/len(sorted_log)*100:.1f}%
"""
            
            self.parent.after(0, lambda: self.results_text.insert(1.0, results))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: messagebox.showinfo("Success", 
                                                             f"Segmentation complete!\nSaved {saved} segments"))
            
        except Exception as e:
            error_msg = f"Segmentation error: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            self.parent.after(0, lambda: self.progress_label.config(text=error_msg))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: messagebox.showerror("Error", error_msg))
    
    def separate_keystrokes(self, energy, threshold, recording, keystroke_length, begin_offset, hop_size):
        """
        Separate keystrokes based on energy threshold - EXACTLY like reference implementation.
        When energy exceeds threshold, extract FIXED-LENGTH keystroke and skip past it.
        Does NOT wait for energy to drop - just finds starts and extracts fixed duration.
        """
        keystrokes = []
        i = 0
        
        rate = self.sample_rate
        keystroke_samples = int(keystroke_length * rate)
        offset_samples = int(begin_offset * rate)
        
        while i < len(energy):
            if energy[i] > threshold:
                # Found keystroke start - extract fixed-length segment
                # Convert energy index to sample position
                begin_sample = i * hop_size
                begin = max(0, begin_sample - offset_samples)
                end = min(len(recording), begin_sample + keystroke_samples)
                
                if end - begin > 100:  # Minimum validation
                    keystroke_audio = recording[begin:end]
                    keystrokes.append((keystroke_audio, begin_sample))
                
                # Skip to END of this keystroke to avoid re-detecting it
                # Skip in ENERGY space (not sample space)
                i += int(keystroke_samples / hop_size)
                continue
            
            i += 1
        
        return keystrokes
    
    def find_threshold(self, energy, init_threshold, recording, step, target_keystrokes,
                      keystroke_length, begin_offset, hop_size):
        """
        Find optimal threshold that produces target number of keystrokes (like reference).
        """
        cur_threshold = init_threshold
        keystrokes = self.separate_keystrokes(energy, init_threshold, recording, 
                                              keystroke_length, begin_offset, hop_size)
        
        iteration = 0
        max_iterations = 1000
        
        while len(keystrokes) != target_keystrokes and iteration < max_iterations:
            if len(keystrokes) > target_keystrokes:
                # Too many, raise threshold
                cur_threshold += step
            else:
                # Too few, lower threshold
                cur_threshold -= step
            
            step = step * 0.99  # Decay step size
            keystrokes = self.separate_keystrokes(energy, cur_threshold, recording,
                                                  keystroke_length, begin_offset, hop_size)
            
            if iteration % 10 == 0:
                print(f'Iteration {iteration}: Count={len(keystrokes)}, Threshold={cur_threshold:.4f}, Step={step:.6f}')
            
            iteration += 1
        
        print(f'Final: Keystroke count={len(keystrokes)}, Threshold={cur_threshold:.4f}')
        
        return (keystrokes, cur_threshold)
