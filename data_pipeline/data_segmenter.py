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
        """Run the segmentation process in background - ROBUST ENERGY-BASED EXTRACTION."""
        try:
            # Create output directory
            output_base = os.path.join(os.path.dirname(self.current_session_path),
                                       self.output_name_var.get())
            os.makedirs(output_base, exist_ok=True)
            
            # Get parameters
            segment_samples = int(self.segment_duration.get() * self.sample_rate)
            pre_samples = int(self.pre_trigger.get() * self.sample_rate)
            post_samples = int(self.post_trigger.get() * self.sample_rate)
            
            # Metadata
            metadata = []
            metadata_file = os.path.join(output_base, 'metadata.csv')
            
            total = len(self.keystroke_log)
            processed = 0
            saved = 0
            skipped = 0
            
            for idx, keystroke in enumerate(self.keystroke_log):
                key = keystroke['key']
                rel_time = keystroke['relative_time']
                
                # Calculate sample position
                center_sample = int(rel_time * self.sample_rate)
                
                # Extract segment with robust energy detection
                if self.enable_peak_centering.get():
                    segment = self.extract_keystroke_robust(center_sample, segment_samples)
                else:
                    # Simple extraction without peak detection
                    start = max(0, center_sample - pre_samples)
                    end = start + segment_samples
                    if end > len(self.audio_data):
                        end = len(self.audio_data)
                        start = max(0, end - segment_samples)
                    segment = self.audio_data[start:end].copy()
                
                # Skip if segment is empty or too short
                if segment is None or len(segment) < segment_samples // 2:
                    print(f"Skipping {key} at {rel_time:.3f}s - extraction failed")
                    skipped += 1
                    processed += 1
                    continue
                
                # Ensure correct length
                if len(segment) < segment_samples:
                    # Pad with silence
                    padding = segment_samples - len(segment)
                    if len(segment.shape) == 2:
                        segment = np.pad(segment, ((0, padding), (0, 0)), mode='constant')
                    else:
                        segment = np.pad(segment, (0, padding), mode='constant')
                elif len(segment) > segment_samples:
                    # Trim
                    segment = segment[:segment_samples]
                
                # Apply filtering BEFORE saving (important for quality)
                if self.enable_filtering.get():
                    segment = self.audio.apply_bandpass_filter(segment,
                                                               self.filter_low.get(),
                                                               self.filter_high.get())
                
                # Quality check - skip if RMS is too low (likely noise/silence)
                rms = np.sqrt(np.mean(segment**2))
                if rms < 0.001:  # Threshold for noise
                    print(f"Skipping {key} at {rel_time:.3f}s - too quiet (RMS: {rms:.6f})")
                    skipped += 1
                    processed += 1
                    continue
                
                # Save segment
                key_folder = os.path.join(output_base, key)
                os.makedirs(key_folder, exist_ok=True)
                
                existing_files = [f for f in os.listdir(key_folder) if f.endswith('.wav')]
                file_num = len(existing_files)
                filename = f"{file_num}.wav"  # Simple numbering like main_old.py
                filepath = os.path.join(key_folder, filename)
                
                if self.audio.save_audio(filepath, segment):
                    # Calculate stats
                    peak = np.max(np.abs(segment))
                    
                    metadata.append({
                        'key': key,
                        'filename': filename,
                        'timestamp': keystroke['timestamp'],
                        'relative_time': rel_time,
                        'rms': rms,
                        'peak': peak,
                        'samples': len(segment),
                        'file_number': file_num
                    })
                    
                    saved += 1
                    print(f"Saved {key}/{filename} - RMS: {rms:.4f}, Peak: {peak:.4f}")
                
                processed += 1
                
                # Update progress
                progress = (processed / total) * 100
                self.parent.after(0, lambda p=progress: self.progress_var.set(p))
                self.parent.after(0, lambda p=processed, t=total, s=saved: 
                                self.progress_label.config(text=f"Processing: {p}/{t} (Saved: {s}, Skipped: {p-s})"))
            
            # Save metadata
            with open(metadata_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['key', 'filename', 'timestamp',
                                                        'relative_time', 'rms', 'peak',
                                                        'samples', 'file_number'])
                writer.writeheader()
                writer.writerows(metadata)
            
            # Show results
            results = f"""Segmentation Complete!
            
Total keystrokes: {total}
Successfully saved: {saved}
Skipped (low quality): {skipped}
Output directory: {output_base}

Segment duration: {self.segment_duration.get():.3f}s
Sample rate: {self.sample_rate}Hz
Samples per segment: {segment_samples}

Peak centering: {'Enabled' if self.enable_peak_centering.get() else 'Disabled'}
Filtering: {'Enabled' if self.enable_filtering.get() else 'Disabled'}
Filter range: {self.filter_low.get()}-{self.filter_high.get()} Hz

Success rate: {saved/total*100:.1f}%
"""
            
            self.parent.after(0, lambda: self.results_text.insert(1.0, results))
            self.parent.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: messagebox.showinfo("Success", 
                                                             f"Segmentation complete!\nSaved {saved} segments\nSkipped {skipped} low-quality"))
            
        except Exception as e:
            selfkeystroke_robust(self, center_sample: int, target_samples: int) -> Optional[np.ndarray]:
        """
        Extract keystroke using robust energy-based peak detection.
        Similar to the reference implementation but adapted for CSV timestamps.
        """
        try:
            # Define search window around timestamp (±200ms)
            search_window_ms = 0.2
            search_radius = int(search_window_ms * self.sample_rate)
            
            search_start = max(0, center_sample - search_radius)
            search_end = min(len(self.audio_data), center_sample + search_radius)
            
            if search_end - search_start < 100:  # Too short
                return None
            
            search_audio = self.audio_data[search_start:search_end]
            
            # Convert to mono for analysis
            if len(search_audio.shape) == 2:
                mono = np.mean(search_audio, axis=1)
            else:
                mono = search_audio
            
            # Calculate energy envelope using RMS in sliding window
            window_size = int(0.005 * self.sample_rate)  # 5ms windows
            if window_size < 1:
                window_size = 1
            
            hop_size = max(1, window_size // 2)
            energy = []
            
            for i in range(0, len(mono) - window_size, hop_size):
                window = mono[i:i+window_size]
                rms = np.sqrt(np.mean(window**2))
                energy.append(rms)
            
            if len(energy) == 0:
                return None
            
            energy = np.array(energy)
            
            # Find peak energy
            peak_energy_idx = np.argmax(energy)
            peak_sample_in_search = peak_energy_idx * hop_size + window_size // 2
            
            # Convert to absolute position in audio
            peak_absolute = search_start + peak_sample_in_search
            
            # Extract segment centered on peak
            pre_trigger_samples = int(self.pre_trigger.get() * self.sample_rate)
            
            start = max(0, peak_absolute - pre_trigger_samples)
            end = start + target_samples
            
            # Handle boundary cases
            if end > len(self.audio_data):
                end = len(self.audio_data)
                start = max(0, end - target_samples)
            
            segment = self.audio_data[start:end].copy()
            
            # Validate segment
            if len(segment) < target_samples // 4:  # Less than 25% of target
                return None
            
            return segment
            
        except Exception as e:
            print(f"Peak detection error: {e}")
            return None
    
    def extract_peak_centered(self, center_sample: int, target_samples: int,
                             pre_samples: int, post_samples: int) -> np.ndarray:
        """Legacy method - kept for compatibility."""
        return self.extract_keystroke_robust(center_sample, target_samples)ples)
        end = min(len(self.audio_data), peak_pos + post_samples)
        
        return self.audio_data[start:end]
