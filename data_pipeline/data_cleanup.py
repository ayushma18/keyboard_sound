import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import sounddevice as sd
import threading
import csv
from datetime import datetime
import shutil
import json
from scipy import signal

class DataCleanupApp:
    def __init__(self, parent):
        self.root = parent  # Parent frame, not root window
        self.parent = parent
        
        # Data storage
        self.base_dir = "recordings"
        self.sessions = []
        self.current_session = None
        self.audio_files = []
        self.current_index = 0
        self.current_audio = None
        self.current_metadata = {}
        
        # Playback control
        self.is_playing = False
        self.play_thread = None
        self.playback_stream = None  # Added to track the stream
        
        # Threshold and filtering - IMPROVED
        self.threshold_value = 0.0001
        self.auto_threshold = 0.0001
        self.concentration_threshold = 0.3  # Energy concentration threshold
        
        self.files_to_delete = set()
        self.stats = {
            'total': 0,
            'above_threshold': 0,
            'below_threshold': 0,
            'marked_for_deletion': 0,
            'noise_detected': 0
        }
        
        # Analysis cache
        self.analysis_cache = {}
        
        # Settings
        self.show_spectrogram = tk.BooleanVar(value=False)
        self.auto_play = tk.BooleanVar(value=False)
        self.show_only_below_threshold = tk.BooleanVar(value=False)
        self.use_concentration_detection = tk.BooleanVar(value=True)
        self.playback_speed = tk.DoubleVar(value=1.0)
        
        # Audio properties
        self.sample_rate = 44100  # Default sample rate
        
        # Build UI
        self.build_ui()
        
        # Load sessions
        self.load_sessions()

    def calculate_energy_concentration(self, audio, sample_rate):
        """
        Calculate energy concentration metric to distinguish keystroke from noise.
        Keystrokes have concentrated energy in a short burst, noise is distributed.
        Returns a score between 0 and 1, where higher = more concentrated.
        """
        # Calculate short-term energy using sliding window
        window_size = int(0.005 * sample_rate)  # 5ms windows (smaller for better resolution)
        hop_size = window_size // 2
        
        # Calculate RMS energy for each window
        energy = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            rms = np.sqrt(np.mean(window**2))
            energy.append(rms)
        
        energy = np.array(energy)
        
        if len(energy) == 0:
            return 0.0
        
        total_energy = np.sum(energy)
        if total_energy < 1e-10:
            return 0.0
        
        # Method 1: Peak concentration (what % of energy is in the peak)
        max_energy = np.max(energy)
        mean_energy = np.mean(energy)
        
        # Find the peak and measure its width
        peak_idx = np.argmax(energy)
        
        # Calculate energy in top 10% of windows
        sorted_energy = np.sort(energy)[::-1]
        top_10_percent = int(len(energy) * 0.1) if len(energy) > 10 else 1
        top_energy_sum = np.sum(sorted_energy[:top_10_percent])
        top_concentration = top_energy_sum / total_energy
        
        # Method 2: Temporal sharpness (how quickly energy rises and falls)
        # Good keystrokes have sharp peaks
        if len(energy) > 5:
            # Calculate gradient (rate of change)
            gradient = np.abs(np.gradient(energy))
            avg_gradient = np.mean(gradient)
            max_gradient = np.max(gradient)
            sharpness = max_gradient / (avg_gradient + 1e-10)
            sharpness_normalized = min(sharpness / 20.0, 1.0)  # Normalize
        else:
            sharpness_normalized = 0.0
        
        # Method 3: Peak-to-mean ratio
        peak_to_mean = max_energy / (mean_energy + 1e-10)
        peak_ratio_normalized = min(peak_to_mean / 10.0, 1.0)  # Normalize
        
        # Combined concentration score (weighted average)
        concentration_score = (
            top_concentration * 0.5 +      # 50% weight on top energy concentration
            sharpness_normalized * 0.3 +   # 30% weight on temporal sharpness
            peak_ratio_normalized * 0.2    # 20% weight on peak ratio
        )
        
        return min(concentration_score, 1.0)

    def calculate_spectral_flatness(self, audio):
        """
        Calculate spectral flatness - noise has flatter spectrum, keystrokes have peaks.
        Returns value between 0 and 1, where higher = more noise-like.
        """
        # Compute FFT
        fft = np.fft.rfft(audio)
        magnitude = np.abs(fft)
        
        # Avoid log of zero
        magnitude = magnitude + 1e-10
        
        # Spectral flatness: geometric mean / arithmetic mean
        geometric_mean = np.exp(np.mean(np.log(magnitude)))
        arithmetic_mean = np.mean(magnitude)
        
        flatness = geometric_mean / arithmetic_mean
        return flatness

    def calculate_zero_crossing_rate(self, audio):
        """Calculate zero crossing rate - noise tends to have higher ZCR"""
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio)))) / (2 * len(audio))
        return zero_crossings

    def is_likely_noise(self, audio, sample_rate, rms):
        """
        Determine if audio is likely noise based on multiple features.
        Returns: (is_noise, reason, concentration_score, details_dict)
        """
        if not self.use_concentration_detection.get():
            # Fall back to simple RMS threshold
            is_noise = rms < self.threshold_value
            return is_noise, f"RMS: {rms:.6f}", 0.0, {}
        
        # Calculate features
        concentration = self.calculate_energy_concentration(audio, sample_rate)
        zcr = self.calculate_zero_crossing_rate(audio)
        spectral_flat = self.calculate_spectral_flatness(audio)
        
        # Store details for display
        details = {
            'rms': rms,
            'concentration': concentration,
            'zcr': zcr,
            'spectral_flatness': spectral_flat
        }
        
        # Decision logic - primary focus on concentration
        reasons = []
        
        # Primary criterion: Energy concentration
        if concentration < self.concentration_threshold:
            reasons.append(f"Low concentration: {concentration:.3f}")
        
        # Secondary criteria (only matter if concentration is borderline)
        if rms < self.threshold_value:
            reasons.append(f"Low RMS: {rms:.6f}")
        
        if zcr > 0.4:  # Very high ZCR suggests noise
            reasons.append(f"High ZCR: {zcr:.3f}")
        
        if spectral_flat > 0.5:  # Flat spectrum suggests noise
            reasons.append(f"Flat spectrum: {spectral_flat:.3f}")
        
        # Primary decision based on concentration
        is_noise = concentration < self.concentration_threshold
        
        # Override: Even with low concentration, if RMS is very high, it might be valid
        if is_noise and rms > (self.threshold_value * 10):
            is_noise = False
            reasons = [f"High RMS overrides: {rms:.6f}"]
        
        # Override: Very low RMS is always noise regardless of concentration
        if rms < (self.threshold_value * 0.1):
            is_noise = True
            if f"Low RMS: {rms:.6f}" not in reasons:
                reasons.insert(0, f"Very low RMS: {rms:.6f}")
        
        reason_str = " | ".join(reasons) if reasons else f"Valid (Conc: {concentration:.3f}, RMS: {rms:.6f})"
        
        return is_noise, reason_str, concentration, details

    def build_ui(self):
        # Create main container with left and right panels
        main_container = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED)
        main_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # LEFT PANEL - Controls (with scrollbar for small screens)
        left_frame = tk.Frame(main_container, width=400)
        
        # Add canvas and scrollbar
        canvas = tk.Canvas(left_frame, width=380)
        scrollbar = tk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        left_panel = tk.Frame(canvas)
        
        left_panel.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=left_panel, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_container.add(left_frame, minsize=350)
        
        # RIGHT PANEL - Visualization
        right_panel = tk.Frame(main_container)
        main_container.add(right_panel, minsize=600)
        
        self.build_left_panel(left_panel)
        self.build_right_panel(right_panel)

    def build_left_panel(self, parent):
        # Title
        title = tk.Label(parent, text="Data Cleanup Tool", font=("Arial", 16, "bold"), fg="#1976D2")
        title.pack(pady=10)
        
        # Session Selection
        session_frame = tk.LabelFrame(parent, text="1. Select Session", font=("Arial", 10, "bold"), padx=10, pady=10)
        session_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.session_listbox = tk.Listbox(session_frame, height=6, font=("Arial", 9))
        self.session_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.session_listbox.bind('<<ListboxSelect>>', self.on_session_select)
        
        session_btn_frame = tk.Frame(session_frame)
        session_btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(session_btn_frame, text="Refresh Sessions", command=self.load_sessions, 
                  bg="#4CAF50", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        tk.Button(session_btn_frame, text="Browse Folder", command=self.browse_folder, 
                  bg="#2196F3", fg="white", font=("Arial", 9)).pack(side=tk.LEFT, padx=2)
        
        # Session Info
        self.session_info_label = tk.Label(session_frame, text="No session selected", font=("Arial", 9), fg="gray")
        self.session_info_label.pack(pady=5)
        
        # Detection Method
        method_frame = tk.LabelFrame(parent, text="Detection Method", font=("Arial", 10, "bold"), padx=10, pady=10)
        method_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Checkbutton(method_frame, text="Use Energy Concentration Detection (Recommended)", 
                       variable=self.use_concentration_detection, command=self.update_detection_method,
                       font=("Arial", 9, "bold"), fg="#1976D2").pack(anchor=tk.W, pady=5)
        
        tk.Label(method_frame, text="Detects keystrokes by energy concentration.\nKeystrokes = concentrated burst\nNoise = distributed energy",
                 font=("Arial", 8), fg="gray", justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        
        # Threshold Control
        threshold_frame = tk.LabelFrame(parent, text="2. Adjust Detection Thresholds", font=("Arial", 10, "bold"), padx=10, pady=10)
        threshold_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # === PRIMARY: Concentration Threshold (Main Control) ===
        tk.Label(threshold_frame, text="PRIMARY: Energy Concentration Threshold", 
                 font=("Arial", 9, "bold"), fg="#9C27B0").pack(anchor=tk.W, pady=5)
        
        tk.Label(threshold_frame, text="Higher = stricter (only very concentrated sounds pass)\nLower = more lenient (distributed sounds may pass)",
                 font=("Arial", 8), fg="gray", justify=tk.LEFT).pack(anchor=tk.W, pady=2)
        
        conc_control = tk.Frame(threshold_frame)
        conc_control.pack(fill=tk.X, pady=5)
        
        self.conc_scale = tk.Scale(conc_control, from_=0.05, to=0.9, resolution=0.01, 
                                    orient=tk.HORIZONTAL, command=self.on_concentration_change, length=250)
        self.conc_scale.set(self.concentration_threshold)
        self.conc_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.conc_value_label = tk.Label(conc_control, text=f"{self.concentration_threshold:.2f}", 
                                          font=("Arial", 12, "bold"), fg="#9C27B0", width=8)
        self.conc_value_label.pack(side=tk.LEFT, padx=5)
        
        # Concentration presets
        conc_preset_frame = tk.Frame(threshold_frame)
        conc_preset_frame.pack(fill=tk.X, pady=5)
        tk.Label(conc_preset_frame, text="Presets:", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)
        
        conc_presets = [
            ("Very Strict", 0.5),
            ("Strict", 0.4),
            ("Balanced", 0.3),
            ("Lenient", 0.2),
            ("Very Lenient", 0.15),
        ]
        
        for name, value in conc_presets:
            tk.Button(conc_preset_frame, text=name, command=lambda v=value: self.set_concentration_threshold(v),
                      font=("Arial", 8), width=10).pack(side=tk.LEFT, padx=2)
        
        # Separator
        tk.Frame(threshold_frame, height=2, bg="gray").pack(fill=tk.X, pady=10)
        
        # === SECONDARY: RMS Threshold (Fallback/Override) ===
        tk.Label(threshold_frame, text="SECONDARY: RMS Threshold (Override for very quiet sounds)", 
                 font=("Arial", 9, "bold"), fg="#FF5722").pack(anchor=tk.W, pady=5)
        
        tk.Label(threshold_frame, text="Sounds quieter than this are always classified as noise",
                 font=("Arial", 8), fg="gray").pack(anchor=tk.W, pady=2)
        
        # Auto-calculate threshold button
        auto_frame = tk.Frame(threshold_frame)
        auto_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(auto_frame, text="Auto-Calculate RMS", command=self.auto_calculate_threshold, 
                  bg="#9C27B0", fg="white", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)
        
        self.auto_threshold_label = tk.Label(auto_frame, text="", font=("Arial", 8), fg="blue")
        self.auto_threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Manual threshold slider - FIXED RANGE
        threshold_control = tk.Frame(threshold_frame)
        threshold_control.pack(fill=tk.X, pady=5)
        
        self.threshold_scale = tk.Scale(threshold_control, from_=0.00001, to=0.1, resolution=0.00001, 
                                         orient=tk.HORIZONTAL, command=self.on_threshold_change, length=250)
        self.threshold_scale.set(self.threshold_value)
        self.threshold_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.threshold_value_label = tk.Label(threshold_control, text=f"{self.threshold_value:.5f}", 
                                               font=("Arial", 10, "bold"), fg="#FF5722", width=10)
        self.threshold_value_label.pack(side=tk.LEFT, padx=5)
        
        # RMS Preset buttons
        preset_frame = tk.Frame(threshold_frame)
        preset_frame.pack(fill=tk.X, pady=5)
        tk.Label(preset_frame, text="Presets:", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)
        
        presets = [
            ("Very Low", 0.00005),
            ("Low", 0.0001),
            ("Medium", 0.001),
            ("High", 0.01),
        ]
        
        for name, value in presets:
            tk.Button(preset_frame, text=name, command=lambda v=value: self.set_threshold(v),
                      font=("Arial", 8), width=8).pack(side=tk.LEFT, padx=2)
        
        # Statistics
        stats_frame = tk.LabelFrame(parent, text="Statistics", font=("Arial", 10, "bold"), padx=10, pady=10)
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=7, font=("Courier", 9), bg="#f5f5f5", relief=tk.FLAT)
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        self.update_stats_display()
        
        # Navigation
        nav_frame = tk.LabelFrame(parent, text="3. Navigate & Review", font=("Arial", 10, "bold"), padx=10, pady=10)
        nav_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Filter option
        tk.Checkbutton(nav_frame, text="Show only files classified as noise", 
                       variable=self.show_only_below_threshold, command=self.apply_filter,
                       font=("Arial", 9)).pack(anchor=tk.W, pady=5)
        
        # Quick navigation buttons for noise
        noise_nav_frame = tk.Frame(nav_frame)
        noise_nav_frame.pack(pady=5, fill=tk.X)
        
        tk.Button(noise_nav_frame, text="◀ Previous Noise", command=self.find_previous_noise_direct, 
                  bg="#FF9800", fg="white", font=("Arial", 9, "bold"), width=15).pack(side=tk.LEFT, padx=2, expand=True)
        
        tk.Button(noise_nav_frame, text="Next Noise ▶", command=self.find_next_noise_direct, 
                  bg="#FF9800", fg="white", font=("Arial", 9, "bold"), width=15).pack(side=tk.LEFT, padx=2, expand=True)
        
        # Current file info
        self.current_file_label = tk.Label(nav_frame, text="No file loaded", font=("Arial", 9, "bold"), fg="#1976D2")
        self.current_file_label.pack(pady=5)
        
        self.current_stats_label = tk.Label(nav_frame, text="", font=("Arial", 8), fg="black")
        self.current_stats_label.pack(pady=2)
        
        # Navigation buttons
        nav_buttons = tk.Frame(nav_frame)
        nav_buttons.pack(pady=10)
        
        tk.Button(nav_buttons, text="⏮ First", command=self.go_first, font=("Arial", 10), width=8).grid(row=0, column=0, padx=2, pady=2)
        tk.Button(nav_buttons, text="◀ Previous", command=self.go_previous, font=("Arial", 10), width=8).grid(row=0, column=1, padx=2, pady=2)
        tk.Button(nav_buttons, text="Next ▶", command=self.go_next, font=("Arial", 10), width=8).grid(row=0, column=2, padx=2, pady=2)
        tk.Button(nav_buttons, text="Last ⏭", command=self.go_last, font=("Arial", 10), width=8).grid(row=0, column=3, padx=2, pady=2)
        
        # Jump to index
        jump_frame = tk.Frame(nav_frame)
        jump_frame.pack(pady=5)
        
        tk.Label(jump_frame, text="Jump to:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        self.jump_entry = tk.Entry(jump_frame, width=8, font=("Arial", 9))
        self.jump_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(jump_frame, text="Go", command=self.jump_to_index, font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        playback_frame = tk.Frame(nav_frame)
        playback_frame.pack(pady=10)
        
        # Main play button
        self.play_button = tk.Button(playback_frame, text="▶ Play Audio", command=self.toggle_playback, 
                                      bg="#4CAF50", fg="white", font=("Arial", 11, "bold"), width=14, height=2)
        self.play_button.pack(pady=5)
        
        # Quick play button (plays at current speed without stopping)
        tk.Button(playback_frame, text="🔊 Play Current Audio", command=self.play_audio_quick, 
                  bg="#2196F3", fg="white", font=("Arial", 10, "bold"), width=14).pack(pady=2)
        
        # Playback speed control
        speed_frame = tk.Frame(playback_frame)
        speed_frame.pack(pady=5)
        tk.Label(speed_frame, text="Speed:", font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        self.playback_speed = tk.DoubleVar(value=1.0)
        tk.Button(speed_frame, text="0.5x", command=lambda: self.set_playback_speed(0.5), 
                  font=("Arial", 8), width=5).pack(side=tk.LEFT, padx=1)
        tk.Button(speed_frame, text="0.75x", command=lambda: self.set_playback_speed(0.75), 
                  font=("Arial", 8), width=5).pack(side=tk.LEFT, padx=1)
        tk.Button(speed_frame, text="1x", command=lambda: self.set_playback_speed(1.0), 
                  font=("Arial", 8, "bold"), width=5, bg="#E3F2FD").pack(side=tk.LEFT, padx=1)
        tk.Button(speed_frame, text="1.5x", command=lambda: self.set_playback_speed(1.5), 
                  font=("Arial", 8), width=5).pack(side=tk.LEFT, padx=1)
        tk.Button(speed_frame, text="2x", command=lambda: self.set_playback_speed(2.0), 
                  font=("Arial", 8), width=5).pack(side=tk.LEFT, padx=1)
        
        self.speed_label = tk.Label(playback_frame, text="Speed: 1.0x", font=("Arial", 9, "bold"), fg="#1976D2")
        self.speed_label.pack(pady=2)
        
        tk.Checkbutton(playback_frame, text="Auto-play on navigation", variable=self.auto_play, 
                       font=("Arial", 9)).pack(pady=5)
        
        # Mark for deletion
        self.delete_button = tk.Button(nav_frame, text="❌ Mark for Deletion", command=self.toggle_mark_for_deletion, 
                                        bg="#FF5722", fg="white", font=("Arial", 10, "bold"))
        self.delete_button.pack(pady=10, fill=tk.X)
        
        # Cleanup Actions
        action_frame = tk.LabelFrame(parent, text="4. Cleanup Actions", font=("Arial", 10, "bold"), padx=10, pady=10)
        action_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Button(action_frame, text="Mark All Detected Noise", command=self.mark_all_below_threshold, 
                  bg="#FF9800", fg="white", font=("Arial", 9, "bold")).pack(fill=tk.X, pady=3)
        
        tk.Button(action_frame, text="Unmark All", command=self.unmark_all, 
                  bg="#607D8B", fg="white", font=("Arial", 9, "bold")).pack(fill=tk.X, pady=3)
        
        tk.Button(action_frame, text="🗑 DELETE Marked Files", command=self.delete_marked_files, 
                  bg="#D32F2F", fg="white", font=("Arial", 10, "bold")).pack(fill=tk.X, pady=10)
        
        # Export
        tk.Button(action_frame, text="📊 Export Analysis Report", command=self.export_report, 
                  bg="#1976D2", fg="white", font=("Arial", 9, "bold")).pack(fill=tk.X, pady=3)

    def build_right_panel(self, parent):
        # Visualization controls
        viz_control = tk.Frame(parent)
        viz_control.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(viz_control, text="Visualization:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Checkbutton(viz_control, text="Show Spectrogram", variable=self.show_spectrogram, 
                       command=self.update_visualization, font=("Arial", 9)).pack(side=tk.LEFT, padx=10)
        
        # Matplotlib figure
        self.fig = Figure(figsize=(10, 8), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Initialize with empty plot
        self.ax_wave = self.fig.add_subplot(211)
        self.ax_wave.set_title("Waveform", fontsize=12, fontweight='bold')
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.3)
        
        self.ax_spec = self.fig.add_subplot(212)
        self.ax_spec.set_title("Energy Over Time", fontsize=12, fontweight='bold')
        self.ax_spec.set_xlabel("Time (s)")
        self.ax_spec.set_ylabel("Energy (RMS)")
        self.ax_spec.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def on_concentration_change(self, value):
        """Handle concentration threshold change"""
        self.concentration_threshold = float(value)
        self.conc_value_label.config(text=f"{self.concentration_threshold:.2f}")
        
        # Reanalyze with new threshold
        if self.analysis_cache:
            for file_path, cached in self.analysis_cache.items():
                audio = cached['audio']
                sample_rate = cached['sample_rate']
                rms = cached['rms']
                
                is_noise, reason, concentration, details = self.is_likely_noise(audio, sample_rate, rms)
                
                cached['is_noise'] = is_noise
                cached['reason'] = reason
                cached['concentration'] = concentration
            
            # Update stats and visualization
            self.update_stats_display()
            if self.current_audio is not None:
                self.load_current_file()

    def set_concentration_threshold(self, value):
        """Set concentration threshold to a preset value"""
        self.conc_scale.set(value)
        self.on_concentration_change(value)

    def update_detection_method(self):
        """Update when detection method changes"""
        # Reanalyze if files are loaded
        if self.audio_files:
            self.analyze_all_files()
            self.load_current_file()

    def load_sessions(self):
        """Load all available recording sessions"""
        self.session_listbox.delete(0, tk.END)
        self.sessions = []
        
        if not os.path.exists(self.base_dir):
            messagebox.showwarning("Warning", f"Recordings directory '{self.base_dir}' not found!")
            return
        
        # Find all session folders (format: micid-keyboardid)
        for item in os.listdir(self.base_dir):
            item_path = os.path.join(self.base_dir, item)
            if os.path.isdir(item_path):
                # Check if it has audio files
                has_audio = False
                for root, dirs, files in os.walk(item_path):
                    if any(f.endswith('.wav') for f in files):
                        has_audio = True
                        break
                
                if has_audio:
                    self.sessions.append(item_path)
                    # Count files
                    file_count = sum(1 for r, d, f in os.walk(item_path) for file in f if file.endswith('.wav'))
                    self.session_listbox.insert(tk.END, f"{item} ({file_count} files)")
        
        if not self.sessions:
            self.session_info_label.config(text="No sessions found", fg="orange")
        else:
            self.session_info_label.config(text=f"{len(self.sessions)} session(s) found", fg="green")

    def browse_folder(self):
        """Browse for custom recordings folder"""
        folder = filedialog.askdirectory(title="Select Recordings Folder")
        if folder:
            self.base_dir = folder
            self.load_sessions()

    def on_session_select(self, event):
        """Handle session selection"""
        selection = self.session_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        self.current_session = self.sessions[index]
        
        # Load all audio files from this session
        self.load_audio_files()

    def load_audio_files(self):
        """Load all audio files from current session"""
        if not self.current_session:
            return
        
        self.audio_files = []
        self.analysis_cache = {}
        
        # Walk through session directory and find all WAV files
        for root, dirs, files in os.walk(self.current_session):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    self.audio_files.append(file_path)
        
        # Sort files
        self.audio_files.sort()
        
        # Load metadata if available
        self.load_metadata()
        
        # Reset state
        self.current_index = 0
        self.files_to_delete = set()
        
        # Update UI
        session_name = os.path.basename(self.current_session)
        self.session_info_label.config(
            text=f"Session: {session_name} | {len(self.audio_files)} files loaded",
            fg="green"
        )
        
        # Analyze all files
        self.analyze_all_files()
        
        # Load first file
        if self.audio_files:
            self.load_current_file()

    def load_metadata(self):
        """Load metadata.csv if available"""
        metadata_path = os.path.join(self.current_session, "metadata.csv")
        self.current_metadata = {}
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', newline='') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # Use wav_file as key
                        wav_file = row.get('wav_file', '')
                        if wav_file:
                            full_path = os.path.join(self.current_session, wav_file)
                            self.current_metadata[full_path] = row
            except Exception as e:
                print(f"Error loading metadata: {e}")

    def analyze_all_files(self):
        """Analyze all audio files and cache results"""
        if not self.audio_files:
            return
        
        print(f"Analyzing {len(self.audio_files)} files...")
        
        for i, file_path in enumerate(self.audio_files):
            if i % 100 == 0:
                print(f"Progress: {i}/{len(self.audio_files)}")
            
            try:
                # Read audio file
                with wave.open(file_path, 'r') as wf:
                    sample_rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Calculate metrics
                rms = np.sqrt(np.mean(audio**2))
                peak = np.max(np.abs(audio))
                
                # Calculate advanced features
                is_noise, reason, concentration, details = self.is_likely_noise(audio, sample_rate, rms)
                
                # Cache analysis
                self.analysis_cache[file_path] = {
                    'rms': rms,
                    'peak': peak,
                    'audio': audio,
                    'sample_rate': sample_rate,
                    'is_noise': is_noise,
                    'reason': reason,
                    'concentration': concentration,
                    'details': details
                }
                
            except Exception as e:
                print(f"Error analyzing {file_path}: {e}")
        
        print("Analysis complete!")
        self.update_stats_display()

    def load_current_file(self):
        """Load and display current audio file"""
        if not self.audio_files or self.current_index >= len(self.audio_files):
            return
        
        file_path = self.audio_files[self.current_index]
        
        try:
            # Always read the WAV file to get sample rate first
            with wave.open(file_path, 'r') as wf:
                self.sample_rate = wf.getframerate()
            
            # Get from cache if available
            if file_path in self.analysis_cache:
                cached = self.analysis_cache[file_path]
                self.current_audio = cached['audio']
                rms = cached['rms']
                peak = cached['peak']
                is_noise = cached['is_noise']
                reason = cached['reason']
                concentration = cached['concentration']
                details = cached.get('details', {})
            else:
                # Read audio file completely
                with wave.open(file_path, 'r') as wf:
                    self.sample_rate = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    self.current_audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                
                # Calculate metrics
                rms = np.sqrt(np.mean(self.current_audio**2))
                peak = np.max(np.abs(self.current_audio))
                
                is_noise, reason, concentration, details = self.is_likely_noise(self.current_audio, self.sample_rate, rms)
                
                # Cache it
                self.analysis_cache[file_path] = {
                    'rms': rms,
                    'peak': peak,
                    'audio': self.current_audio,
                    'sample_rate': self.sample_rate,
                    'is_noise': is_noise,
                    'reason': reason,
                    'concentration': concentration,
                    'details': details
                }
            
            # Update UI
            relative_path = os.path.relpath(file_path, self.current_session)
            self.current_file_label.config(text=f"File {self.current_index + 1}/{len(self.audio_files)}: {relative_path}")
            
            # Check metadata
            metadata_str = ""
            if file_path in self.current_metadata:
                meta = self.current_metadata[file_path]
                metadata_str = f"Key: {meta.get('key', 'N/A')} | Quality: {meta.get('quality', 'N/A')}"
            
            # Status
            if is_noise:
                status = "🔴 NOISE DETECTED"
                status_color = "#FF5722"
            else:
                status = "✓ VALID KEYSTROKE"
                status_color = "#4CAF50"
            
            marked = "❌ MARKED FOR DELETION" if file_path in self.files_to_delete else ""
            
            # Get additional details
            zcr = details.get('zcr', 0)
            spectral_flat = details.get('spectral_flatness', 0)
            
            stats_text = (f"Status: {status}\n"
                         f"═══════════════════\n"
                         f"Concentration: {concentration:.3f} (Threshold: {self.concentration_threshold:.2f})\n"
                         f"RMS: {rms:.6f} (Threshold: {self.threshold_value:.5f})\n"
                         f"Peak: {peak:.4f}\n"
                         f"ZCR: {zcr:.3f}\n"
                         f"Spectral Flatness: {spectral_flat:.3f}\n"
                         f"─────────────────\n"
                         f"Reason: {reason}\n"
                         f"{metadata_str}\n"
                         f"{marked}")
            
            self.current_stats_label.config(text=stats_text, fg=status_color if not marked else "#D32F2F")
            
            # Update delete button
            if file_path in self.files_to_delete:
                self.delete_button.config(text="✓ Unmark for Deletion", bg="#4CAF50")
            else:
                self.delete_button.config(text="❌ Mark for Deletion", bg="#FF5722")
            
            # Update visualization
            self.update_visualization()
            
            # Auto-play if enabled
            if self.auto_play.get() and not self.is_playing:
                self.play_audio()
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio file: {e}")

    def update_visualization(self):
        """Update waveform and energy visualization"""
        if self.current_audio is None:
            return
        
        # Clear previous plots
        self.ax_wave.clear()
        self.ax_spec.clear()
        
        # Time array
        duration = len(self.current_audio) / self.sample_rate
        time = np.linspace(0, duration, len(self.current_audio))
        
        # Get current file info
        file_path = self.audio_files[self.current_index]
        cached = self.analysis_cache.get(file_path, {})
        is_noise = cached.get('is_noise', False)
        concentration = cached.get('concentration', 0)
        rms = cached.get('rms', 0)
        
        # Plot waveform
        self.ax_wave.plot(time, self.current_audio, linewidth=0.5, color='#1976D2')
        self.ax_wave.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # Highlight if noise
        if is_noise:
            self.ax_wave.set_facecolor('#ffebee')
            title_color = '#D32F2F'
            title_text = f"Waveform - 🔴 NOISE DETECTED (Conc: {concentration:.3f})"
        else:
            self.ax_wave.set_facecolor('#e8f5e9')
            title_color = '#4CAF50'
            title_text = f"Waveform - ✓ VALID KEYSTROKE (Conc: {concentration:.3f})"
        
        self.ax_wave.set_title(title_text, fontsize=11, fontweight='bold', color=title_color)
        self.ax_wave.set_xlabel("Time (s)")
        self.ax_wave.set_ylabel("Amplitude")
        self.ax_wave.grid(True, alpha=0.3)
        
        # Plot energy envelope
        window_size = int(0.01 * self.sample_rate)  # 10ms windows
        hop_size = window_size // 2
        
        energy = []
        energy_time = []
        
        for i in range(0, len(self.current_audio) - window_size, hop_size):
            window = self.current_audio[i:i+window_size]
            rms_window = np.sqrt(np.mean(window**2))
            energy.append(rms_window)
            energy_time.append(i / self.sample_rate)
        
        if energy:
            self.ax_spec.plot(energy_time, energy, linewidth=2, color='#FF5722', label='Energy')
            self.ax_spec.axhline(y=self.threshold_value, color='red', linestyle='--', 
                                 label=f'RMS Threshold: {self.threshold_value:.5f}', alpha=0.7)
            
            # Mark peak energy region
            if len(energy) > 0:
                peak_idx = np.argmax(energy)
                self.ax_spec.scatter(energy_time[peak_idx], energy[peak_idx], 
                                     c='green', s=100, zorder=5, label='Peak Energy')
            
            self.ax_spec.set_title(f"Energy Concentration Over Time (RMS: {rms:.6f})", 
                                   fontsize=11, fontweight='bold')
            self.ax_spec.set_xlabel("Time (s)")
            self.ax_spec.set_ylabel("Energy (RMS)")
            self.ax_spec.grid(True, alpha=0.3)
            self.ax_spec.legend(loc='upper right', fontsize=8)
        
        self.fig.tight_layout()
        self.canvas.draw()

    def on_threshold_change(self, value):
        """Handle threshold slider change"""
        self.threshold_value = float(value)
        self.threshold_value_label.config(text=f"{self.threshold_value:.5f}")
        
        # Reanalyze with new threshold
        if self.analysis_cache:
            for file_path, cached in self.analysis_cache.items():
                audio = cached['audio']
                sample_rate = cached['sample_rate']
                rms = cached['rms']
                
                is_noise, reason, concentration, details = self.is_likely_noise(audio, sample_rate, rms)
                
                cached['is_noise'] = is_noise
                cached['reason'] = reason
                cached['concentration'] = concentration
                cached['details'] = details
            
            # Update stats
            self.update_stats_display()
            
            # Update visualization if file is loaded
            if self.current_audio is not None:
                self.load_current_file()

    def set_threshold(self, value):
        """Set threshold to a preset value"""
        self.threshold_scale.set(value)
        self.on_threshold_change(value)

    def auto_calculate_threshold(self):
        """Automatically calculate optimal threshold based on data distribution"""
        if not self.analysis_cache:
            messagebox.showwarning("Warning", "No files analyzed yet!")
            return
        
        # Get all RMS values
        rms_values = [data['rms'] for data in self.analysis_cache.values()]
        rms_array = np.array(rms_values)
        
        # Calculate statistics
        mean_rms = np.mean(rms_array)
        median_rms = np.median(rms_array)
        std_rms = np.std(rms_array)
        
        # Suggested threshold: median - 0.5 * std (captures lower quartile)
        suggested = max(median_rms - 0.5 * std_rms, np.min(rms_array) * 1.5)
        
        self.auto_threshold = suggested
        self.auto_threshold_label.config(text=f"Suggested: {suggested:.5f}")
        
        # Show statistics
        msg = (f"RMS Statistics:\n\n"
               f"Mean: {mean_rms:.6f}\n"
               f"Median: {median_rms:.6f}\n"
               f"Std Dev: {std_rms:.6f}\n"
               f"Min: {np.min(rms_array):.6f}\n"
               f"Max: {np.max(rms_array):.6f}\n\n"
               f"Suggested threshold: {suggested:.5f}\n\n"
               f"Use this threshold?")
        
        if messagebox.askyesno("Auto-Calculate Threshold", msg):
            self.set_threshold(suggested)

    def update_stats_display(self):
        """Update statistics display"""
        total = len(self.audio_files)
        
        if total == 0:
            self.stats_text.delete(1.0, tk.END)
            self.stats_text.insert(1.0, "No files loaded")
            return
        
        # Count files by classification
        valid = 0
        noise = 0
        
        for file_path, data in self.analysis_cache.items():
            if data.get('is_noise', False):
                noise += 1
            else:
                valid += 1
        
        marked = len(self.files_to_delete)
        
        # Calculate percentages
        valid_pct = (valid / total * 100) if total > 0 else 0
        noise_pct = (noise / total * 100) if total > 0 else 0
        marked_pct = (marked / total * 100) if total > 0 else 0
        
        stats_text = f"""Total Files:       {total}
Valid Keystrokes:  {valid} ({valid_pct:.1f}%)
Noise Detected:    {noise} ({noise_pct:.1f}%)
Marked for Del:    {marked} ({marked_pct:.1f}%)

RMS Threshold:     {self.threshold_value:.5f}
Conc. Threshold:   {self.concentration_threshold:.2f}
"""
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        
        # Update stats dict
        self.stats = {
            'total': total,
            'above_threshold': valid,
            'below_threshold': noise,
            'marked_for_deletion': marked,
            'noise_detected': noise
        }

    def apply_filter(self):
        """Apply filter to show only files below threshold"""
        # This would require maintaining a filtered list
        # For now, just notify
        if self.show_only_below_threshold.get():
            messagebox.showinfo("Info", "Use navigation buttons to browse. Noise files will be highlighted.")

    # Navigation methods
    def go_first(self):
        """Go to first file"""
        self.current_index = 0
        self.load_current_file()

    def go_last(self):
        """Go to last file"""
        self.current_index = len(self.audio_files) - 1
        self.load_current_file()

    def go_previous(self):
        """Go to previous file"""
        if self.show_only_below_threshold.get():
            # Find previous file below threshold
            self.find_previous_below_threshold()
        else:
            # Normal navigation - show all files
            if self.current_index > 0:
                self.current_index -= 1
                self.load_current_file()

    def go_next(self):
        """Go to next file"""
        if self.show_only_below_threshold.get():
            # Find next file below threshold
            self.find_next_below_threshold()
        else:
            # Normal navigation - show all files
            if self.current_index < len(self.audio_files) - 1:
                self.current_index += 1
                self.load_current_file()

    def find_next_below_threshold(self):
        """Find next file classified as noise"""
        start = self.current_index + 1
        for i in range(start, len(self.audio_files)):
            file_path = self.audio_files[i]
            if file_path in self.analysis_cache:
                if self.analysis_cache[file_path].get('is_noise', False):
                    self.current_index = i
                    self.load_current_file()
                    return
        
        messagebox.showinfo("Info", "No more noise files")

    def find_previous_below_threshold(self):
        """Find previous file classified as noise"""
        start = self.current_index - 1
        for i in range(start, -1, -1):
            file_path = self.audio_files[i]
            if file_path in self.analysis_cache:
                if self.analysis_cache[file_path].get('is_noise', False):
                    self.current_index = i
                    self.load_current_file()
                    return
        
        messagebox.showinfo("Info", "No more noise files")

    def find_previous_noise_direct(self):
        """Find and jump to previous noise file"""
        self.find_previous_below_threshold()

    def find_next_noise_direct(self):
        """Find and jump to next noise file"""
        self.find_next_below_threshold()

    def set_playback_speed(self, speed):
        """Set playback speed"""
        self.playback_speed.set(speed)
        self.speed_label.config(text=f"Speed: {speed}x")
        
        # If currently playing, restart with new speed
        if self.is_playing:
            self.stop_playback()
            self.root.after(100, self.play_audio)

    def jump_to_index(self):
        """Jump to specific index"""
        try:
            index = int(self.jump_entry.get()) - 1  # User enters 1-based
            if 0 <= index < len(self.audio_files):
                self.current_index = index
                self.load_current_file()
            else:
                messagebox.showerror("Error", f"Index must be between 1 and {len(self.audio_files)}")
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number")

    def toggle_playback(self):
        """Toggle audio playback"""
        if self.is_playing:
            self.stop_playback()
        else:
            self.play_audio()

    def play_audio(self):
        """Play current audio file"""
        if self.current_audio is None:
            return
        
        # Stop any existing playback first
        self.stop_playback()
        
        self.is_playing = True
        self.play_button.config(text="⏸ Stop", bg="#FF5722")
        
        # Start playback in separate thread
        self.play_thread = threading.Thread(target=self._play_audio_thread, daemon=True)
        self.play_thread.start()

    def play_audio_quick(self):
        """Quick play - just play the audio at current speed"""
        if self.current_audio is None:
            messagebox.showwarning("No Audio", "No audio file loaded!")
            return
        
        # Stop current playback if any
        if self.is_playing:
            self.stop_playback()
            self.root.after(100, self.play_audio)
        else:
            self.play_audio()

    def _play_audio_thread(self):
        """Audio playback thread with speed control - FIXED"""
        try:
            # Get playback speed
            speed = self.playback_speed.get()
            
            # Adjust sample rate for speed (higher rate = faster playback)
            adjusted_rate = int(self.sample_rate * speed)
            
            # Play the audio and wait for it to finish
            sd.play(self.current_audio, adjusted_rate)
            sd.wait()  # This blocks until playback is complete
            
        except Exception as e:
            print(f"Playback error: {e}")
        finally:
            # Schedule UI update on main thread
            if self.is_playing:  # Only update if we haven't been stopped externally
                self.root.after(0, self._playback_finished)

    def _playback_finished(self):
        """Called when playback naturally finishes"""
        self.is_playing = False
        self.play_button.config(text="▶ Play Audio", bg="#4CAF50")

    def stop_playback(self):
        """Stop audio playback"""
        if self.is_playing:
            self.is_playing = False
            sd.stop()
            self.play_button.config(text="▶ Play Audio", bg="#4CAF50")

    def toggle_mark_for_deletion(self):
        """Mark/unmark current file for deletion"""
        if not self.audio_files:
            return
        
        file_path = self.audio_files[self.current_index]
        
        if file_path in self.files_to_delete:
            self.files_to_delete.remove(file_path)
        else:
            self.files_to_delete.add(file_path)
        
        # Update display
        self.update_stats_display()
        self.load_current_file()

    def mark_all_below_threshold(self):
        """Mark all files classified as noise for deletion"""
        if not self.audio_files:
            return
        
        count = 0
        for file_path, data in self.analysis_cache.items():
            if data.get('is_noise', False):
                self.files_to_delete.add(file_path)
                count += 1
        
        self.update_stats_display()
        self.load_current_file()
        messagebox.showinfo("Marked", f"Marked {count} noise files for deletion")

    def unmark_all(self):
        """Unmark all files"""
        self.files_to_delete.clear()
        self.update_stats_display()
        self.load_current_file()
        messagebox.showinfo("Unmarked", "All files unmarked")

    def delete_marked_files(self):
        """Delete all marked files"""
        if not self.files_to_delete:
            messagebox.showwarning("Warning", "No files marked for deletion")
            return
        
        count = len(self.files_to_delete)
        msg = (f"You are about to DELETE {count} files permanently!\n\n"
               f"This action CANNOT be undone.\n\n"
               f"Do you want to continue?")
        
        if not messagebox.askyesno("Confirm Deletion", msg, icon='warning'):
            return
        
        # Create backup folder
        backup_folder = os.path.join(self.current_session, "_deleted_backup_" + 
                                      datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(backup_folder, exist_ok=True)
        
        deleted_count = 0
        failed = []
        
        for file_path in self.files_to_delete:
            try:
                # Move to backup instead of deleting
                relative_path = os.path.relpath(file_path, self.current_session)
                backup_path = os.path.join(backup_folder, relative_path)
                os.makedirs(os.path.dirname(backup_path), exist_ok=True)
                shutil.move(file_path, backup_path)
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
                failed.append(file_path)
        
        # Update metadata CSV
        self.update_metadata_after_deletion()
        
        # Reload files
        self.load_audio_files()
        
        result_msg = f"Moved {deleted_count} files to backup folder:\n{backup_folder}\n\n"
        if failed:
            result_msg += f"Failed to delete {len(failed)} files."
        
        messagebox.showinfo("Deletion Complete", result_msg)

    def update_metadata_after_deletion(self):
        """Update metadata.csv after deletion"""
        metadata_path = os.path.join(self.current_session, "metadata.csv")
        
        if not os.path.exists(metadata_path):
            return
        
        try:
            # Read existing metadata
            rows = []
            with open(metadata_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                for row in reader:
                    wav_file = row.get('wav_file', '')
                    full_path = os.path.join(self.current_session, wav_file)
                    # Only keep if file still exists
                    if os.path.exists(full_path):
                        rows.append(row)
            
            # Write back
            with open(metadata_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
        except Exception as e:
            print(f"Error updating metadata: {e}")

    def export_report(self):
        """Export analysis report"""
        if not self.audio_files:
            messagebox.showwarning("Warning", "No data to export")
            return
        
        # Ask for save location
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile=f"cleanup_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        
        if not filename:
            return
        
        try:
            with open(filename, 'w', newline='') as f:
                fieldnames = ['file_path', 'rms', 'peak', 'concentration', 'classification', 'reason', 'marked_for_deletion']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for file_path in self.audio_files:
                    if file_path in self.analysis_cache:
                        data = self.analysis_cache[file_path]
                        writer.writerow({
                            'file_path': os.path.relpath(file_path, self.current_session),
                            'rms': f"{data['rms']:.6f}",
                            'peak': f"{data['peak']:.4f}",
                            'concentration': f"{data.get('concentration', 0):.3f}",
                            'classification': 'Noise' if data.get('is_noise', False) else 'Valid',
                            'reason': data.get('reason', ''),
                            'marked_for_deletion': 'Yes' if file_path in self.files_to_delete else 'No'
                        })
            
            messagebox.showinfo("Export Complete", f"Report exported to:\n{filename}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export report: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = DataCleanupApp(root)
    root.mainloop()