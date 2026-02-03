import tkinter as tk
from tkinter import messagebox, ttk, simpledialog
import threading
import time
import sounddevice as sd
import numpy as np
import wave
import os
import csv
from pynput import keyboard
from datetime import datetime
import queue
import json
from scipy import signal


class ResearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Keyboard Acoustic Side-Channel Research Tool")
        self.is_recording = False
        self.is_testing_mic = False
        
        # --- Recording duration UI ---
        self.recording_duration = 60  # default seconds
        self.remaining_time = 0
        self.timer_running = False
        self.duration_var = tk.StringVar(value="60")
        
        # Audio parameters - MATCHED TO ORIGINAL DATASET
        self.fs = 44100  # Sample rate
        self.buffer_duration = 2.0  # seconds
        self.segment_duration = 0.430  # Updated: 18963 samples / 44100 Hz = 0.43 seconds
        self.target_segment_samples = 18963  # EXACT length from original dataset
        self.buffer_samples = int(self.fs * self.buffer_duration)
        self.segment_samples = self.target_segment_samples
        
        # Initialize buffer properly - STEREO
        self.audio_buffer = np.zeros((self.buffer_samples, 2), dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # NOISE PROFILE for spectral subtraction
        self.noise_profile = None
        self.noise_profile_lock = threading.Lock()
        self.calibrating_noise = False
        
        # KEY DEBOUNCING: Track pressed keys and timing
        self.pressed_keys = {}  # key -> timestamp of last recording
        self.key_debounce_time = 0.15  # minimum time between same key recordings (seconds)
        self.keys_lock = threading.Lock()
        
        # Audio devices
        self.input_device = None
        self.output_device = None
        self.input_device_list = []
        self.output_device_list = []
        
        # Noise threshold for filtering
        self.noise_threshold = 0.001  # Minimum RMS level to consider as valid signal
        
        # Noise reduction parameters
        self.noise_reduction_strength = 1.5  # How aggressively to reduce noise
        self.enable_noise_reduction = True
        
        # Microphone and Keyboard IDs
        self.mic_id = "mic1"
        self.keyboard_id = "kb1"
        self.session_folder = None
        
        # Output directories - will be set based on mic/keyboard IDs
        self.base_output_dir = "recordings"
        self.output_dir = None
        self.metadata_file = None
        self.metadata_fields = ["timestamp", "key", "wav_file", "rms_level", "peak_level", "quality", 
                               "mic_id", "keyboard_id", "session_id", "file_number", "channels", "samples"]
        
        # Configuration file for storing IDs
        self.config_file = "recording_config.json"
        self.load_config()
        
        # Threads
        self.audio_thread = None
        self.keyboard_thread = None
        self.listener = None
        self.test_thread = None
        self.audio_queue = queue.Queue()
        
        # Build UI
        self.build_ui()
        
        # Initialize output directory
        self.update_output_directory()
        
        # Load available devices
        self.load_audio_devices()

    def load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.mic_id = config.get('mic_id', 'mic1')
                    self.keyboard_id = config.get('keyboard_id', 'kb1')
        except Exception as e:
            print(f"Error loading config: {e}")

    def save_config(self):
        """Save configuration to file"""
        try:
            config = {
                'mic_id': self.mic_id,
                'keyboard_id': self.keyboard_id
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            print(f"Error saving config: {e}")

    def update_output_directory(self):
        """Update output directory based on mic and keyboard IDs"""
        # Create folder structure: recordings/micid-keyboardid/
        self.session_folder = f"{self.mic_id}-{self.keyboard_id}"
        self.output_dir = os.path.join(self.base_output_dir, self.session_folder)
        
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            print(f"Output directory: {self.output_dir}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create output directory: {e}")
        
        # Update metadata file path
        self.metadata_file = os.path.join(self.output_dir, "metadata.csv")
        
        # Initialize metadata CSV if it doesn't exist
        if not os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.metadata_fields)
                    writer.writeheader()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to initialize metadata file: {e}")
        
        # Update UI label
        if hasattr(self, 'folder_label'):
            self.folder_label.config(text=f"Recording to: {self.session_folder}/")

    def build_ui(self):
        # Main container
        main_frame = tk.Frame(self.root, padx=10, pady=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = tk.Label(main_frame, text="Keyboard Acoustic Recorder", 
                              font=("Arial", 16, "bold"))
        title_label.pack(pady=(0, 10))
        
        # DATASET FORMAT INFO
        format_frame = tk.LabelFrame(main_frame, text="Dataset Format (Auto-configured)", 
                                    font=("Arial", 9, "bold"), padx=10, pady=5, bg="#E3F2FD")
        format_frame.pack(fill=tk.X, pady=(0, 10))
        
        format_text = f"✓ Stereo (2 channels) - mono mics auto-converted\n✓ 44.1 kHz sample rate\n✓ {self.target_segment_samples} samples per key (~{self.segment_duration:.3f}s)\n✓ Produces 64 time frames for model"
        format_label = tk.Label(format_frame, text=format_text, 
                            font=("Arial", 8), fg="#1565C0", justify=tk.LEFT, bg="#E3F2FD")
        format_label.pack(pady=2)
        
        # Microphone and Keyboard ID Section
        id_frame = tk.LabelFrame(main_frame, text="Session Configuration", 
                                font=("Arial", 10, "bold"), padx=10, pady=10)
        id_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Mic ID
        mic_row = tk.Frame(id_frame)
        mic_row.pack(fill=tk.X, pady=5)
        tk.Label(mic_row, text="Microphone ID:", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.mic_id_entry = tk.Entry(mic_row, width=20)
        self.mic_id_entry.insert(0, self.mic_id)
        self.mic_id_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(mic_row, text="(e.g., mic1, mic2, blue_yeti)", fg="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)
        
        # Keyboard ID
        kb_row = tk.Frame(id_frame)
        kb_row.pack(fill=tk.X, pady=5)
        tk.Label(kb_row, text="Keyboard ID:", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.keyboard_id_entry = tk.Entry(kb_row, width=20)
        self.keyboard_id_entry.insert(0, self.keyboard_id)
        self.keyboard_id_entry.pack(side=tk.LEFT, padx=5)
        tk.Label(kb_row, text="(e.g., kb1, mechanical, laptop)", fg="gray", font=("Arial", 8)).pack(side=tk.LEFT, padx=5)
        
        # Update button
        update_btn_row = tk.Frame(id_frame)
        update_btn_row.pack(fill=tk.X, pady=5)
        self.update_ids_btn = tk.Button(update_btn_row, text="Update Session IDs", 
                                       command=self.update_session_ids,
                                       bg="#FF9800", fg="white", font=("Arial", 9, "bold"))
        self.update_ids_btn.pack(side=tk.LEFT, padx=5)
        
        # Current folder display
        self.folder_label = tk.Label(id_frame, text=f"Recording to: {self.mic_id}-{self.keyboard_id}/", 
                                    font=("Arial", 9, "bold"), fg="#1976D2")
        self.folder_label.pack(pady=5)
        
        # Device Selection Section
        device_frame = tk.LabelFrame(main_frame, text="Audio Device Selection", 
                                     font=("Arial", 10, "bold"), padx=10, pady=10)
        device_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Input device
        tk.Label(device_frame, text="Input Device:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(device_frame, textvariable=self.input_device_var, 
                                         width=50, state="readonly")
        self.input_device_combo.grid(row=0, column=1, padx=5, pady=5)
        self.input_device_combo.bind("<<ComboboxSelected>>", self.on_input_device_selected)
        
        # Output device
        tk.Label(device_frame, text="Output Device:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(device_frame, textvariable=self.output_device_var, 
                                         width=50, state="readonly")
        self.output_device_combo.grid(row=1, column=1, padx=5, pady=5)
        self.output_device_combo.bind("<<ComboboxSelected>>", self.on_output_device_selected)
        
        refresh_btn = tk.Button(device_frame, text="Refresh Devices", 
                               command=self.load_audio_devices)
        refresh_btn.grid(row=0, column=2, rowspan=2, padx=5, pady=5)
        
        # Noise Calibration Section
        noise_frame = tk.LabelFrame(main_frame, text="Noise Reduction", 
                                   font=("Arial", 10, "bold"), padx=10, pady=10)
        noise_frame.pack(fill=tk.X, pady=(0, 10))
        
        calibrate_row = tk.Frame(noise_frame)
        calibrate_row.pack(fill=tk.X, pady=5)
        
        self.calibrate_button = tk.Button(calibrate_row, text="Calibrate Background Noise (2 sec)", 
                                         command=self.calibrate_noise,
                                         bg="#9C27B0", fg="white", font=("Arial", 9, "bold"))
        self.calibrate_button.pack(side=tk.LEFT, padx=5)
        
        self.noise_status_label = tk.Label(calibrate_row, text="Not calibrated", 
                                          fg="orange", font=("Arial", 9))
        self.noise_status_label.pack(side=tk.LEFT, padx=10)
        
        # Noise reduction controls
        nr_controls = tk.Frame(noise_frame)
        nr_controls.pack(fill=tk.X, pady=5)
        
        self.nr_var = tk.BooleanVar(value=True)
        nr_check = tk.Checkbutton(nr_controls, text="Enable Noise Reduction", 
                                 variable=self.nr_var, command=self.toggle_noise_reduction)
        nr_check.pack(side=tk.LEFT, padx=5)
        
        tk.Label(nr_controls, text="Strength:").pack(side=tk.LEFT, padx=5)
        self.nr_strength_var = tk.DoubleVar(value=1.5)
        nr_scale = tk.Scale(nr_controls, from_=0.5, to=3.0, resolution=0.1,
                           orient=tk.HORIZONTAL, variable=self.nr_strength_var,
                           length=150, command=self.update_nr_strength)
        nr_scale.pack(side=tk.LEFT, padx=5)
        self.nr_strength_label = tk.Label(nr_controls, text="1.5")
        self.nr_strength_label.pack(side=tk.LEFT, padx=5)
        
        # Microphone Test Section
        test_frame = tk.LabelFrame(main_frame, text="Microphone Test", 
                                  font=("Arial", 10, "bold"), padx=10, pady=10)
        test_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.test_button = tk.Button(test_frame, text="Start Mic Test (Live Playback)", 
                                    command=self.toggle_mic_test, bg="#4CAF50", fg="white")
        self.test_button.pack(side=tk.LEFT, padx=5)
        
        self.level_label = tk.Label(test_frame, text="Level: ", font=("Arial", 10))
        self.level_label.pack(side=tk.LEFT, padx=10)
        
        self.level_bar = ttk.Progressbar(test_frame, length=200, mode='determinate')
        self.level_bar.pack(side=tk.LEFT, padx=5)
        
        # Recording Duration Section
        duration_frame = tk.LabelFrame(main_frame, text="Recording Settings", 
                                      font=("Arial", 10, "bold"), padx=10, pady=10)
        duration_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.build_duration_selector(duration_frame)
        
        # Noise threshold control
        threshold_frame = tk.Frame(duration_frame)
        threshold_frame.pack(pady=5)
        tk.Label(threshold_frame, text="Noise Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=0.001)
        threshold_scale = tk.Scale(threshold_frame, from_=0.0001, to=0.01, resolution=0.0001,
                                  orient=tk.HORIZONTAL, variable=self.threshold_var,
                                  length=200, command=self.update_threshold)
        threshold_scale.pack(side=tk.LEFT, padx=5)
        self.threshold_label = tk.Label(threshold_frame, text="0.0010")
        self.threshold_label.pack(side=tk.LEFT, padx=5)
        
        # Timer Display
        self.timer_label = tk.Label(main_frame, text="", font=("Arial", 14, "bold"), fg="blue")
        self.timer_label.pack(pady=5)
        
        # Status Display
        self.status_label = tk.Label(main_frame, text="Status: Idle", 
                                    font=("Arial", 12), fg="green")
        self.status_label.pack(pady=10)
        
        # Recording stats
        self.stats_label = tk.Label(main_frame, text="Keys recorded: 0 | Rejected (noise): 0", 
                                   font=("Arial", 10), fg="blue")
        self.stats_label.pack(pady=5)
        
        # Recording Control Buttons
        button_frame = tk.Frame(main_frame)
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="Start Recording", 
                                     command=self.start_recording, 
                                     bg="#2196F3", fg="white", 
                                     font=("Arial", 11, "bold"),
                                     width=15, height=2)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(button_frame, text="Stop Recording", 
                                    command=self.stop_recording, 
                                    state=tk.DISABLED,
                                    bg="#f44336", fg="white",
                                    font=("Arial", 11, "bold"),
                                    width=15, height=2)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Info Label
        info_text = "Instructions:\n1. Set Microphone ID and Keyboard ID\n2. Select audio device (mono mics will be auto-converted to stereo)\n3. Calibrate background noise (stay quiet for 2 seconds)\n4. Test microphone\n5. Start recording and type 0-9, a-z!\n\nFormat: Stereo, 44.1kHz, 20948 samples (matches dataset requirement)\nFiles organized in folders (0-9, a-z) with sequential numbering"
        info_label = tk.Label(main_frame, text=info_text, 
                            font=("Arial", 9), fg="gray", justify=tk.LEFT)
        info_label.pack(pady=(10, 0))
        
        # Set minimum window size
        self.root.minsize(700, 800)
        
        # Initialize stats
        self.keys_recorded = 0
        self.keys_rejected = 0
        self.session_id = None

    def toggle_noise_reduction(self):
        """Toggle noise reduction on/off"""
        self.enable_noise_reduction = self.nr_var.get()
        status = "enabled" if self.enable_noise_reduction else "disabled"
        print(f"Noise reduction {status}")

    def update_nr_strength(self, value):
        """Update noise reduction strength"""
        self.noise_reduction_strength = float(value)
        self.nr_strength_label.config(text=f"{self.noise_reduction_strength:.1f}")

    def calibrate_noise(self):
        """Calibrate background noise profile"""
        if self.input_device is None:
            messagebox.showerror("Error", "Please select an audio input device first!")
            return
        
        if self.is_recording or self.is_testing_mic:
            messagebox.showwarning("Warning", "Please stop other operations before calibrating.")
            return
        
        self.calibrating_noise = True
        self.calibrate_button.config(state=tk.DISABLED, text="Calibrating...")
        self.noise_status_label.config(text="Stay quiet...", fg="orange")
        
        # Run calibration in thread
        threading.Thread(target=self.run_noise_calibration, daemon=True).start()

    def run_noise_calibration(self):
        """Record background noise and create noise profile - HANDLES MONO AND STEREO"""
        try:
            duration = 2.0  # seconds
            samples = int(self.fs * duration)
            
            # Detect device channels
            device_info = sd.query_devices(self.input_device, 'input')
            record_channels = min(2, device_info['max_input_channels'])
            
            # Record background noise
            noise_data = sd.rec(samples, samplerate=self.fs, channels=record_channels, 
                               dtype='float32', device=self.input_device)
            sd.wait()
            
            # Average channels if stereo, or use mono directly
            if noise_data.shape[1] > 1:
                noise_signal = np.mean(noise_data, axis=1)
            else:
                noise_signal = noise_data[:, 0]
            
            # Compute noise profile (power spectrum)
            f, t, Zxx = signal.stft(noise_signal, fs=self.fs, nperseg=512)
            noise_power = np.mean(np.abs(Zxx)**2, axis=1)
            
            with self.noise_profile_lock:
                self.noise_profile = noise_power
            
            # Update UI
            self.root.after(0, self.on_calibration_complete)
            
            print(f"Noise calibration complete ({record_channels} channel{'s' if record_channels > 1 else ''})")
            print(f"Noise profile shape: {noise_power.shape}")
            print(f"Mean noise power: {np.mean(noise_power):.6f}")
            
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Calibration Error", 
                          f"Error during noise calibration: {e}")
            self.root.after(0, self.on_calibration_failed)

    def on_calibration_complete(self):
        """Update UI after successful calibration"""
        self.calibrating_noise = False
        self.calibrate_button.config(state=tk.NORMAL, text="Recalibrate Background Noise")
        self.noise_status_label.config(text="✓ Calibrated", fg="green")

    def on_calibration_failed(self):
        """Update UI after failed calibration"""
        self.calibrating_noise = False
        self.calibrate_button.config(state=tk.NORMAL, text="Calibrate Background Noise (2 sec)")
        self.noise_status_label.config(text="Calibration failed", fg="red")

    def apply_noise_reduction(self, audio_segment):
        """Apply spectral subtraction noise reduction - works with mono or stereo"""
        if not self.enable_noise_reduction:
            return audio_segment
        
        with self.noise_profile_lock:
            if self.noise_profile is None:
                return self.apply_highpass_filter(audio_segment)
        
        try:
            # Perform STFT on the audio segment
            f, t, Zxx = signal.stft(audio_segment, fs=self.fs, nperseg=512)
            
            # Get magnitude and phase
            magnitude = np.abs(Zxx)
            phase = np.angle(Zxx)
            
            # Spectral subtraction
            with self.noise_profile_lock:
                noise_magnitude = np.sqrt(self.noise_profile)
            
            # Ensure noise profile matches
            if len(noise_magnitude) != magnitude.shape[0]:
                from scipy.interpolate import interp1d
                old_freqs = np.linspace(0, 1, len(noise_magnitude))
                new_freqs = np.linspace(0, 1, magnitude.shape[0])
                f_interp = interp1d(old_freqs, noise_magnitude, kind='linear', fill_value='extrapolate')
                noise_magnitude = f_interp(new_freqs)
            
            noise_magnitude = noise_magnitude.reshape(-1, 1)
            reduced_magnitude = np.maximum(
                magnitude - self.noise_reduction_strength * noise_magnitude, 
                magnitude * 0.1
            )
            
            # Reconstruct
            Zxx_reduced = reduced_magnitude * np.exp(1j * phase)
            _, audio_reduced = signal.istft(Zxx_reduced, fs=self.fs, nperseg=512)
            
            # Match length
            if len(audio_reduced) > len(audio_segment):
                audio_reduced = audio_reduced[:len(audio_segment)]
            elif len(audio_reduced) < len(audio_segment):
                audio_reduced = np.pad(audio_reduced, (0, len(audio_segment) - len(audio_reduced)))
            
            audio_reduced = self.apply_highpass_filter(audio_reduced)
            
            return audio_reduced.astype(np.float32)
            
        except Exception as e:
            print(f"Error in noise reduction: {e}")
            return self.apply_highpass_filter(audio_segment)

    def apply_highpass_filter(self, audio_segment):
        """Apply high-pass filter to remove low-frequency noise"""
        try:
            nyquist = self.fs / 2
            cutoff = 80  # Hz
            order = 4
            
            b, a = signal.butter(order, cutoff / nyquist, btype='high')
            filtered = signal.filtfilt(b, a, audio_segment)
            
            return filtered.astype(np.float32)
        except Exception as e:
            print(f"Error in high-pass filter: {e}")
            return audio_segment

    def update_session_ids(self):
        """Update microphone and keyboard IDs from UI"""
        new_mic_id = self.mic_id_entry.get().strip()
        new_keyboard_id = self.keyboard_id_entry.get().strip()
        
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', new_mic_id):
            messagebox.showerror("Invalid ID", "Microphone ID can only contain letters, numbers, underscore, and hyphen.")
            return
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', new_keyboard_id):
            messagebox.showerror("Invalid ID", "Keyboard ID can only contain letters, numbers, underscore, and hyphen.")
            return
        
        if new_mic_id != self.mic_id or new_keyboard_id != self.keyboard_id:
            self.mic_id = new_mic_id
            self.keyboard_id = new_keyboard_id
            self.save_config()
            self.update_output_directory()
            
            messagebox.showinfo("Session Updated", 
                              f"Session IDs updated!\n\nRecordings will be saved to:\n{self.session_folder}/")

    def update_threshold(self, value):
        """Update noise threshold value"""
        self.noise_threshold = float(value)
        self.threshold_label.config(text=f"{self.noise_threshold:.4f}")

    def build_duration_selector(self, parent):
        frame = tk.Frame(parent)
        tk.Label(frame, text="Recording Duration:").pack(side=tk.LEFT, padx=5)
        
        options = ["30", "60", "120", "300", "Custom"]
        self.duration_menu = tk.OptionMenu(frame, self.duration_var, *options, 
                                          command=self.on_duration_change)
        self.duration_menu.pack(side=tk.LEFT, padx=5)
        
        self.custom_entry = tk.Entry(frame, width=8)
        self.custom_entry.pack(side=tk.LEFT, padx=5)
        self.custom_entry.insert(0, "60")
        self.custom_entry.configure(state=tk.DISABLED)
        
        tk.Label(frame, text="seconds").pack(side=tk.LEFT)
        frame.pack(pady=5)

    def on_duration_change(self, value):
        if value == "Custom":
            self.custom_entry.configure(state=tk.NORMAL)
        else:
            self.custom_entry.configure(state=tk.DISABLED)
            self.custom_entry.delete(0, tk.END)
            self.custom_entry.insert(0, value)

    def load_audio_devices(self):
        """Load and display available audio input and output devices - ACCEPTS BOTH MONO AND STEREO"""
        try:
            devices = sd.query_devices()
            self.input_device_list = []
            self.output_device_list = []
            input_device_names = []
            output_device_names = []
            
            for i, device in enumerate(devices):
                # Input devices - ACCEPT BOTH MONO AND STEREO
                if device['max_input_channels'] > 0:
                    self.input_device_list.append(i)
                    channels = device['max_input_channels']
                    ch_type = "Stereo" if channels >= 2 else "Mono (→stereo)"
                    device_name = f"{i}: {device['name']} ({ch_type}, {channels} ch)"
                    input_device_names.append(device_name)
                
                # Output devices
                if device['max_output_channels'] > 0:
                    self.output_device_list.append(i)
                    device_name = f"{i}: {device['name']} (Out: {device['max_output_channels']})"
                    output_device_names.append(device_name)
            
            if not self.input_device_list:
                messagebox.showerror("Error", "No audio input devices found!")
                self.status_label.config(text="Status: No input devices available", fg="red")
                return
            
            self.input_device_combo['values'] = input_device_names
            if input_device_names:
                self.input_device_combo.current(0)
                self.input_device = self.input_device_list[0]
            
            if not self.output_device_list:
                messagebox.showwarning("Warning", "No audio output devices found! Live playback won't work.")
                self.output_device = None
            else:
                self.output_device_combo['values'] = output_device_names
                if output_device_names:
                    default_output = sd.default.device[1]
                    if default_output in self.output_device_list:
                        default_idx = self.output_device_list.index(default_output)
                        self.output_device_combo.current(default_idx)
                        self.output_device = default_output
                    else:
                        self.output_device_combo.current(0)
                        self.output_device = self.output_device_list[0]
            
            status_text = f"Status: Input: {input_device_names[0] if input_device_names else 'None'}"
            if self.output_device is not None:
                status_text += f" | Output: Device {self.output_device}"
            self.status_label.config(text=status_text, fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio devices: {e}")
            self.status_label.config(text=f"Status: Error loading devices", fg="red")

    def on_input_device_selected(self, event):
        """Handle input device selection"""
        try:
            selected_index = self.input_device_combo.current()
            if selected_index >= 0:
                self.input_device = self.input_device_list[selected_index]
                device_name = self.input_device_combo.get()
                print(f"Input device selected: {device_name}")
                self.update_device_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select input device: {e}")
    
    def on_output_device_selected(self, event):
        """Handle output device selection"""
        try:
            selected_index = self.output_device_combo.current()
            if selected_index >= 0:
                self.output_device = self.output_device_list[selected_index]
                device_name = self.output_device_combo.get()
                print(f"Output device selected: {device_name}")
                self.update_device_status()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select output device: {e}")
    
    def update_device_status(self):
        """Update status label with current device selection"""
        status_parts = []
        if self.input_device is not None:
            status_parts.append(f"Input: Device {self.input_device}")
        if self.output_device is not None:
            status_parts.append(f"Output: Device {self.output_device}")
        
        if status_parts:
            self.status_label.config(text=f"Status: {' | '.join(status_parts)}", fg="green")
        else:
            self.status_label.config(text="Status: No devices selected", fg="orange")

    def toggle_mic_test(self):
        """Toggle microphone test with live playback"""
        if self.is_testing_mic:
            self.stop_mic_test()
        else:
            self.start_mic_test()

    def start_mic_test(self):
        """Start microphone test with live audio playback - HANDLES MONO AND STEREO"""
        if self.input_device is None:
            messagebox.showerror("Error", "Please select an audio input device first!")
            return
        
        if self.output_device is None:
            messagebox.showerror("Error", "Please select an audio output device for playback!")
            return
        
        self.is_testing_mic = True
        self.test_button.config(text="Stop Mic Test", bg="#f44336")
        self.status_label.config(text="Status: Testing microphone - speak into mic!", fg="orange")
        
        self.test_thread = threading.Thread(target=self.run_mic_test, daemon=True)
        self.test_thread.start()

    def stop_mic_test(self):
        """Stop microphone test"""
        self.is_testing_mic = False
        self.test_button.config(text="Start Mic Test (Live Playback)", bg="#4CAF50")
        self.level_bar['value'] = 0
        self.level_label.config(text="Level: ")
        self.status_label.config(text="Status: Mic test stopped", fg="green")

    def run_mic_test(self):
        """Run microphone test with live playback - HANDLES MONO AND STEREO"""
        def callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            
            if not self.is_testing_mic:
                outdata[:] = np.zeros_like(outdata)
                return
            
            # Convert mono to stereo if needed
            if indata.shape[1] == 1:
                audio_stereo = np.repeat(indata, 2, axis=1)
            else:
                audio_stereo = indata.copy()
            
            # Apply noise reduction if enabled - per channel
            if self.enable_noise_reduction and self.noise_profile is not None:
                audio_L = self.apply_noise_reduction(audio_stereo[:, 0].copy())
                audio_R = self.apply_noise_reduction(audio_stereo[:, 1].copy())
                outdata[:, 0] = audio_L
                outdata[:, 1] = audio_R
            else:
                audio_L = self.apply_highpass_filter(audio_stereo[:, 0].copy())
                audio_R = self.apply_highpass_filter(audio_stereo[:, 1].copy())
                outdata[:, 0] = audio_L
                outdata[:, 1] = audio_R
            
            # Calculate audio level
            level = np.sqrt(np.mean(outdata**2)) * 100
            self.audio_queue.put(level)
        
        try:
            # Detect input channels
            device_info = sd.query_devices(self.input_device, 'input')
            input_channels = min(2, device_info['max_input_channels'])
            
            with sd.Stream(device=(self.input_device, self.output_device),
                          channels=(input_channels, 2),  # Input: mono or stereo, Output: stereo
                          samplerate=self.fs,
                          dtype='float32',
                          blocksize=2048,
                          callback=callback):
                
                print(f"Mic test started: Input={self.input_device} ({input_channels}ch), Output={self.output_device}")
                
                while self.is_testing_mic:
                    try:
                        level = self.audio_queue.get_nowait()
                        display_level = min(level * 20, 100)
                        self.root.after(0, self.update_level_display, display_level)
                    except queue.Empty:
                        pass
                    time.sleep(0.05)
                    
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Microphone Test Error", 
                          f"Error during mic test: {e}\n\nMake sure your devices are not being used by another application.")
            self.root.after(0, self.stop_mic_test)

    def update_level_display(self, level):
        """Update the level bar display"""
        self.level_bar['value'] = level
        self.level_label.config(text=f"Level: {int(level)}%")

    def start_recording(self):
        """Start recording keystrokes and audio - HANDLES MONO AND STEREO"""
        if self.input_device is None:
            messagebox.showerror("Error", "Please select an audio input device first!")
            return
        
        if self.is_testing_mic:
            messagebox.showwarning("Warning", "Please stop the microphone test before starting recording.")
            return
        
        # Get duration from UI
        try:
            if self.duration_var.get() == "Custom":
                duration = int(self.custom_entry.get())
            else:
                duration = int(self.duration_var.get())
            if duration <= 0:
                raise ValueError
            self.recording_duration = duration
        except Exception:
            messagebox.showerror("Error", "Please enter a valid recording duration in seconds.")
            return
        
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.remaining_time = self.recording_duration
        self.update_timer_label()
        
        self.is_recording = True
        self.timer_running = True
        self.keys_recorded = 0
        self.keys_rejected = 0
        self.pressed_keys.clear()
        self.stats_label.config(text="Keys recorded: 0 | Rejected (noise): 0")
        self.root.after(1000, self.countdown_timer)
        
        nr_status = "ON" if self.enable_noise_reduction else "OFF"
        self.status_label.config(text=f"Status: Recording (NR: {nr_status})... Type!", fg="red")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.DISABLED)
        self.update_ids_btn.config(state=tk.DISABLED)
        self.calibrate_button.config(state=tk.DISABLED)
        
        # Reset audio buffer - STEREO
        with self.buffer_lock:
            self.audio_buffer = np.zeros((self.buffer_samples, 2), dtype=np.float32)
        
        self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.audio_thread.start()
        
        self.keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.keyboard_thread.start()
        
        # Detect actual recording format
        device_info = sd.query_devices(self.input_device, 'input')
        record_channels = min(2, device_info['max_input_channels'])
        
        print(f"\n{'='*60}")
        print(f"RECORDING SESSION STARTED")
        print(f"Session ID: {self.session_id}")
        print(f"Input: {record_channels} channel{'s' if record_channels > 1 else ''} → Output: Stereo (2 channels)")
        print(f"Format: 2 channels, 44100 Hz, {self.target_segment_samples} samples")
        print(f"Microphone: {self.mic_id}")
        print(f"Keyboard: {self.keyboard_id}")
        print(f"Output folder: {self.output_dir}")
        print(f"Noise Reduction: {nr_status} (Strength: {self.noise_reduction_strength})")
        print(f"{'='*60}\n")

    def stop_recording(self):
        """Stop recording"""
        self.is_recording = False
        self.timer_running = False
        self.timer_label.config(text="")
        self.status_label.config(text=f"Status: Recording stopped - {self.keys_recorded} keys captured, {self.keys_rejected} rejected", fg="green")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.test_button.config(state=tk.NORMAL)
        self.update_ids_btn.config(state=tk.NORMAL)
        self.calibrate_button.config(state=tk.NORMAL)
        
        if self.listener:
            try:
                self.listener.stop()
            except Exception as e:
                print(f"Error stopping keyboard listener: {e}")
        
        print(f"\n{'='*60}")
        print(f"RECORDING SESSION ENDED")
        print(f"Total keys recorded: {self.keys_recorded}")
        print(f"Total keys rejected: {self.keys_rejected}")
        print(f"{'='*60}\n")

    def countdown_timer(self):
        """Update countdown timer"""
        if not self.is_recording or not self.timer_running:
            return
        
        if self.remaining_time > 0:
            self.remaining_time -= 1
            self.update_timer_label()
            self.root.after(1000, self.countdown_timer)
        else:
            self.stop_recording()

    def update_timer_label(self):
        """Update timer display"""
        if self.remaining_time > 0:
            mins, secs = divmod(self.remaining_time, 60)
            self.timer_label.config(text=f"Time left: {mins:02d}:{secs:02d}")
        else:
            self.timer_label.config(text="")

    def record_audio(self):
        """Continuously record audio into a rolling buffer - HANDLES MONO AND STEREO"""
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            
            if not self.is_recording:
                return
            
            with self.buffer_lock:
                # Extract audio data and convert to stereo if needed
                audio_chunk = indata.astype(np.float32).copy()
                
                # Convert mono to stereo if necessary
                if audio_chunk.shape[1] == 1:
                    audio_chunk = np.repeat(audio_chunk, 2, axis=1)
                
                # Apply noise reduction per channel if enabled
                if self.enable_noise_reduction:
                    audio_chunk[:, 0] = self.apply_noise_reduction(audio_chunk[:, 0])
                    audio_chunk[:, 1] = self.apply_noise_reduction(audio_chunk[:, 1])
                
                # Rolling buffer
                self.audio_buffer = np.roll(self.audio_buffer, -frames, axis=0)
                self.audio_buffer[-frames:, :] = audio_chunk
        
        try:
            # Detect number of channels the device supports
            device_info = sd.query_devices(self.input_device, 'input')
            max_channels = device_info['max_input_channels']
            
            # Use 2 channels if available, otherwise 1 (will convert to stereo in callback)
            record_channels = min(2, max_channels)
            
            print(f"Recording with {record_channels} channel(s), will output as stereo")
            
            with sd.InputStream(device=self.input_device, 
                              channels=record_channels,
                              samplerate=self.fs,
                              dtype='float32',
                              callback=callback,
                              blocksize=2048,
                              latency='low'):
                
                print("Audio recording started")
                while self.is_recording:
                    time.sleep(0.05)
                    
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Audio Recording Error", 
                          f"Audio recording error: {e}\n\nMake sure your microphone is not being used by another application.")
            self.root.after(0, self.stop_recording)

    def listen_keyboard(self):
        """Listen for global keyboard events with debouncing"""
        def on_press(key):
            if not self.is_recording:
                return False
            
            try:
                if hasattr(key, 'char') and key.char:
                    k = key.char
                else:
                    k = str(key)
            except Exception:
                k = str(key)
            
            is_valid_key = (k and len(k) == 1 and (k.isalnum() or k.isdigit()))
            
            if is_valid_key:
                current_time = time.time()
                with self.keys_lock:
                    last_time = self.pressed_keys.get(k, 0)
                    time_since_last = current_time - last_time
                    
                    if time_since_last >= self.key_debounce_time:
                        self.pressed_keys[k] = current_time
                        threading.Thread(target=self.save_key_audio, args=(k,), daemon=True).start()
                    else:
                        print(f"Debounced: {k} (time since last: {time_since_last:.3f}s)")
        
        def on_release(key):
            pass
        
        try:
            self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.listener.start()
            self.listener.join()
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Keyboard Listener Error", 
                          f"Keyboard listening error: {e}")
            self.root.after(0, self.stop_recording)

    def get_next_file_number(self, key_folder):
        """Get the next sequential file number for a key folder"""
        try:
            existing_files = [f for f in os.listdir(key_folder) if f.endswith('.wav')]
            if not existing_files:
                return 0
            numbers = []
            for f in existing_files:
                try:
                    num = int(f.replace('.wav', ''))
                    numbers.append(num)
                except ValueError:
                    continue
            return max(numbers) + 1 if numbers else 0
        except Exception:
            return 0

    def save_key_audio(self, key_label):
        """Extract audio segment and save as STEREO WAV with EXACT 20948 samples"""
        clean_key = key_label.lower()
        
        if not (len(clean_key) == 1 and clean_key.isalnum()):
            print(f"Skipping non-alphanumeric key: {key_label}")
            return
        
        key_folder = os.path.join(self.output_dir, clean_key)
        try:
            os.makedirs(key_folder, exist_ok=True)
        except Exception as e:
            print(f"Error creating key folder {key_folder}: {e}")
            return
        
        file_number = self.get_next_file_number(key_folder)
        wav_filename = f"{file_number}.wav"
        wav_path = os.path.join(key_folder, wav_filename)
        relative_wav_path = f"{clean_key}/{wav_filename}"
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        try:
            # Extract EXACTLY target_segment_samples from STEREO buffer
            with self.buffer_lock:
                segment = self.audio_buffer[-self.target_segment_samples:, :].copy()
            
            # Verify exact length
            if segment.shape[0] != self.target_segment_samples:
                print(f"WARNING: Segment length mismatch: {segment.shape[0]} vs {self.target_segment_samples}")
                # Pad or trim to exact length
                if segment.shape[0] < self.target_segment_samples:
                    padding = self.target_segment_samples - segment.shape[0]
                    segment = np.pad(segment, ((0, padding), (0, 0)), mode='constant')
                else:
                    segment = segment[:self.target_segment_samples, :]
            
            # Calculate levels (average both channels)
            rms_level = np.sqrt(np.mean(segment**2))
            peak_level = np.max(np.abs(segment))
            
            # Noise filtering
            if rms_level < self.noise_threshold:
                print(f"REJECTED (noise): {key_label} -> {file_number}.wav - RMS: {rms_level:.6f}")
                self.keys_rejected += 1
                self.root.after(0, self.update_stats)
                return
            
            # Quality assessment
            if rms_level > self.noise_threshold * 5:
                quality = "good"
            elif rms_level > self.noise_threshold * 2:
                quality = "fair"
            else:
                quality = "weak"
            
            # Normalization
            if peak_level > 0.001:
                scale_factor = 0.7 / peak_level
                segment_normalized = segment * scale_factor
            else:
                segment_normalized = segment
            
            segment_normalized = np.clip(segment_normalized, -1.0, 1.0)
            
            # Convert to int16 STEREO
            segment_int16 = np.int16(segment_normalized * 32767)
            
            # Save STEREO WAV file
            with wave.open(wav_path, 'w') as wf:
                wf.setnchannels(2)  # STEREO
                wf.setsampwidth(2)
                wf.setframerate(self.fs)
                wf.writeframes(segment_int16.tobytes())
            
            # Write metadata
            with open(self.metadata_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.metadata_fields)
                writer.writerow({
                    "timestamp": timestamp,
                    "key": key_label,
                    "wav_file": relative_wav_path,
                    "rms_level": f"{rms_level:.6f}",
                    "peak_level": f"{peak_level:.6f}",
                    "quality": quality,
                    "mic_id": self.mic_id,
                    "keyboard_id": self.keyboard_id,
                    "session_id": self.session_id,
                    "file_number": file_number,
                    "channels": 2,
                    "samples": self.target_segment_samples
                })
            
            self.keys_recorded += 1
            self.root.after(0, self.update_stats)
            
            print(f"SAVED [{quality}]: {key_label} -> {relative_wav_path} (STEREO, {self.target_segment_samples} samples, RMS: {rms_level:.4f})")
            
        except Exception as e:
            print(f"Error saving key audio: {e}")
            import traceback
            traceback.print_exc()

    def update_stats(self):
        """Update recording statistics display"""
        self.stats_label.config(text=f"Keys recorded: {self.keys_recorded} | Rejected (noise): {self.keys_rejected}")


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ResearchApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user.")