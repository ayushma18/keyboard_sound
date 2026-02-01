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
        
        # Audio parameters
        self.fs = 44100  # Sample rate
        self.buffer_duration = 2.0  # seconds
        self.segment_duration = 0.25  # seconds per key (increased for better capture)
        self.buffer_samples = int(self.fs * self.buffer_duration)
        self.segment_samples = int(self.fs * self.segment_duration)
        
        # Initialize buffer properly
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self.buffer_lock = threading.Lock()
        
        # KEY DEBOUNCING: Track pressed keys and timing
        self.pressed_keys = {}  # key -> timestamp of last recording
        self.key_debounce_time = 0.15  # minimum time between same key recordings (seconds)
        self.keys_lock = threading.Lock()
        
        # Audio device
        self.input_device = None
        self.device_list = []
        
        # Noise threshold for filtering
        self.noise_threshold = 0.001  # Minimum RMS level to consider as valid signal
        
        # Microphone and Keyboard IDs
        self.mic_id = "mic1"
        self.keyboard_id = "kb1"
        self.session_folder = None
        
        # Output directories - will be set based on mic/keyboard IDs
        self.base_output_dir = "recordings"
        self.output_dir = None
        self.metadata_file = None
        self.metadata_fields = ["timestamp", "key", "wav_file", "rms_level", "peak_level", "quality", 
                               "mic_id", "keyboard_id", "session_id"]
        
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
        
        tk.Label(device_frame, text="Input Device:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.device_var = tk.StringVar()
        self.device_combo = ttk.Combobox(device_frame, textvariable=self.device_var, 
                                         width=50, state="readonly")
        self.device_combo.grid(row=0, column=1, padx=5, pady=5)
        self.device_combo.bind("<<ComboboxSelected>>", self.on_device_selected)
        
        refresh_btn = tk.Button(device_frame, text="Refresh Devices", 
                               command=self.load_audio_devices)
        refresh_btn.grid(row=0, column=2, padx=5, pady=5)
        
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
        noise_frame = tk.Frame(duration_frame)
        noise_frame.pack(pady=5)
        tk.Label(noise_frame, text="Noise Threshold:").pack(side=tk.LEFT, padx=5)
        self.threshold_var = tk.DoubleVar(value=0.001)
        threshold_scale = tk.Scale(noise_frame, from_=0.0001, to=0.01, resolution=0.0001,
                                  orient=tk.HORIZONTAL, variable=self.threshold_var,
                                  length=200, command=self.update_threshold)
        threshold_scale.pack(side=tk.LEFT, padx=5)
        self.threshold_label = tk.Label(noise_frame, text="0.0010")
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
        info_text = "Instructions:\n1. Set Microphone ID and Keyboard ID for this session\n2. Select audio device and test microphone\n3. Start recording and type on your keyboard\n\nEach key's audio will be saved in its own subfolder (e.g., a/, b/, space/, etc.)"
        info_label = tk.Label(main_frame, text=info_text, 
                            font=("Arial", 9), fg="gray", justify=tk.LEFT)
        info_label.pack(pady=(10, 0))
        
        # Set minimum window size
        self.root.minsize(700, 650)
        
        # Initialize stats
        self.keys_recorded = 0
        self.keys_rejected = 0
        self.session_id = None

    def update_session_ids(self):
        """Update microphone and keyboard IDs from UI"""
        new_mic_id = self.mic_id_entry.get().strip()
        new_keyboard_id = self.keyboard_id_entry.get().strip()
        
        # Validate IDs (only alphanumeric, underscore, hyphen)
        import re
        if not re.match(r'^[a-zA-Z0-9_-]+$', new_mic_id):
            messagebox.showerror("Invalid ID", "Microphone ID can only contain letters, numbers, underscore, and hyphen.")
            return
        
        if not re.match(r'^[a-zA-Z0-9_-]+$', new_keyboard_id):
            messagebox.showerror("Invalid ID", "Keyboard ID can only contain letters, numbers, underscore, and hyphen.")
            return
        
        # Check if IDs changed
        if new_mic_id != self.mic_id or new_keyboard_id != self.keyboard_id:
            self.mic_id = new_mic_id
            self.keyboard_id = new_keyboard_id
            
            # Save config
            self.save_config()
            
            # Update output directory
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
        """Load and display available audio input devices"""
        try:
            devices = sd.query_devices()
            self.device_list = []
            device_names = []
            
            for i, device in enumerate(devices):
                if device['max_input_channels'] > 0:
                    self.device_list.append(i)
                    device_name = f"{i}: {device['name']} (Channels: {device['max_input_channels']})"
                    device_names.append(device_name)
            
            if not self.device_list:
                messagebox.showerror("Error", "No audio input devices found!")
                self.status_label.config(text="Status: No input devices available", fg="red")
                return
            
            self.device_combo['values'] = device_names
            if device_names:
                self.device_combo.current(0)
                self.input_device = self.device_list[0]
                self.status_label.config(text=f"Status: Device selected - {device_names[0]}", fg="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio devices: {e}")
            self.status_label.config(text=f"Status: Error loading devices", fg="red")

    def on_device_selected(self, event):
        """Handle device selection"""
        try:
            selected_index = self.device_combo.current()
            if selected_index >= 0:
                self.input_device = self.device_list[selected_index]
                device_name = self.device_combo.get()
                self.status_label.config(text=f"Status: Device selected - {device_name}", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to select device: {e}")

    def toggle_mic_test(self):
        """Toggle microphone test with live playback"""
        if self.is_testing_mic:
            self.stop_mic_test()
        else:
            self.start_mic_test()

    def start_mic_test(self):
        """Start microphone test with live audio playback"""
        if self.input_device is None:
            messagebox.showerror("Error", "Please select an audio input device first!")
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
        """Run microphone test with live playback and level monitoring"""
        def callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            
            if not self.is_testing_mic:
                outdata[:] = np.zeros_like(outdata)
                return
            
            # Copy input to output for playback
            outdata[:] = indata
            
            # Calculate audio level for display
            level = np.sqrt(np.mean(indata**2)) * 100  # RMS level
            self.audio_queue.put(level)
        
        try:
            # Open duplex stream (input and output)
            with sd.Stream(device=(self.input_device, sd.default.device[1]),
                          channels=1, 
                          samplerate=self.fs,
                          dtype='float32',
                          blocksize=2048,
                          callback=callback):
                
                while self.is_testing_mic:
                    try:
                        level = self.audio_queue.get_nowait()
                        # Update level bar (scale to 0-100)
                        display_level = min(level * 20, 100)
                        self.root.after(0, self.update_level_display, display_level)
                    except queue.Empty:
                        pass
                    time.sleep(0.05)
                    
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Microphone Test Error", 
                          f"Error during mic test: {e}\n\nMake sure your microphone is not being used by another application.")
            self.root.after(0, self.stop_mic_test)

    def update_level_display(self, level):
        """Update the level bar display"""
        self.level_bar['value'] = level
        self.level_label.config(text=f"Level: {int(level)}%")

    def start_recording(self):
        """Start recording keystrokes and audio"""
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
        
        # Generate unique session ID for this recording session
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.remaining_time = self.recording_duration
        self.update_timer_label()
        
        self.is_recording = True
        self.timer_running = True
        self.keys_recorded = 0
        self.keys_rejected = 0
        self.pressed_keys.clear()  # Clear key debounce tracking
        self.stats_label.config(text="Keys recorded: 0 | Rejected (noise): 0")
        self.root.after(1000, self.countdown_timer)
        
        self.status_label.config(text=f"Status: Recording session {self.session_id}... Type on your keyboard!", fg="red")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.test_button.config(state=tk.DISABLED)
        self.update_ids_btn.config(state=tk.DISABLED)  # Prevent changing IDs during recording
        
        # Reset audio buffer properly
        with self.buffer_lock:
            self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        
        self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.audio_thread.start()
        
        self.keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.keyboard_thread.start()
        
        print(f"\n{'='*60}")
        print(f"RECORDING SESSION STARTED")
        print(f"Session ID: {self.session_id}")
        print(f"Microphone: {self.mic_id}")
        print(f"Keyboard: {self.keyboard_id}")
        print(f"Output folder: {self.output_dir}")
        print(f"Sounds will be organized into key-specific subfolders")
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
        self.update_ids_btn.config(state=tk.NORMAL)  # Re-enable ID changes
        
        if self.listener:
            try:
                self.listener.stop()
            except Exception as e:
                print(f"Error stopping keyboard listener: {e}")
        
        print(f"\n{'='*60}")
        print(f"RECORDING SESSION ENDED")
        print(f"Session ID: {self.session_id}")
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
        """Continuously record audio into a rolling buffer"""
        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio callback status: {status}")
            
            if not self.is_recording:
                return
            
            # Proper buffer update with correct data handling
            with self.buffer_lock:
                # Extract float32 audio data correctly
                audio_chunk = indata[:, 0].astype(np.float32).copy()
                
                # Rolling buffer: shift left and append new frames
                self.audio_buffer = np.roll(self.audio_buffer, -frames)
                self.audio_buffer[-frames:] = audio_chunk
        
        try:
            # Specify dtype explicitly and use proper blocksize
            with sd.InputStream(device=self.input_device, 
                              channels=1, 
                              samplerate=self.fs,
                              dtype='float32',
                              callback=callback,
                              blocksize=2048,
                              latency='low'):
                
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
                return False  # Stop listener if not recording
            
            try:
                # Get key identifier
                if hasattr(key, 'char') and key.char:
                    k = key.char
                else:
                    k = str(key)
            except Exception:
                k = str(key)
            
            # Only record specific keys
            is_valid_key = (
                (k and len(k) == 1 and k.isprintable()) or 
                k in ['Key.space', 'Key.enter', 'Key.backspace', 'Key.shift', 'Key.tab']
            )
            
            if is_valid_key:
                # KEY DEBOUNCING: Check if this key was recently recorded
                current_time = time.time()
                with self.keys_lock:
                    last_time = self.pressed_keys.get(k, 0)
                    time_since_last = current_time - last_time
                    
                    # Only record if enough time has passed since last recording of this key
                    if time_since_last >= self.key_debounce_time:
                        self.pressed_keys[k] = current_time
                        # Save audio in separate thread to avoid blocking
                        threading.Thread(target=self.save_key_audio, args=(k,), daemon=True).start()
                    else:
                        # Key is being held down or pressed too quickly - ignore
                        print(f"Debounced: {k} (time since last: {time_since_last:.3f}s)")
        
        def on_release(key):
            # Optional: could reset debounce timer on release for more precise control
            pass
        
        try:
            self.listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self.listener.start()
            self.listener.join()
        except Exception as e:
            self.root.after(0, messagebox.showerror, "Keyboard Listener Error", 
                          f"Keyboard listening error: {e}")
            self.root.after(0, self.stop_recording)

    def save_key_audio(self, key_label):
        """Extract audio segment from buffer and save as WAV with noise filtering in key-specific subfolders"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Clean key label for filename and folder
        clean_key = key_label.replace("Key.", "").replace(".", "_")
        
        # Create key-specific subfolder
        key_folder = os.path.join(self.output_dir, clean_key)
        try:
            os.makedirs(key_folder, exist_ok=True)
        except Exception as e:
            print(f"Error creating key folder {key_folder}: {e}")
            return
        
        wav_filename = f"{clean_key}_{timestamp}.wav"
        wav_path = os.path.join(key_folder, wav_filename)
        
        # Update the relative path for metadata (includes subfolder)
        relative_wav_path = f"{clean_key}/{wav_filename}"
        
        try:
            # Extract the most recent segment
            with self.buffer_lock:
                segment = self.audio_buffer[-self.segment_samples:].copy()
            
            # Calculate audio levels
            rms_level = np.sqrt(np.mean(segment**2))
            peak_level = np.max(np.abs(segment))
            
            # NOISE FILTERING: Check if signal is above threshold
            if rms_level < self.noise_threshold:
                # Signal is too weak - likely just noise
                print(f"REJECTED (noise): {key_label} - RMS: {rms_level:.6f} < threshold: {self.noise_threshold:.6f}")
                self.keys_rejected += 1
                self.root.after(0, self.update_stats)
                return
            
            # Signal is valid - determine quality
            if rms_level > self.noise_threshold * 5:
                quality = "good"
            elif rms_level > self.noise_threshold * 2:
                quality = "fair"
            else:
                quality = "weak"
            
            # Better normalization strategy
            if peak_level > 0.001:
                # Use adaptive scaling based on peak level
                # Target peak at 70% to avoid clipping while maintaining dynamics
                scale_factor = 0.7 / peak_level
                segment_normalized = segment * scale_factor
            else:
                segment_normalized = segment
            
            # Clip to prevent overflow
            segment_normalized = np.clip(segment_normalized, -1.0, 1.0)
            
            # Convert to int16 for WAV file
            segment_int16 = np.int16(segment_normalized * 32767)
            
            # Save WAV file
            with wave.open(wav_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.fs)
                wf.writeframes(segment_int16.tobytes())
            
            # Write metadata with audio levels, quality, and session info
            with open(self.metadata_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.metadata_fields)
                writer.writerow({
                    "timestamp": timestamp,
                    "key": key_label,
                    "wav_file": relative_wav_path,  # Include subfolder in path
                    "rms_level": f"{rms_level:.6f}",
                    "peak_level": f"{peak_level:.6f}",
                    "quality": quality,
                    "mic_id": self.mic_id,
                    "keyboard_id": self.keyboard_id,
                    "session_id": self.session_id
                })
            
            # Update stats
            self.keys_recorded += 1
            self.root.after(0, self.update_stats)
            
            print(f"SAVED [{quality}]: {key_label} -> {self.session_folder}/{relative_wav_path} (RMS: {rms_level:.4f}, Peak: {peak_level:.4f})")
            
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