"""
Continuous recording data collector - records audio continuously and logs keystrokes.
"""
import os
import csv
import threading
import numpy as np
from datetime import datetime
from typing import Optional
import tkinter as tk
from tkinter import ttk, messagebox
from .audio_handler import AudioHandler
from .keystroke_logger import KeystrokeLogger
from .config import Config
import sounddevice as sd


class DataCollectorTab:
    """Tab for continuous audio recording with keystroke logging."""
    
    def __init__(self, parent, config: Config, audio_handler: AudioHandler):
        self.parent = parent
        self.config = config
        self.audio = audio_handler
        
        self.is_recording = False
        self.is_testing_mic = False
        self.recording_thread = None
        self.mic_test_thread = None
        self.keystroke_logger = None
        
        self.current_session = None
        self.keystroke_log = []
        self.recorded_audio = None
        self.start_time = None
        
        self.input_devices = []
        self.output_devices = []
        
        self.build_ui()
        self.load_audio_devices()
    
    def build_ui(self):
        """Build the data collector UI."""
        main_frame = tk.Frame(self.parent, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="Continuous Data Collection",
                        font=("Arial", 16, "bold"), fg="#1976D2")
        title.pack(pady=(0, 20))
        
        # Info frame
        info_frame = tk.LabelFrame(main_frame, text="How It Works", 
                                   font=("Arial", 10, "bold"), padx=15, pady=15)
        info_frame.pack(fill=tk.X, pady=(0, 15))
        
        info_text = """✓ Records audio continuously (no individual file per keystroke)
✓ Logs each keystroke with precise timestamp
✓ Saves one audio file + one keystroke log (CSV)
✓ Later use Data Segmentation tab to extract individual keystrokes
✓ More flexible - change segmentation parameters without re-recording"""
        
        tk.Label(info_frame, text=info_text, font=("Arial", 9),
                justify=tk.LEFT, fg="#424242").pack()
        
        # Device selection
        device_frame = tk.LabelFrame(main_frame, text="Audio Device Setup",
                                    font=("Arial", 10, "bold"), padx=15, pady=15)
        device_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Input device
        input_row = tk.Frame(device_frame)
        input_row.pack(fill=tk.X, pady=5)
        tk.Label(input_row, text="Input Device:", width=12, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.input_device_var = tk.StringVar()
        self.input_device_combo = ttk.Combobox(input_row, textvariable=self.input_device_var,
                                              width=45, state="readonly")
        self.input_device_combo.pack(side=tk.LEFT, padx=5)
        self.input_device_combo.bind("<<ComboboxSelected>>", self.on_input_device_selected)
        
        # Output device
        output_row = tk.Frame(device_frame)
        output_row.pack(fill=tk.X, pady=5)
        tk.Label(output_row, text="Output Device:", width=12, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.output_device_var = tk.StringVar()
        self.output_device_combo = ttk.Combobox(output_row, textvariable=self.output_device_var,
                                               width=45, state="readonly")
        self.output_device_combo.pack(side=tk.LEFT, padx=5)
        self.output_device_combo.bind("<<ComboboxSelected>>", self.on_output_device_selected)
        
        tk.Button(device_frame, text="Refresh Devices", command=self.load_audio_devices).pack(pady=5)
        
        # Mic test
        test_frame = tk.Frame(device_frame)
        test_frame.pack(fill=tk.X, pady=10)
        
        self.test_btn = tk.Button(test_frame, text="Test Microphone",
                                  command=self.toggle_mic_test,
                                  bg="#4CAF50", fg="white",
                                  font=("Arial", 9, "bold"))
        self.test_btn.pack(side=tk.LEFT, padx=5)
        
        tk.Label(test_frame, text="Level:").pack(side=tk.LEFT, padx=10)
        self.level_bar = ttk.Progressbar(test_frame, length=200, mode='determinate')
        self.level_bar.pack(side=tk.LEFT, padx=5)
        
        self.level_label = tk.Label(test_frame, text="0.0", width=8)
        self.level_label.pack(side=tk.LEFT, padx=5)
        
        # Session configuration
        session_frame = tk.LabelFrame(main_frame, text="Session Configuration",
                                     font=("Arial", 10, "bold"), padx=15, pady=15)
        session_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Session name
        name_row = tk.Frame(session_frame)
        name_row.pack(fill=tk.X, pady=5)
        tk.Label(name_row, text="Session Name:", width=15, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.session_name_var = tk.StringVar(value=f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tk.Entry(name_row, textvariable=self.session_name_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # Mic & Keyboard IDs
        id_row = tk.Frame(session_frame)
        id_row.pack(fill=tk.X, pady=5)
        
        tk.Label(id_row, text="Mic ID:", width=8).pack(side=tk.LEFT, padx=5)
        self.mic_id_var = tk.StringVar(value=self.config.get('mic_id', 'mic1'))
        tk.Entry(id_row, textvariable=self.mic_id_var, width=12).pack(side=tk.LEFT, padx=5)
        
        tk.Label(id_row, text="Keyboard ID:", width=10).pack(side=tk.LEFT, padx=5)
        self.kb_id_var = tk.StringVar(value=self.config.get('keyboard_id', 'kb1'))
        tk.Entry(id_row, textvariable=self.kb_id_var, width=12).pack(side=tk.LEFT, padx=5)
        
        # Output directory display
        self.output_label = tk.Label(session_frame, text="", font=("Arial", 9),
                                     fg="#1976D2", justify=tk.LEFT)
        self.output_label.pack(pady=5)
        self.update_output_label()
        
        # Recording controls
        control_frame = tk.LabelFrame(main_frame, text="Recording Controls",
                                     font=("Arial", 10, "bold"), padx=15, pady=15)
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Status
        self.status_label = tk.Label(control_frame, text="Status: Idle",
                                     font=("Arial", 12, "bold"), fg="green")
        self.status_label.pack(pady=10)
        
        # Timer
        self.timer_label = tk.Label(control_frame, text="00:00:00",
                                    font=("Arial", 24, "bold"), fg="#1976D2")
        self.timer_label.pack(pady=10)
        
        # Statistics
        self.stats_label = tk.Label(control_frame, text="Keystrokes: 0",
                                    font=("Arial", 11), fg="#424242")
        self.stats_label.pack(pady=5)
        
        # Buttons
        btn_frame = tk.Frame(control_frame)
        btn_frame.pack(pady=15)
        
        self.start_btn = tk.Button(btn_frame, text="Start Recording",
                                   command=self.start_recording,
                                   bg="#4CAF50", fg="white",
                                   font=("Arial", 12, "bold"),
                                   width=15, height=2)
        self.start_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = tk.Button(btn_frame, text="Stop Recording",
                                  command=self.stop_recording,
                                  bg="#F44336", fg="white",
                                  font=("Arial", 12, "bold"),
                                  width=15, height=2,
                                  state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        # Recent sessions
        recent_frame = tk.LabelFrame(main_frame, text="Recent Sessions",
                                    font=("Arial", 10, "bold"), padx=15, pady=15)
        recent_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Listbox with scrollbar
        list_container = tk.Frame(recent_frame)
        list_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(list_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.session_listbox = tk.Listbox(list_container, yscrollcommand=scrollbar.set,
                                          font=("Courier", 9))
        self.session_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.session_listbox.yview)
        
        # Refresh button
        tk.Button(recent_frame, text="Refresh List", command=self.load_recent_sessions).pack(pady=5)
        
        self.load_recent_sessions()
    
    def update_output_label(self):
        """Update the output directory label."""
        base_dir = self.config.get('base_output_dir', 'recordings')
        mic = self.mic_id_var.get()
        kb = self.kb_id_var.get()
        session = self.session_name_var.get()
        
        output_path = os.path.join(base_dir, f"{mic}-{kb}", "continuous", session)
        self.output_label.config(text=f"Output: {output_path}")
    
    def load_recent_sessions(self):
        """Load and display recent recording sessions."""
        self.session_listbox.delete(0, tk.END)
        
        base_dir = self.config.get('base_output_dir', 'recordings')
        if not os.path.exists(base_dir):
            return
        
        sessions = []
        for root, dirs, files in os.walk(base_dir):
            if 'continuous' in root and 'keystroke_log.csv' in files:
                # Get session info
                session_path = root
                audio_file = os.path.join(root, 'audio.wav')
                log_file = os.path.join(root, 'keystroke_log.csv')
                
                if os.path.exists(audio_file):
                    # Count keystrokes
                    try:
                        with open(log_file, 'r') as f:
                            keystroke_count = sum(1 for line in f) - 1  # minus header
                        
                        # Get file size
                        audio_size = os.path.getsize(audio_file) / (1024 * 1024)  # MB
                        
                        rel_path = os.path.relpath(session_path, base_dir)
                        sessions.append(f"{rel_path} - {keystroke_count} keys, {audio_size:.1f} MB")
                    except:
                        pass
        
        for session in sorted(sessions, reverse=True)[:20]:
            self.session_listbox.insert(tk.END, session)
    
    def start_recording(self):
        """Start continuous recording."""
        if self.is_recording:
            return
        
        # Update config
        self.config.set('mic_id', self.mic_id_var.get())
        self.config.set('keyboard_id', self.kb_id_var.get())
        
        # Create output directory
        base_dir = self.config.get('base_output_dir', 'recordings')
        mic = self.mic_id_var.get()
        kb = self.kb_id_var.get()
        session = self.session_name_var.get()
        
        self.current_session = os.path.join(base_dir, f"{mic}-{kb}", "continuous", session)
        os.makedirs(self.current_session, exist_ok=True)
        
        # Initialize
        self.keystroke_log = []
        self.recorded_audio = []
        self.start_time = datetime.now()
        self.is_recording = True
        
        # Start keystroke logger
        self.keystroke_logger = KeystrokeLogger(on_key_press=self.on_keystroke)
        self.keystroke_logger.start()
        
        # Start recording thread
        self.recording_thread = threading.Thread(target=self.record_continuous, daemon=True)
        self.recording_thread.start()
        
        # Update UI
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Status: Recording...", fg="red")
        
        # Start timer
        self.update_timer()
    
    def stop_recording(self):
        """Stop recording and save data."""
        if not self.is_recording:
            return
        
        self.is_recording = False
        
        # Stop keystroke logger
        if self.keystroke_logger:
            self.keystroke_logger.stop()
        
        # Wait for recording thread
        if self.recording_thread:
            self.recording_thread.join(timeout=2.0)
        
        # Save audio
        self.status_label.config(text="Status: Saving...", fg="orange")
        self.save_session()
        
        # Update UI
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Saved", fg="green")
        
        # Refresh list
        self.load_recent_sessions()
        
        messagebox.showinfo("Success", f"Session saved to:\n{self.current_session}")
    
    def record_continuous(self):
        """Record audio continuously in chunks - MATCHES main_old.py."""
        chunk_duration = 1.0  # Record in 1-second chunks
        
        # Get input device info and adjust if needed (like main_old.py)
        try:
            input_dev = self.config.get('input_device')
            if input_dev is not None:
                import sounddevice as sd
                device_info = sd.query_devices(input_dev, 'input')
                max_channels = device_info.get('max_input_channels', 2)
                default_samplerate = int(device_info.get('default_samplerate', 44100))
                
                # Update audio handler channels if needed
                if max_channels < self.audio.channels:
                    print(f"Device supports {max_channels} channel(s), adjusting from {self.audio.channels}")
                    self.audio.set_channels(max_channels)
                
                # Update sample rate if device doesn't support current rate
                if default_samplerate != self.audio.sample_rate:
                    print(f"Adjusting sample rate to {default_samplerate} Hz for recording")
                    self.audio.set_sample_rate(default_samplerate)
        except Exception as e:
            print(f"Warning: Could not detect device settings: {e}")
        
        while self.is_recording:
            try:
                chunk = self.audio.record_stream(chunk_duration)
                self.recorded_audio.append(chunk)
            except Exception as e:
                print(f"Recording error: {e}")
                self.parent.after(0, lambda: messagebox.showerror("Recording Error", 
                    f"Failed to record: {e}\n\nMake sure your microphone is not in use by another application."))
                break
    
    def on_keystroke(self, key: str, timestamp: float):
        """Handle keystroke event."""
        # Calculate relative time from recording start
        relative_time = timestamp - self.start_time.timestamp()
        
        # Add to log
        self.keystroke_log.append({
            'timestamp': timestamp,
            'relative_time': relative_time,
            'key': key,
            'datetime': datetime.fromtimestamp(timestamp).isoformat()
        })
        
        # Update UI
        self.stats_label.config(text=f"Keystrokes: {len(self.keystroke_log)}")
    
    def save_session(self):
        """Save recorded audio and keystroke log."""
        try:
            # Concatenate audio chunks
            if self.recorded_audio:
                full_audio = np.concatenate(self.recorded_audio, axis=0)
                
                # Save audio
                audio_path = os.path.join(self.current_session, 'audio.wav')
                self.audio.save_audio(audio_path, full_audio)
            
            # Save keystroke log
            log_path = os.path.join(self.current_session, 'keystroke_log.csv')
            with open(log_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['timestamp', 'relative_time', 'key', 'datetime'])
                writer.writeheader()
                writer.writerows(self.keystroke_log)
            
            # Save session info
            info_path = os.path.join(self.current_session, 'session_info.txt')
            with open(info_path, 'w') as f:
                f.write(f"Session: {self.session_name_var.get()}\n")
                f.write(f"Microphone: {self.mic_id_var.get()}\n")
                f.write(f"Keyboard: {self.kb_id_var.get()}\n")
                f.write(f"Start Time: {self.start_time.isoformat()}\n")
                f.write(f"Duration: {len(self.recorded_audio)} seconds\n")
                f.write(f"Keystrokes: {len(self.keystroke_log)}\n")
                f.write(f"Sample Rate: {self.config.get('sample_rate')}\n")
                f.write(f"Channels: {self.config.get('channels')}\n")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save session: {e}")
    
    def update_timer(self):
        """Update the recording timer."""
        if self.is_recording and self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            
            self.timer_label.config(text=f"{hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Schedule next update
            self.parent.after(1000, self.update_timer)
    
    def load_audio_devices(self):
        """Load available audio devices."""
        try:
            self.input_devices, self.output_devices = self.audio.get_audio_devices()
            
            # Update input combo
            input_names = [f"{d['index']}: {d['name']}" for d in self.input_devices]
            self.input_device_combo['values'] = input_names
            
            # Set current device if configured
            current_input = self.config.get('input_device')
            if current_input is not None:
                for i, d in enumerate(self.input_devices):
                    if d['index'] == current_input:
                        self.input_device_combo.current(i)
                        break
            elif len(input_names) > 0:
                self.input_device_combo.current(0)
            
            # Update output combo
            output_names = [f"{d['index']}: {d['name']}" for d in self.output_devices]
            self.output_device_combo['values'] = output_names
            
            # Set current device if configured
            current_output = self.config.get('output_device')
            if current_output is not None:
                for i, d in enumerate(self.output_devices):
                    if d['index'] == current_output:
                        self.output_device_combo.current(i)
                        break
            elif len(output_names) > 0:
                self.output_device_combo.current(0)
                
        except Exception as e:
            print(f"Error loading devices: {e}")
    
    def on_input_device_selected(self, event):
        """Handle input device selection - matches main_old.py implementation."""
        try:
            selection = self.input_device_var.get()
            if selection:
                device_idx = int(selection.split(':')[0])
                self.audio.set_device(input_device=device_idx)
                self.config.set('input_device', device_idx)
                
                # Auto-adjust channels and sample rate based on device capabilities
                device_info = self.audio.get_device_info(device_idx)
                max_input_channels = device_info.get('max_input_channels', 2)
                default_samplerate = int(device_info.get('default_samplerate', 44100))
                
                # Use min of configured channels and device max channels
                configured_channels = self.config.get('channels', 2)
                actual_channels = min(configured_channels, max_input_channels)
                
                if actual_channels != self.audio.channels:
                    print(f"Adjusting channels from {self.audio.channels} to {actual_channels} for device {device_idx}")
                    self.audio.set_channels(actual_channels)
                
                # Adjust sample rate if device doesn't support current rate
                if default_samplerate != self.audio.sample_rate:
                    print(f"Adjusting sample rate from {self.audio.sample_rate} to {default_samplerate} for device {device_idx}")
                    self.audio.set_sample_rate(default_samplerate)
                    self.config.set('sample_rate', default_samplerate)
        except Exception as e:
            print(f"Error setting input device: {e}")
    
    def on_output_device_selected(self, event):
        """Handle output device selection."""
        try:
            selection = self.output_device_var.get()
            if selection:
                device_idx = int(selection.split(':')[0])
                self.audio.set_device(output_device=device_idx)
                self.config.set('output_device', device_idx)
        except Exception as e:
            print(f"Error setting output device: {e}")
    
    def toggle_mic_test(self):
        """Toggle microphone test."""
        if self.is_testing_mic:
            self.stop_mic_test()
        else:
            self.start_mic_test()
    
    def start_mic_test(self):
        """Start microphone test with live audio playback - MATCHES main_old.py."""
        input_dev = self.config.get('input_device')
        output_dev = self.config.get('output_device')
        
        if input_dev is None:
            messagebox.showerror("Error", "Please select an audio input device first!")
            return
        
        if output_dev is None:
            messagebox.showerror("Error", "Please select an audio output device for playback!")
            return
        
        if self.is_recording or self.is_testing_mic:
            return
        
        self.is_testing_mic = True
        self.test_btn.config(text="Stop Test", bg="#F44336")
        
        # Start test thread
        self.mic_test_thread = threading.Thread(target=self.run_mic_test, daemon=True)
        self.mic_test_thread.start()
    
    def stop_mic_test(self):
        """Stop microphone test."""
        self.is_testing_mic = False
        if self.test_btn.winfo_exists():
            self.test_btn.config(text="Test Microphone", bg="#4CAF50")
            self.level_bar['value'] = 0
            self.level_label.config(text="0.0")
    
    def run_mic_test(self):
        """Run microphone test with live playback - MATCHES main_old.py EXACTLY."""
        import queue
        audio_queue = queue.Queue()
        
        def callback(indata, outdata, frames, time_info, status):
            if status:
                print(f"Status: {status}")
            
            if not self.is_testing_mic:
                outdata.fill(0)
                return
            
            # Convert mono to stereo if needed
            if indata.shape[1] == 1:
                audio_stereo = np.column_stack([indata[:, 0], indata[:, 0]])
            else:
                audio_stereo = indata.copy()
            
            # Apply bandpass filter per channel for clean sound
            audio_L = self.audio.apply_bandpass_filter(audio_stereo[:, 0].copy())
            audio_R = self.audio.apply_bandpass_filter(audio_stereo[:, 1].copy())
            
            outdata[:, 0] = audio_L
            outdata[:, 1] = audio_R
            
            # Calculate audio level
            level = np.sqrt(np.mean(outdata**2)) * 100
            audio_queue.put(level)
        
        try:
            input_dev = self.config.get('input_device')
            output_dev = self.config.get('output_device')
            
            # Detect input channels
            device_info = self.audio.get_device_info(input_dev)
            input_channels = min(2, device_info.get('max_input_channels', 1))
            
            with sd.Stream(device=(input_dev, output_dev),
                          channels=(input_channels, 2),
                          samplerate=self.audio.sample_rate,
                          dtype='float32',
                          blocksize=2048,
                          callback=callback):
                
                while self.is_testing_mic:
                    try:
                        level = audio_queue.get(timeout=0.1)
                        self.parent.after(0, lambda l=level: self.update_level_display(l))
                    except:
                        pass
                    
        except Exception as e:
            print(f"Microphone Test Error: {e}")
            self.parent.after(0, lambda: messagebox.showerror("Microphone Test Error",
                f"Error during mic test: {e}\n\nMake sure your devices are not being used by another application."))
            self.parent.after(0, self.stop_mic_test)
    
    def update_level_display(self, level):
        """Update level display."""
        try:
            # Scale to 0-100
            display_level = min(level * 500, 100)
            self.level_bar['value'] = display_level
            self.level_label.config(text=f"{level:.4f}")
        except:
            pass
