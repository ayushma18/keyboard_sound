import tkinter as tk
from tkinter import messagebox
import threading
import time
import sounddevice as sd
import numpy as np
import wave
import os
import csv
from pynput import keyboard
from datetime import datetime

class ResearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Keyboard Acoustic Side-Channel Research Tool")
        self.is_recording = False

        # --- Recording duration UI ---
        self.recording_duration = 60  # default seconds
        self.remaining_time = 0
        self.timer_running = False
        self.duration_var = tk.StringVar(value="60")
        self.timer_label = tk.Label(root, text="", font=("Arial", 12), fg="blue")
        self.timer_label.pack(pady=2)
        self.build_duration_selector(root)

        self.status_label = tk.Label(root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(pady=10)

        self.start_button = tk.Button(root, text="Start Recording", command=self.start_recording)
        self.start_button.pack(pady=5)

        self.stop_button = tk.Button(root, text="Stop Recording", command=self.stop_recording, state=tk.DISABLED)
        self.stop_button.pack(pady=5)

        # Set minimum window size for proper label display
        self.root.minsize(450, 300)

        # Audio buffer parameters
        self.fs = 44100  # Sample rate
        self.buffer_duration = 2.0  # seconds
        self.segment_duration = 0.2  # seconds per key
        self.buffer_samples = int(self.fs * self.buffer_duration)
        self.segment_samples = int(self.fs * self.segment_duration)
        self.audio_buffer = np.zeros(self.buffer_samples, dtype=np.float32)
        self.buffer_lock = threading.Lock()

        # Select a valid input device for sounddevice
        self.input_device = None
        try:
            devices = sd.query_devices()
            input_devices = [i for i, d in enumerate(devices) if d['max_input_channels'] > 0]
            if input_devices:
                self.input_device = input_devices[0]
            else:
                self.input_device = None
        except Exception as e:
            self.input_device = None

        # Output directories
        self.output_dir = "recordings"
        try:
            os.makedirs(self.output_dir, exist_ok=True)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to create output directory: {e}")

        self.metadata_file = os.path.join(self.output_dir, "metadata.csv")
        self.metadata_fields = ["timestamp", "key", "wav_file"]

        # Threads
        self.audio_thread = None
        self.keyboard_thread = None
        self.listener = None

        # CSV metadata init
        if not os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=self.metadata_fields)
                    writer.writeheader()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to initialize metadata file: {e}")

    def build_duration_selector(self, parent):
        frame = tk.Frame(parent)
        tk.Label(frame, text="Recording Duration:").pack(side=tk.LEFT)
        options = ["30", "60", "300", "Custom"]
        self.duration_menu = tk.OptionMenu(frame, self.duration_var, *options, command=self.on_duration_change)
        self.duration_menu.pack(side=tk.LEFT)
        self.custom_entry = tk.Entry(frame, width=5)
        self.custom_entry.pack(side=tk.LEFT)
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

    def start_recording(self):
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
        self.remaining_time = self.recording_duration
        self.update_timer_label()
        self.is_recording = True
        self.timer_running = True
        self.root.after(1000, self.countdown_timer)
        self.status_label.config(text="Status: Recording...")
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.audio_thread = threading.Thread(target=self.record_audio, daemon=True)
        self.audio_thread.start()
        self.keyboard_thread = threading.Thread(target=self.listen_keyboard, daemon=True)
        self.keyboard_thread.start()

    def stop_recording(self):
        self.is_recording = False
        self.timer_running = False
        self.timer_label.config(text="")
        self.status_label.config(text="Status: Idle")
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        if self.listener:
            try:
                self.listener.stop()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to stop keyboard listener: {e}")

    def countdown_timer(self):
        if not self.is_recording or not self.timer_running:
            return
        if self.remaining_time > 0:
            self.remaining_time -= 1
            self.update_timer_label()
            self.root.after(1000, self.countdown_timer)
        else:
            self.stop_recording()

    def update_timer_label(self):
        if self.remaining_time > 0:
            self.timer_label.config(text=f"Time left: {self.remaining_time} s")
        else:
            self.timer_label.config(text="")

    def record_audio(self):
        """
        Continuously record audio into a rolling buffer.
        Uses a callback to update the buffer with new audio frames.
        """
        def callback(indata, frames, time_info, status):
            if not self.is_recording:
                return
            with self.buffer_lock:
                # Rolling buffer: shift left and append new frames
                self.audio_buffer = np.roll(self.audio_buffer, -frames)
                self.audio_buffer[-frames:] = indata[:, 0]
        try:
            if self.input_device is None:
                self.root.after(0, self.status_label.config, {"text": "Audio error: No valid input device found"})
                return
            with sd.InputStream(device=self.input_device, channels=1, samplerate=self.fs, callback=callback):
                while self.is_recording:
                    time.sleep(0.05)
        except Exception as e:
            self.root.after(0, self.status_label.config, {"text": f"Audio error: {e}"})
            return

    def listen_keyboard(self):
        """
        Listen for global keyboard events.
        On key press, extract audio segment and save with metadata.
        """
        def on_press(key):
            if not self.is_recording:
                return
            try:
                k = key.char if hasattr(key, 'char') and key.char else str(key)
            except Exception:
                k = str(key)
            # Only record printable keys and selected controls
            if (k.isprintable() and len(k) == 1) or k in ['Key.space', 'Key.enter', 'Key.backspace']:
                self.save_key_audio(k)
        try:
            self.listener = keyboard.Listener(on_press=on_press)
            self.listener.start()
            self.listener.join()
        except Exception as e:
            self.root.after(0, self.status_label.config, {"text": f"Keyboard error: {e}"})
            return

    def save_key_audio(self, key_label):
        """
        Extract audio segment from buffer, save as WAV, and log metadata.
        Ensures output is suitable for ML workflows (spectrograms, CNN, etc.).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        wav_filename = f"{key_label}_{timestamp}.wav"
        wav_path = os.path.join(self.output_dir, wav_filename)
        try:
            with self.buffer_lock:
                segment = self.audio_buffer[-self.segment_samples:]
            # Normalize segment for WAV output
            if np.max(np.abs(segment)) > 0:
                segment_int16 = np.int16(segment / np.max(np.abs(segment)) * 32767)
            else:
                segment_int16 = np.int16(segment)
            with wave.open(wav_path, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.fs)
                wf.writeframes(segment_int16.tobytes())
            # Write metadata for ML use
            with open(self.metadata_file, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.metadata_fields)
                writer.writerow({
                    "timestamp": timestamp,
                    "key": key_label,
                    "wav_file": wav_filename
                })
        except Exception as e:
            self.root.after(0, self.status_label.config, {"text": f"Save error: {e}"})
            return

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ResearchApp(root)
        root.mainloop()
    except KeyboardInterrupt:
        print("Application closed by user.")
