import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import numpy as np
import wave
import os
import threading
import time
from scipy import signal
from scipy.io import wavfile
import librosa
import librosa.display
import sounddevice as sd

class AudioAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Keyboard Audio Analyzer - Waveform & Spectrogram Viewer")
        self.root.geometry("1400x900")
        
        # Current audio data
        self.current_file = None
        self.audio_data = None
        self.sample_rate = None
        self.file_list = []
        self.current_index = 0
        
        # Playback state
        self.is_playing = False
        self.is_paused = False
        self.playback_position = 0  # in samples
        self.playback_thread = None
        self.playback_stream = None
        self.waveform_line = None  # Single line for waveform only
        
        # Build UI
        self.build_ui()
        
        # Bind keyboard shortcuts
        self.root.bind('<space>', lambda e: self.toggle_play_pause())
        self.root.bind('<Escape>', lambda e: self.stop_audio())
        self.root.bind('<Left>', lambda e: self.skip_backward())
        self.root.bind('<Right>', lambda e: self.skip_forward())

    def build_ui(self):
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top control panel
        control_frame = tk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # File controls
        file_frame = tk.LabelFrame(control_frame, text="File Controls", font=("Arial", 10, "bold"))
        file_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        btn_frame = tk.Frame(file_frame)
        btn_frame.pack(pady=5)
        
        self.load_file_btn = tk.Button(btn_frame, text="Load Audio File", command=self.load_single_file,
                                       bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
        self.load_file_btn.pack(side=tk.LEFT, padx=5)
        
        self.load_folder_btn = tk.Button(btn_frame, text="Load Folder", command=self.load_folder,
                                        bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
        self.load_folder_btn.pack(side=tk.LEFT, padx=5)
        
        self.prev_btn = tk.Button(btn_frame, text="← Previous", command=self.previous_file, state=tk.DISABLED)
        self.prev_btn.pack(side=tk.LEFT, padx=5)
        
        self.next_btn = tk.Button(btn_frame, text="Next →", command=self.next_file, state=tk.DISABLED)
        self.next_btn.pack(side=tk.LEFT, padx=5)
        
        # File info
        info_frame = tk.Frame(file_frame)
        info_frame.pack(pady=5)
        
        self.file_label = tk.Label(info_frame, text="No file loaded", font=("Arial", 9), fg="gray")
        self.file_label.pack()
        
        self.info_label = tk.Label(info_frame, text="", font=("Arial", 8), fg="blue")
        self.info_label.pack()
        
        # Playback controls
        playback_frame = tk.LabelFrame(control_frame, text="Playback Controls", font=("Arial", 10, "bold"))
        playback_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        btn_row1 = tk.Frame(playback_frame)
        btn_row1.pack(pady=2)
        
        self.play_btn = tk.Button(btn_row1, text="▶ Play", command=self.play_audio,
                                  bg="#4CAF50", fg="white", font=("Arial", 10, "bold"),
                                  state=tk.DISABLED, width=8)
        self.play_btn.pack(side=tk.LEFT, padx=2)
        
        self.pause_btn = tk.Button(btn_row1, text="⏸ Pause", command=self.pause_audio,
                                   state=tk.DISABLED, width=8)
        self.pause_btn.pack(side=tk.LEFT, padx=2)
        
        self.stop_btn = tk.Button(btn_row1, text="⏹ Stop", command=self.stop_audio,
                                 state=tk.DISABLED, width=8)
        self.stop_btn.pack(side=tk.LEFT, padx=2)
        
        btn_row2 = tk.Frame(playback_frame)
        btn_row2.pack(pady=2)
        
        tk.Label(btn_row2, text="Volume:").pack(side=tk.LEFT, padx=2)
        self.volume_var = tk.DoubleVar(value=0.7)
        self.volume_slider = tk.Scale(btn_row2, from_=0, to=1, orient=tk.HORIZONTAL,
                                     resolution=0.1, variable=self.volume_var, length=100, showvalue=True)
        self.volume_slider.pack(side=tk.LEFT, padx=2)
        
        # Performance option
        self.show_playback_line_var = tk.BooleanVar(value=True)
        tk.Checkbutton(btn_row2, text="Show Position Line", 
                      variable=self.show_playback_line_var).pack(side=tk.LEFT, padx=10)
        
        # Playback position label
        self.playback_label = tk.Label(playback_frame, text="00:00.000 / 00:00.000",
                                      font=("Courier", 9), fg="blue")
        self.playback_label.pack(pady=2)
        
        # Seekable progress bar
        seek_frame = tk.Frame(playback_frame)
        seek_frame.pack(fill=tk.X, pady=2)
        
        tk.Label(seek_frame, text="Position:").pack(side=tk.LEFT, padx=2)
        self.seek_var = tk.DoubleVar(value=0)
        self.seek_slider = tk.Scale(seek_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                   resolution=0.1, variable=self.seek_var, length=150,
                                   showvalue=False, command=self.on_seek)
        self.seek_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.seek_slider.config(state=tk.DISABLED)
        
        # Analysis options
        options_frame = tk.LabelFrame(control_frame, text="Analysis Options", font=("Arial", 10, "bold"))
        options_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        self.show_waveform_var = tk.BooleanVar(value=True)
        self.show_spectrogram_var = tk.BooleanVar(value=True)
        self.show_mel_spectrogram_var = tk.BooleanVar(value=True)
        self.show_energy_var = tk.BooleanVar(value=True)
        self.show_mfcc_var = tk.BooleanVar(value=False)
        
        tk.Checkbutton(options_frame, text="Waveform", variable=self.show_waveform_var,
                      command=self.update_plots).pack(anchor=tk.W, padx=5)
        tk.Checkbutton(options_frame, text="Spectrogram (STFT)", variable=self.show_spectrogram_var,
                      command=self.update_plots).pack(anchor=tk.W, padx=5)
        tk.Checkbutton(options_frame, text="Mel Spectrogram", variable=self.show_mel_spectrogram_var,
                      command=self.update_plots).pack(anchor=tk.W, padx=5)
        tk.Checkbutton(options_frame, text="Energy/Onset", variable=self.show_energy_var,
                      command=self.update_plots).pack(anchor=tk.W, padx=5)
        tk.Checkbutton(options_frame, text="MFCC Heatmap", variable=self.show_mfcc_var,
                      command=self.update_plots).pack(anchor=tk.W, padx=5)
        
        # Analysis parameters
        params_frame = tk.LabelFrame(control_frame, text="Parameters", font=("Arial", 10, "bold"))
        params_frame.pack(side=tk.LEFT, fill=tk.X, padx=5)
        
        tk.Label(params_frame, text="FFT Size:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.fft_size_var = tk.StringVar(value="2048")
        fft_combo = ttk.Combobox(params_frame, textvariable=self.fft_size_var, width=8,
                                values=["512", "1024", "2048", "4096"])
        fft_combo.grid(row=0, column=1, padx=5, pady=2)
        fft_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plots())
        
        tk.Label(params_frame, text="Window:").grid(row=1, column=0, sticky=tk.W, padx=5)
        self.window_var = tk.StringVar(value="hann")
        window_combo = ttk.Combobox(params_frame, textvariable=self.window_var, width=8,
                                   values=["hann", "hamming", "blackman"])
        window_combo.grid(row=1, column=1, padx=5, pady=2)
        window_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plots())
        
        # Statistics panel
        stats_frame = tk.LabelFrame(main_frame, text="Audio Statistics", font=("Arial", 10, "bold"))
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.stats_text = tk.Text(stats_frame, height=4, font=("Courier", 9))
        self.stats_text.pack(fill=tk.X, padx=5, pady=5)
        
        # Keyboard shortcuts help
        shortcuts_frame = tk.Frame(main_frame)
        shortcuts_frame.pack(fill=tk.X, pady=(5, 5))
        
        shortcuts_text = "⌨️ Keyboard Shortcuts: [SPACE] Play/Pause | [ESC] Stop | [←] Back 0.1s | [→] Forward 0.1s"
        shortcuts_label = tk.Label(shortcuts_frame, text=shortcuts_text, font=("Arial", 8, "bold"),
                                  fg="#1976D2", justify=tk.CENTER)
        shortcuts_label.pack()
        
        # Plots area with notebook (tabs)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Main analysis
        self.main_tab = tk.Frame(self.notebook)
        self.notebook.add(self.main_tab, text="Main Analysis")
        
        # Tab 2: Detailed spectrogram
        self.detail_tab = tk.Frame(self.notebook)
        self.notebook.add(self.detail_tab, text="Detailed Spectrogram")
        
        # Tab 3: Comparative view (for folder)
        self.compare_tab = tk.Frame(self.notebook)
        self.notebook.add(self.compare_tab, text="Comparison View")
        
        # Create matplotlib figures
        self.create_main_plots()
        self.create_detail_plots()
        self.create_comparison_plots()

    def create_main_plots(self):
        """Create main analysis plots"""
        self.main_fig = Figure(figsize=(12, 8))
        self.main_canvas = FigureCanvasTkAgg(self.main_fig, master=self.main_tab)
        self.main_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add toolbar
        toolbar = NavigationToolbar2Tk(self.main_canvas, self.main_tab)
        toolbar.update()

    def create_detail_plots(self):
        """Create detailed spectrogram plots"""
        self.detail_fig = Figure(figsize=(12, 8))
        self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=self.detail_tab)
        self.detail_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.detail_canvas, self.detail_tab)
        toolbar.update()

    def create_comparison_plots(self):
        """Create comparison plots for multiple files"""
        self.compare_fig = Figure(figsize=(12, 8))
        self.compare_canvas = FigureCanvasTkAgg(self.compare_fig, master=self.compare_tab)
        self.compare_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        toolbar = NavigationToolbar2Tk(self.compare_canvas, self.compare_tab)
        toolbar.update()

    def load_single_file(self):
        """Load a single audio file"""
        filepath = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        if filepath:
            self.file_list = [filepath]
            self.current_index = 0
            self.load_audio(filepath)
            self.prev_btn.config(state=tk.DISABLED)
            self.next_btn.config(state=tk.DISABLED)

    def load_folder(self):
        """Load all WAV files from a folder"""
        folder = filedialog.askdirectory(title="Select Folder with WAV Files")
        if folder:
            self.file_list = [os.path.join(folder, f) for f in os.listdir(folder) 
                            if f.lower().endswith('.wav')]
            if not self.file_list:
                messagebox.showwarning("No Files", "No WAV files found in the selected folder.")
                return
            
            self.file_list.sort()
            self.current_index = 0
            self.load_audio(self.file_list[0])
            self.prev_btn.config(state=tk.NORMAL if len(self.file_list) > 1 else tk.DISABLED)
            self.next_btn.config(state=tk.NORMAL if len(self.file_list) > 1 else tk.DISABLED)
            
            # Update comparison view
            self.update_comparison_view()

    def previous_file(self):
        """Load previous file in list"""
        if self.current_index > 0:
            self.current_index -= 1
            self.load_audio(self.file_list[self.current_index])

    def next_file(self):
        """Load next file in list"""
        if self.current_index < len(self.file_list) - 1:
            self.current_index += 1
            self.load_audio(self.file_list[self.current_index])

    def load_audio(self, filepath):
        """Load audio file and analyze it"""
        # Stop any current playback
        if self.is_playing or self.is_paused:
            self.stop_audio()
        
        try:
            # Read WAV file
            with wave.open(filepath, 'r') as wav_file:
                self.sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(n_frames)
                
                # Convert to numpy array - keep original format
                if wav_file.getsampwidth() == 2:
                    self.audio_data = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                    self.audio_data = self.audio_data / 32768.0  # Normalize to [-1, 1]
                else:
                    self.audio_data = np.frombuffer(audio_bytes, dtype=np.float32)
            
            self.current_file = filepath
            
            # Update UI
            filename = os.path.basename(filepath)
            self.file_label.config(text=f"File: {filename}")
            duration = len(self.audio_data) / self.sample_rate
            self.info_label.config(
                text=f"Sample Rate: {self.sample_rate} Hz | Duration: {duration:.3f} s | "
                     f"Samples: {len(self.audio_data)} | File {self.current_index + 1}/{len(self.file_list)}"
            )
            
            # Enable playback controls
            self.play_btn.config(state=tk.NORMAL)
            self.seek_slider.config(state=tk.NORMAL)
            self.playback_position = 0
            self.seek_var.set(0)
            self.playback_label.config(text=f"00:00.000 / {self.format_time(duration)}")
            
            # Update all plots
            self.update_plots()
            self.update_statistics()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio file:\n{e}")

    def update_plots(self):
        """Update all visualization plots"""
        if self.audio_data is None:
            return
        
        # Clear and update main plots
        self.main_fig.clear()
        
        # Count active plots
        n_plots = sum([
            self.show_waveform_var.get(),
            self.show_spectrogram_var.get(),
            self.show_mel_spectrogram_var.get(),
            self.show_energy_var.get(),
            self.show_mfcc_var.get()
        ])
        
        if n_plots == 0:
            self.main_canvas.draw()
            return
        
        plot_idx = 1
        
        # Time axis
        time_axis = np.arange(len(self.audio_data)) / self.sample_rate
        
        # Store waveform axis reference
        self.waveform_ax = None
        
        # 1. Waveform
        if self.show_waveform_var.get():
            ax = self.main_fig.add_subplot(n_plots, 1, plot_idx)
            ax.plot(time_axis, self.audio_data, linewidth=0.5, color='blue')
            ax.set_ylabel('Amplitude')
            ax.set_title('Waveform', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, time_axis[-1])
            self.waveform_ax = ax  # Store reference for playback line
            plot_idx += 1
        
        # 2. Spectrogram
        if self.show_spectrogram_var.get():
            ax = self.main_fig.add_subplot(n_plots, 1, plot_idx)
            fft_size = int(self.fft_size_var.get())
            hop_length = fft_size // 4
            
            f, t, Sxx = signal.spectrogram(
                self.audio_data,
                self.sample_rate,
                window=self.window_var.get(),
                nperseg=fft_size,
                noverlap=fft_size - hop_length
            )
            
            # Convert to dB
            Sxx_db = 10 * np.log10(Sxx + 1e-10)
            pcm = ax.pcolormesh(t, f, Sxx_db, shading='gouraud', cmap='viridis')
            ax.set_ylabel('Frequency (Hz)')
            ax.set_title('STFT Spectrogram (dB)', fontweight='bold')
            self.main_fig.colorbar(pcm, ax=ax, label='Power (dB)')
            ax.set_ylim(0, self.sample_rate / 2)
            plot_idx += 1
        
        # 3. Mel Spectrogram
        if self.show_mel_spectrogram_var.get():
            ax = self.main_fig.add_subplot(n_plots, 1, plot_idx)
            fft_size = int(self.fft_size_var.get())
            
            # Calculate Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=self.audio_data,
                sr=self.sample_rate,
                n_fft=fft_size,
                hop_length=fft_size // 4,
                n_mels=128,
                fmin=0,
                fmax=self.sample_rate / 2
            )
            
            # Convert to dB
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            img = librosa.display.specshow(
                mel_spec_db,
                x_axis='time',
                y_axis='mel',
                sr=self.sample_rate,
                ax=ax,
                cmap='magma'
            )
            ax.set_ylabel('Mel Frequency')
            ax.set_title('Mel Spectrogram (dB) - Perceptually Scaled', fontweight='bold')
            self.main_fig.colorbar(img, ax=ax, label='Power (dB)', format='%+2.0f dB')
            plot_idx += 1
        
        # 4. Energy and Onset Detection
        if self.show_energy_var.get():
            ax = self.main_fig.add_subplot(n_plots, 1, plot_idx)
            
            # Calculate energy envelope
            hop_length = 512
            energy = librosa.feature.rms(y=self.audio_data, hop_length=hop_length)[0]
            energy_times = librosa.frames_to_time(np.arange(len(energy)), sr=self.sample_rate, hop_length=hop_length)
            
            ax.plot(energy_times, energy, label='RMS Energy', color='green', linewidth=2)
            
            # Onset detection
            onset_env = librosa.onset.onset_strength(y=self.audio_data, sr=self.sample_rate)
            onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=self.sample_rate)
            onsets = librosa.onset.onset_detect(y=self.audio_data, sr=self.sample_rate)
            onset_times_detected = librosa.frames_to_time(onsets, sr=self.sample_rate)
            
            ax2 = ax.twinx()
            ax2.plot(onset_times, onset_env, label='Onset Strength', color='orange', alpha=0.7, linewidth=1.5)
            ax2.vlines(onset_times_detected, 0, onset_env.max(), color='red', alpha=0.8, 
                      linestyle='--', linewidth=2, label='Detected Onsets')
            
            ax.set_ylabel('RMS Energy', color='green')
            ax2.set_ylabel('Onset Strength', color='orange')
            ax.set_xlabel('Time (s)')
            ax.set_title('Energy Envelope and Onset Detection', fontweight='bold')
            ax.legend(loc='upper left')
            ax2.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            plot_idx += 1
        
        # 5. MFCC Heatmap
        if self.show_mfcc_var.get():
            ax = self.main_fig.add_subplot(n_plots, 1, plot_idx)
            
            # Calculate MFCCs
            mfccs = librosa.feature.mfcc(y=self.audio_data, sr=self.sample_rate, n_mfcc=13)
            img = librosa.display.specshow(mfccs, x_axis='time', sr=self.sample_rate, ax=ax, cmap='coolwarm')
            ax.set_ylabel('MFCC Coefficients')
            ax.set_title('MFCC Heatmap', fontweight='bold')
            self.main_fig.colorbar(img, ax=ax, label='MFCC Value')
            plot_idx += 1
        
        self.main_fig.tight_layout()
        self.main_canvas.draw()
        
        # Update detailed view
        self.update_detail_view()

    def update_detail_view(self):
        """Update detailed spectrogram view"""
        if self.audio_data is None:
            return
        
        self.detail_fig.clear()
        
        # Create multiple spectrogram views
        fft_size = int(self.fft_size_var.get())
        
        # 1. Linear frequency spectrogram
        ax1 = self.detail_fig.add_subplot(2, 2, 1)
        D = librosa.stft(self.audio_data, n_fft=fft_size)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', 
                                       sr=self.sample_rate, ax=ax1, cmap='magma')
        ax1.set_title('Linear Frequency Spectrogram', fontweight='bold')
        self.detail_fig.colorbar(img1, ax=ax1, format='%+2.0f dB')
        
        # 2. Log frequency spectrogram
        ax2 = self.detail_fig.add_subplot(2, 2, 2)
        img2 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', 
                                       sr=self.sample_rate, ax=ax2, cmap='magma')
        ax2.set_title('Log Frequency Spectrogram', fontweight='bold')
        self.detail_fig.colorbar(img2, ax=ax2, format='%+2.0f dB')
        
        # 3. Mel spectrogram
        ax3 = self.detail_fig.add_subplot(2, 2, 3)
        mel_spec = librosa.feature.melspectrogram(y=self.audio_data, sr=self.sample_rate, 
                                                 n_fft=fft_size, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        img3 = librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel', 
                                       sr=self.sample_rate, ax=ax3, cmap='viridis')
        ax3.set_title('Mel Spectrogram', fontweight='bold')
        self.detail_fig.colorbar(img3, ax=ax3, format='%+2.0f dB')
        
        # 4. Chromagram
        ax4 = self.detail_fig.add_subplot(2, 2, 4)
        chroma = librosa.feature.chroma_stft(y=self.audio_data, sr=self.sample_rate)
        img4 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', 
                                       sr=self.sample_rate, ax=ax4, cmap='coolwarm')
        ax4.set_title('Chromagram', fontweight='bold')
        self.detail_fig.colorbar(img4, ax=ax4)
        
        self.detail_fig.tight_layout()
        self.detail_canvas.draw()

    def update_comparison_view(self):
        """Update comparison view for multiple files"""
        if not self.file_list or len(self.file_list) < 2:
            return
        
        self.compare_fig.clear()
        
        # Limit to first 16 files for visualization
        files_to_compare = self.file_list[:16]
        n_files = len(files_to_compare)
        
        # Calculate grid size
        n_cols = min(4, n_files)
        n_rows = (n_files + n_cols - 1) // n_cols
        
        for idx, filepath in enumerate(files_to_compare):
            try:
                # Load audio
                with wave.open(filepath, 'r') as wav_file:
                    sr = wav_file.getframerate()
                    n_frames = wav_file.getnframes()
                    audio_bytes = wav_file.readframes(n_frames)
                    
                    if wav_file.getsampwidth() == 2:
                        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32)
                        audio = audio / 32768.0
                    else:
                        audio = np.frombuffer(audio_bytes, dtype=np.float32)
                
                # Create subplot
                ax = self.compare_fig.add_subplot(n_rows, n_cols, idx + 1)
                
                # Calculate and plot spectrogram
                mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=64)
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                librosa.display.specshow(mel_spec_db, x_axis='time', y_axis='mel',
                                       sr=sr, ax=ax, cmap='viridis')
                
                filename = os.path.basename(filepath)
                # Extract key from filename
                key_name = filename.split('_')[0] if '_' in filename else filename[:10]
                ax.set_title(f'{key_name}', fontsize=8)
                ax.set_xlabel('')
                ax.set_ylabel('')
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
        
        self.compare_fig.suptitle(f'Comparison of {n_files} Audio Files (Mel Spectrograms)', 
                                 fontweight='bold', fontsize=12)
        self.compare_fig.tight_layout()
        self.compare_canvas.draw()

    def update_statistics(self):
        """Calculate and display audio statistics"""
        if self.audio_data is None:
            return
        
        # Calculate statistics
        duration = len(self.audio_data) / self.sample_rate
        rms = np.sqrt(np.mean(self.audio_data ** 2))
        peak = np.max(np.abs(self.audio_data))
        zero_crossings = np.sum(np.diff(np.sign(self.audio_data)) != 0)
        zcr = zero_crossings / len(self.audio_data)
        
        # Spectral features
        spec_centroid = librosa.feature.spectral_centroid(y=self.audio_data, sr=self.sample_rate)[0]
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=self.audio_data, sr=self.sample_rate)[0]
        
        # Energy
        energy = np.sum(self.audio_data ** 2)
        
        # Format statistics
        stats_text = f"""
Duration: {duration:.4f} s | Sample Rate: {self.sample_rate} Hz | Samples: {len(self.audio_data)}
RMS Energy: {rms:.6f} | Peak Amplitude: {peak:.6f} | Total Energy: {energy:.6f}
Zero Crossing Rate: {zcr:.6f} | Spectral Centroid: {np.mean(spec_centroid):.2f} Hz | Bandwidth: {np.mean(spec_bandwidth):.2f} Hz
        """
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)

    def play_audio(self):
        """Start or resume audio playback"""
        if self.audio_data is None:
            return
        
        if self.is_paused:
            # Resume from pause
            self.is_paused = False
            self.is_playing = True
            self.play_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
        else:
            # Start from beginning or current position
            self.is_playing = True
            self.is_paused = False
            self.play_btn.config(state=tk.DISABLED)
            self.pause_btn.config(state=tk.NORMAL)
            self.stop_btn.config(state=tk.NORMAL)
        
        # Start playback thread
        self.playback_thread = threading.Thread(target=self.run_playback, daemon=True)
        self.playback_thread.start()
        
        # Start position update
        self.update_playback_position()

    def pause_audio(self):
        """Pause audio playback"""
        if self.is_playing:
            self.is_playing = False
            self.is_paused = True
            self.play_btn.config(state=tk.NORMAL)
            self.pause_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

    def stop_audio(self):
        """Stop audio playback and reset position"""
        self.is_playing = False
        self.is_paused = False
        self.playback_position = 0
        
        self.play_btn.config(state=tk.NORMAL)
        self.pause_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.DISABLED)
        
        # Remove playback line
        self.remove_playback_line()
        
        # Update time display
        if self.audio_data is not None:
            duration = len(self.audio_data) / self.sample_rate
            self.playback_label.config(text=f"00:00.000 / {self.format_time(duration)}")
            self.seek_var.set(0)

    def run_playback(self):
        """Run audio playback in separate thread - OPTIMIZED"""
        try:
            # Use optimal block size for smooth playback
            block_size = 4096  # Larger block for smoother playback
            
            def callback(outdata, frames, time_info, status):
                if status:
                    print(f"Playback status: {status}")
                
                if not self.is_playing or self.playback_position >= len(self.audio_data):
                    outdata[:] = 0
                    raise sd.CallbackStop()
                
                # Get audio chunk
                end_pos = min(self.playback_position + frames, len(self.audio_data))
                chunk = self.audio_data[self.playback_position:end_pos]
                
                # Apply volume
                chunk = chunk * self.volume_var.get()
                
                # Fill output buffer
                if len(chunk) < frames:
                    outdata[:len(chunk), 0] = chunk
                    outdata[len(chunk):, 0] = 0
                else:
                    outdata[:, 0] = chunk
                
                # Update position
                self.playback_position = end_pos
            
            # Start playback stream with OPTIMIZED settings
            with sd.OutputStream(
                channels=1,
                samplerate=self.sample_rate,
                callback=callback,
                blocksize=block_size,
                latency='low',  # Low latency for responsive playback
                prime_output_buffers_using_stream_callback=True  # Better buffer management
            ):
                while self.is_playing and self.playback_position < len(self.audio_data):
                    sd.sleep(50)  # Optimal sleep time
            
            # Playback finished
            if self.playback_position >= len(self.audio_data):
                self.root.after(0, self.stop_audio)
                
        except Exception as e:
            print(f"Playback error: {e}")
            self.root.after(0, self.stop_audio)

    def update_playback_position(self):
        """Update playback position indicator and time display"""
        if not self.is_playing and not self.is_paused:
            return
        
        # Update time display
        current_time = self.playback_position / self.sample_rate
        total_time = len(self.audio_data) / self.sample_rate
        time_text = f"{self.format_time(current_time)} / {self.format_time(total_time)}"
        self.playback_label.config(text=time_text)
        
        # Update seek slider
        if total_time > 0:
            progress = (current_time / total_time) * 100
            self.seek_var.set(progress)
        
        # Update playback line on WAVEFORM ONLY (only if enabled)
        if self.show_playback_line_var.get() and self.waveform_ax is not None:
            self.update_playback_line(current_time)
        
        # Schedule next update - 30 FPS for smooth visual updates
        if self.is_playing or self.is_paused:
            self.root.after(33, self.update_playback_position)

    def on_seek(self, value):
        """Handle seek slider movement"""
        if self.audio_data is None:
            return
        
        # Allow seeking only when paused or stopped
        if self.is_playing:
            return
        
        # Calculate new position
        progress = float(value) / 100.0
        new_position = int(progress * len(self.audio_data))
        self.playback_position = new_position
        
        # Update display
        current_time = self.playback_position / self.sample_rate
        total_time = len(self.audio_data) / self.sample_rate
        self.playback_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")
        
        # Update line position
        if self.show_playback_line_var.get() and self.waveform_ax is not None:
            self.update_playback_line(current_time)

    def update_playback_line(self, current_time):
        """Update the vertical line showing playback position ONLY on waveform"""
        # Remove old line
        if self.waveform_line is not None:
            try:
                self.waveform_line.remove()
            except:
                pass
        
        # Draw new line ONLY on waveform axis
        if self.waveform_ax is not None:
            self.waveform_line = self.waveform_ax.axvline(
                x=current_time, 
                color='red', 
                linewidth=2.5, 
                linestyle='-', 
                alpha=0.7, 
                zorder=10
            )
            
            # Use blit for faster updates
            try:
                self.main_canvas.draw_idle()
            except:
                pass

    def remove_playback_line(self):
        """Remove playback position line"""
        if self.waveform_line is not None:
            try:
                self.waveform_line.remove()
            except:
                pass
            self.waveform_line = None
            
            try:
                self.main_canvas.draw_idle()
            except:
                pass

    def format_time(self, seconds):
        """Format time as MM:SS.mmm"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"

    def toggle_play_pause(self):
        """Toggle between play and pause"""
        if self.audio_data is None:
            return
        
        if self.is_playing:
            self.pause_audio()
        else:
            self.play_audio()

    def skip_backward(self):
        """Skip backward 0.1 seconds"""
        if self.audio_data is None:
            return
        
        skip_samples = int(0.1 * self.sample_rate)
        self.playback_position = max(0, self.playback_position - skip_samples)
        
        # Update display
        current_time = self.playback_position / self.sample_rate
        total_time = len(self.audio_data) / self.sample_rate
        self.playback_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")
        
        if total_time > 0:
            progress = (current_time / total_time) * 100
            self.seek_var.set(progress)
        
        if self.show_playback_line_var.get() and self.waveform_ax is not None:
            self.update_playback_line(current_time)

    def skip_forward(self):
        """Skip forward 0.1 seconds"""
        if self.audio_data is None:
            return
        
        skip_samples = int(0.1 * self.sample_rate)
        self.playback_position = min(len(self.audio_data) - 1, self.playback_position + skip_samples)
        
        # Update display
        current_time = self.playback_position / self.sample_rate
        total_time = len(self.audio_data) / self.sample_rate
        self.playback_label.config(text=f"{self.format_time(current_time)} / {self.format_time(total_time)}")
        
        if total_time > 0:
            progress = (current_time / total_time) * 100
            self.seek_var.set(progress)
        
        if self.show_playback_line_var.get() and self.waveform_ax is not None:
            self.update_playback_line(current_time)


if __name__ == "__main__":
    root = tk.Tk()
    app = AudioAnalyzer(root)
    root.mainloop()