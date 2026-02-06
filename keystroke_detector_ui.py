"""
Keystroke Detection UI - Reimplemented from phone.ipynb
Exact implementation matching the CoAtNet model architecture and preprocessing pipeline
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
import threading
import time
import os
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from collections import deque
import math
from torchvision.ops import SqueezeExcitation
from torchvision.transforms import Compose, ToTensor


# ============================================================================
# MODEL ARCHITECTURE (from phone.ipynb)
# ============================================================================

class Stem(nn.Sequential):
    def __init__(self, out_channels):
        super().__init__(
            nn.Conv2d(1, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3)
        )


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super().__init__()
        self.mb_conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.GELU(),
            nn.Conv2d(in_channels * expansion_factor, in_channels * expansion_factor, 
                     kernel_size=3, padding=1, groups=in_channels * expansion_factor),
            nn.BatchNorm2d(in_channels * expansion_factor),
            nn.GELU(),
            SqueezeExcitation(in_channels * expansion_factor, in_channels, activation=nn.GELU),
            nn.Conv2d(in_channels * expansion_factor, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels)
        )
    
    def forward(self, x):
        return x + self.mb_conv(x)


class DownsamplingMBConv(MBConv):
    def __init__(self, in_channels, out_channels, expansion_factor=4):
        super().__init__(in_channels, out_channels, expansion_factor=4)
        self.mb_conv[1] = nn.Conv2d(in_channels, in_channels * expansion_factor, kernel_size=1, stride=2)
        self.channel_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.channel_projection(self.pool(x)) + self.mb_conv(x)


class RelativeAttention2d(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, heads=8, head_size=32):
        super().__init__()
        heads = out_channels // head_size
        self.heads = heads
        self.head_size = head_size
        self.image_size = image_size
        self.head_dim = heads * head_size
        self.attend = nn.Softmax(dim=-2)
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.to_q = nn.Linear(in_channels, self.head_dim)
        self.to_k = nn.Linear(in_channels, self.head_dim)
        self.to_v = nn.Linear(in_channels, self.head_dim)
        self.to_output = nn.Sequential(
            nn.Linear(self.head_dim, out_channels),
            nn.Dropout(0.3)
        )
        self.normalization = nn.LayerNorm(in_channels)
        
        self.relative_bias = nn.Parameter(torch.randn(heads, (2 * image_size - 1) * (2 * image_size - 1)))
        self.register_buffer("relative_indices", self.get_indices(image_size, image_size))
        self.precomputed_relative_bias = None
    
    def norm(self, x):
        x = x.transpose(1, -1)
        x = self.normalization(x)
        x = x.transpose(-1, 1)
        return x
    
    def get_relative_biases(self):
        if not self.training:
            return self.precomputed_relative_bias
        indices = self.relative_indices.expand(self.heads, -1)
        rel_pos_enc = self.relative_bias.gather(-1, indices)
        rel_pos_enc = rel_pos_enc.unflatten(-1, (self.image_size * self.image_size, self.image_size * self.image_size))
        return rel_pos_enc
    
    def reshape_for_linear(self, x):
        b, _, _, _ = x.shape
        return x.reshape(b, self.image_size * self.image_size, self.in_channels)
    
    def attention_score(self, x):
        b, _, h, _ = x.shape
        q = self.to_q(self.reshape_for_linear(x)).view(b, self.heads, self.head_size, -1)
        k = self.to_k(self.reshape_for_linear(x)).view(b, self.heads, self.head_size, -1)
        dots = torch.matmul(k.transpose(-1, -2), q) / math.sqrt(self.head_dim)
        relative_biases_indexed = self.get_relative_biases()
        return self.attend(dots + relative_biases_indexed)
    
    def relative_attention(self, x):
        b, _, _, _ = x.shape
        v = self.to_v(self.reshape_for_linear(x)).view(b, self.heads, self.head_size, -1)
        out = torch.matmul(v, self.attention_score(x))
        out = out.view(b, self.image_size, self.image_size, -1)
        return self.to_output(out).view(b, self.out_channels, self.image_size, self.image_size)
        
    def forward(self, x):
        return x + self.relative_attention(self.norm(x))
    
    def train(self, training):
        if not training:
            self.precomputed_relative_bias = self.get_relative_biases()
        super().train(training)
        
    @staticmethod
    def get_indices(h, w):
        y = torch.arange(h, dtype=torch.long)
        x = torch.arange(w, dtype=torch.long)
        
        y1, x1, y2, x2 = torch.meshgrid(y, x, y, x, indexing='ij')
        indices = (y1 - y2 + h - 1) * (2 * w - 1) + x1 - x2 + w - 1
        indices = indices.flatten()
        
        return indices


class DownsamplingRelativeAttention2d(RelativeAttention2d):
    def __init__(self, in_channels, out_channels, image_size, heads=8, head_size=32):
        super().__init__(in_channels, out_channels, image_size, heads=8, head_size=32)
        self.channel_projection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.normalization = nn.LayerNorm(in_channels)

    def norm(self, x):
        x = x.transpose(1, -1)
        x = self.normalization(x)
        x = x.transpose(-1, 1)
        return x
        
    def forward(self, x):
        return self.channel_projection(self.pool(x)) + self.relative_attention(self.pool(self.norm(x)))


class FeedForwardNetwork(nn.Module):
    def __init__(self, out_channels, expansion_factor=4):
        super().__init__()
        hidden_dim = out_channels * expansion_factor
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, out_channels),
            nn.Dropout(0.3)
        )
        self.normalization = nn.LayerNorm(out_channels)
        self.out_channels = out_channels
    
    def norm(self, x):
        x = x.transpose(1, -1)
        x = self.normalization(x)
        x = x.transpose(-1, 1)
        return x
    
    def forward(self, x):
        old_shape = x.shape
        batch_size = old_shape[0]
        return x + torch.reshape(self.ffn(torch.reshape(self.norm(x), (batch_size, -1, self.out_channels))), old_shape)


class DownsampleTransformerBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, image_size):
        attention = DownsamplingRelativeAttention2d(in_channels, out_channels, image_size)
        ffn = FeedForwardNetwork(out_channels)
        super().__init__(attention, ffn)


class TransformerBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, image_size):
        attention = RelativeAttention2d(in_channels, out_channels, image_size)
        ffn = FeedForwardNetwork(out_channels)
        super().__init__(attention, ffn)


class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, num_classes)
        self.in_channels = in_channels
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.pool(x)
        x = torch.reshape(x, (batch_size, -1, self.in_channels))
        return torch.squeeze(self.fc(x))


class MyCoAtNet(nn.Sequential):
    def __init__(self, nums_blocks, layer_out_channels, num_classes=36):
        s0 = nn.Sequential(Stem(layer_out_channels[0]))
        
        s1 = [DownsamplingMBConv(layer_out_channels[0], layer_out_channels[1])]
        for i in range(nums_blocks[1] - 1):
            s1.append(MBConv(layer_out_channels[1], layer_out_channels[1]))
        s1 = nn.Sequential(*s1)
        
        s2 = [DownsamplingMBConv(layer_out_channels[1], layer_out_channels[2])]
        for i in range(nums_blocks[2] - 1):
            s2.append(MBConv(layer_out_channels[2], layer_out_channels[2]))
        s2 = nn.Sequential(*s2)
        
        s3 = [DownsampleTransformerBlock(layer_out_channels[2], layer_out_channels[3], 64 // 16)]
        for i in range(nums_blocks[3] - 1):
            s3.append(TransformerBlock(layer_out_channels[3], layer_out_channels[3], 64 // 16))
        s3 = nn.Sequential(*s3)
        
        s4 = [DownsampleTransformerBlock(layer_out_channels[3], layer_out_channels[4], 64 // 32)]
        for i in range(nums_blocks[4] - 1):
            s4.append(TransformerBlock(layer_out_channels[4], layer_out_channels[4], 64 // 32))
        s4 = nn.Sequential(*s4)
        
        head = Head(layer_out_channels[4], num_classes)
        
        super().__init__(s0, s1, s2, s3, s4, head)


# ============================================================================
# PREPROCESSING (from phone.ipynb)
# ============================================================================

class AudioPreprocessor:
    """Audio preprocessing exactly as in phone.ipynb"""
    
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        # Exact parameters from phone.ipynb
        self.to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate, 
            n_mels=64, 
            hop_length=300, 
            n_fft=2048, 
            win_length=1024
        )
        self.mel_spectrogram_to_numpy = lambda spectrogram: spectrogram.log2()[0,:,:].numpy()
        self.transforms = Compose([self.to_mel_spectrogram, self.mel_spectrogram_to_numpy, ToTensor()])
    
    def process_audio(self, waveform):
        """Process audio to mel spectrogram - returns (1, 64, 64) tensor"""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        
        # Ensure waveform is 2D (channels, samples)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        
        # Apply transforms
        mel_spec = self.transforms(waveform)
        
        return mel_spec


class AudioRecorder:
    """Real-time audio recording with keystroke detection"""
    
    def __init__(self, sample_rate=44100, device_id=None):
        self.sample_rate = sample_rate
        self.device_id = device_id
        self.stream = None
        self.is_recording = False
        self.audio_buffer = deque(maxlen=int(sample_rate * 5))  # 5 second buffer
        self.callback_func = None
        self.threshold = 0.06
        
    def get_available_devices(self):
        """Get list of available input devices"""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels']
                })
        return devices
    
    def set_audio_callback(self, callback):
        """Set callback function for audio chunks"""
        self.callback_func = callback
    
    def audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            print(f"Audio callback status: {status}")
        
        # Add to buffer
        audio_data = indata[:, 0].copy()
        self.audio_buffer.extend(audio_data)
        
        # Call user callback if set
        if self.callback_func:
            self.callback_func(audio_data)
    
    def start_recording(self):
        """Start audio recording"""
        try:
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                callback=self.audio_callback,
                blocksize=2048
            )
            self.stream.start()
            self.is_recording = True
            return True
        except Exception as e:
            print(f"Error starting recording: {e}")
            return False
    
    def stop_recording(self):
        """Stop audio recording"""
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False
    
    def detect_keystrokes(self):
        """Detect keystroke segments from audio buffer"""
        if len(self.audio_buffer) < self.sample_rate * 0.2:  # Need at least 200ms
            return []
        
        # Convert buffer to numpy array
        audio = np.array(self.audio_buffer)
        
        # Simple energy-based detection
        keystrokes = []
        window_size = int(self.sample_rate * 0.02)  # 20ms windows
        hop_size = int(window_size / 2)
        
        # Calculate energy
        energy = []
        for i in range(0, len(audio) - window_size, hop_size):
            window = audio[i:i+window_size]
            energy.append(np.sum(window ** 2))
        
        if not energy:
            return []
        
        energy = np.array(energy)
        threshold = np.max(energy) * self.threshold
        
        # Find peaks above threshold
        is_above = energy > threshold
        
        # Find start and end of events
        starts = []
        ends = []
        in_event = False
        
        for i, above in enumerate(is_above):
            if above and not in_event:
                starts.append(i * hop_size)
                in_event = True
            elif not above and in_event:
                ends.append(i * hop_size)
                in_event = False
        
        # Extract keystroke segments
        min_duration = int(self.sample_rate * 0.05)  # 50ms minimum
        max_duration = int(self.sample_rate * 0.5)   # 500ms maximum
        
        for start, end in zip(starts, ends):
            duration = end - start
            if min_duration <= duration <= max_duration:
                # Extract segment with some padding
                pad = int(self.sample_rate * 0.01)
                seg_start = max(0, start - pad)
                seg_end = min(len(audio), end + pad)
                segment = audio[seg_start:seg_end]
                
                # Ensure minimum length for processing
                if len(segment) >= min_duration:
                    keystrokes.append(segment)
        
        # Clear processed audio from buffer (keep last 1 second)
        keep_samples = int(self.sample_rate * 1.0)
        if len(self.audio_buffer) > keep_samples:
            # Convert to list, slice, convert back to deque
            buffer_list = list(self.audio_buffer)
            self.audio_buffer = deque(buffer_list[-keep_samples:], maxlen=self.audio_buffer.maxlen)
        
        return keystrokes
    
    def adjust_threshold(self, threshold):
        """Adjust detection threshold"""
        self.threshold = threshold


# ============================================================================
# UI APPLICATION
# ============================================================================

class KeystrokeDetectionUI:
    """Main UI for keystroke detection"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Keystroke Detection System - CoAtNet")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Model and processing
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        self.model_path = None
        
        # Audio components
        self.preprocessor = AudioPreprocessor(sample_rate=44100)
        self.recorder = AudioRecorder(sample_rate=44100)
        self.recorder.set_audio_callback(self._on_audio_chunk)
        
        # Recording state
        self.is_recording = False
        self.processing_thread = None
        self.stop_processing = False
        self.start_time = None
        
        # Key labels (0-9, a-z)
        digits = [str(digit) for digit in range(10)]
        alphabet = [chr(ascii_code) for ascii_code in range(ord('a'), ord('z') + 1)]
        self.key_labels = digits + alphabet
        
        # Statistics
        self.keystroke_count = 0
        self.detection_history = []
        
        # Debug options
        self.save_mel_specs = False
        self.debug_dir = Path("model/debug_output")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup UI
        self._create_widgets()
        self._log("Application started")
        self._log(f"Device: {self.device.upper()}")
        self._log(f"Sample rate: 44100 Hz (matching phone.ipynb)")
        
        # Auto-load model
        self._auto_load_model()
    
    def _create_widgets(self):
        """Create all UI widgets"""
        # Title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🎹 Keystroke Detection - CoAtNet",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        )
        title_label.pack(pady=15)
        
        # Main container
        main_container = tk.Frame(self.root, padx=20, pady=10)
        main_container.pack(fill=tk.BOTH, expand=True)
        
        # Sections
        self._create_model_section(main_container)
        self._create_recording_section(main_container)
        self._create_results_section(main_container)
        self._create_log_section(main_container)
        self._create_statistics_section(main_container)
        self._create_status_bar()
    
    def _create_model_section(self, parent):
        """Create model loading section"""
        model_frame = tk.LabelFrame(parent, text="Model Configuration", font=('Arial', 11, 'bold'), padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Model path
        path_frame = tk.Frame(model_frame)
        path_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(path_frame, text="Model Path:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.model_path_var = tk.StringVar()
        model_entry = tk.Entry(path_frame, textvariable=self.model_path_var, font=('Arial', 10), state='readonly')
        model_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        
        browse_btn = tk.Button(path_frame, text="Browse", command=self._browse_model, font=('Arial', 10))
        browse_btn.pack(side=tk.LEFT, padx=(0, 5))
        
        load_btn = tk.Button(path_frame, text="Load Model", command=self._load_model, font=('Arial', 10, 'bold'), bg='#3498db', fg='white')
        load_btn.pack(side=tk.LEFT)
        
        # Model status
        self.model_status_label = tk.Label(model_frame, text="⚠ No model loaded", font=('Arial', 10), fg='orange')
        self.model_status_label.pack(pady=5)
        
        # Device info
        device_label = tk.Label(model_frame, text=f"Device: {self.device.upper()}", font=('Arial', 9), fg='gray')
        device_label.pack()
    
    def _create_recording_section(self, parent):
        """Create recording control section"""
        recording_frame = tk.LabelFrame(parent, text="Recording Control", font=('Arial', 11, 'bold'), padx=10, pady=10)
        recording_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Microphone selection
        mic_frame = tk.Frame(recording_frame)
        mic_frame.pack(fill=tk.X, pady=(5, 10))
        
        tk.Label(mic_frame, text="Microphone:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(mic_frame, textvariable=self.mic_var, state='readonly', font=('Arial', 9), width=50)
        self.mic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.mic_combo.bind('<<ComboboxSelected>>', self._on_mic_selected)
        
        refresh_btn = tk.Button(mic_frame, text="🔄 Refresh", command=self._refresh_devices, font=('Arial', 9))
        refresh_btn.pack(side=tk.LEFT)
        
        self._refresh_devices()
        
        # Controls
        controls_frame = tk.Frame(recording_frame)
        controls_frame.pack(pady=10)
        
        self.record_btn = tk.Button(
            controls_frame,
            text="🎤 Start Recording",
            command=self._toggle_recording,
            font=('Arial', 12, 'bold'),
            bg='#27ae60',
            fg='white',
            width=20,
            height=2,
            state=tk.DISABLED
        )
        self.record_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = tk.Button(
            controls_frame,
            text="🗑 Clear Results",
            command=self._clear_results,
            font=('Arial', 10),
            width=15
        )
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        self.test_wav_btn = tk.Button(
            controls_frame,
            text="📁 Test WAV File",
            command=self._test_wav_file,
            font=('Arial', 10),
            bg='#9b59b6',
            fg='white',
            width=15,
            state=tk.DISABLED
        )
        self.test_wav_btn.pack(side=tk.LEFT, padx=10)
        
        # Debug options
        debug_frame = tk.Frame(recording_frame)
        debug_frame.pack(fill=tk.X, pady=5)
        
        self.save_mel_var = tk.BooleanVar(value=False)
        save_mel_check = tk.Checkbutton(
            debug_frame,
            text="Save audio & mel-spectrograms for debugging",
            variable=self.save_mel_var,
            font=('Arial', 9),
            command=self._toggle_save_mel
        )
        save_mel_check.pack(side=tk.LEFT)
        
        # Threshold adjustment
        threshold_frame = tk.Frame(recording_frame)
        threshold_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(threshold_frame, text="Detection Sensitivity:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))
        
        self.threshold_var = tk.DoubleVar(value=0.06)
        threshold_slider = tk.Scale(
            threshold_frame,
            from_=0.01,
            to=0.2,
            resolution=0.01,
            orient=tk.HORIZONTAL,
            variable=self.threshold_var,
            command=self._update_threshold,
            length=200
        )
        threshold_slider.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.threshold_label = tk.Label(threshold_frame, text="0.06", font=('Arial', 10))
        self.threshold_label.pack(side=tk.LEFT, padx=(10, 0))
    
    def _create_results_section(self, parent):
        """Create detection results section"""
        results_frame = tk.LabelFrame(parent, text="Detection Results", font=('Arial', 11, 'bold'), padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Current detection
        current_frame = tk.Frame(results_frame)
        current_frame.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(current_frame, text="Last Detected Key:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))
        
        self.current_key_label = tk.Label(
            current_frame,
            text="-",
            font=('Arial', 28, 'bold'),
            bg='#ecf0f1',
            fg='#2c3e50',
            width=3,
            relief=tk.RAISED,
            borderwidth=2
        )
        self.current_key_label.pack(side=tk.LEFT, padx=10)
        
        self.confidence_label = tk.Label(current_frame, text="Confidence: -", font=('Arial', 11))
        self.confidence_label.pack(side=tk.LEFT, padx=20)
        
        # Top predictions
        tk.Label(results_frame, text="Top 5 Predictions:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        predictions_container = tk.Frame(results_frame)
        predictions_container.pack(fill=tk.X)
        
        self.prediction_labels = []
        for i in range(5):
            pred_frame = tk.Frame(predictions_container, bg='#ecf0f1', relief=tk.GROOVE, borderwidth=1)
            pred_frame.pack(fill=tk.X, pady=2)
            
            rank_label = tk.Label(pred_frame, text=f"#{i+1}", font=('Arial', 9, 'bold'), bg='#ecf0f1', width=4)
            rank_label.pack(side=tk.LEFT, padx=5)
            
            key_label = tk.Label(pred_frame, text="-", font=('Arial', 10, 'bold'), bg='#ecf0f1', width=3)
            key_label.pack(side=tk.LEFT, padx=5)
            
            bar_canvas = tk.Canvas(pred_frame, width=200, height=20, bg='white', highlightthickness=0)
            bar_canvas.pack(side=tk.LEFT, padx=5)
            
            conf_label = tk.Label(pred_frame, text="0.00%", font=('Arial', 9), bg='#ecf0f1', width=8)
            conf_label.pack(side=tk.LEFT, padx=5)
            
            self.prediction_labels.append({
                'key': key_label,
                'bar': bar_canvas,
                'conf': conf_label
            })
        
        # Detection history
        tk.Label(results_frame, text="Detection History:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))
        
        history_container = tk.Frame(results_frame)
        history_container.pack(fill=tk.BOTH, expand=True)
        
        scrollbar = tk.Scrollbar(history_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.history_text = tk.Text(history_container, height=6, font=('Courier', 9), yscrollcommand=scrollbar.set)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_text.yview)
    
    def _create_statistics_section(self, parent):
        """Create statistics section"""
        stats_frame = tk.LabelFrame(parent, text="Statistics", font=('Arial', 11, 'bold'), padx=10, pady=10)
        stats_frame.pack(fill=tk.X)
        
        stats_container = tk.Frame(stats_frame)
        stats_container.pack(fill=tk.X)
        
        self.stats_labels = {}
        
        stats_data = [
            ('Total Keystrokes:', 'keystrokes'),
            ('Recording Time:', 'time'),
            ('Avg Confidence:', 'avg_conf')
        ]
        
        for i, (label_text, key) in enumerate(stats_data):
            frame = tk.Frame(stats_container)
            frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
            
            tk.Label(frame, text=label_text, font=('Arial', 9)).pack()
            value_label = tk.Label(frame, text="0", font=('Arial', 12, 'bold'), fg='#3498db')
            value_label.pack()
            
            self.stats_labels[key] = value_label
    
    def _create_log_section(self, parent):
        """Create log section"""
        log_frame = tk.LabelFrame(parent, text="System Log", font=('Arial', 11, 'bold'), padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        
        log_container = tk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)
        
        log_scrollbar = tk.Scrollbar(log_container)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.log_text = tk.Text(log_container, height=8, font=('Courier', 8), yscrollcommand=log_scrollbar.set, bg='#f8f9fa')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.log_text.yview)
        
        clear_log_btn = tk.Button(log_frame, text="Clear Log", command=self._clear_log, font=('Arial', 9))
        clear_log_btn.pack(pady=(5, 0))
    
    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            bd=1,
            relief=tk.SUNKEN,
            anchor=tk.W,
            font=('Arial', 9)
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _log(self, message, level="INFO"):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        
        try:
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            print(log_entry.strip())
        except:
            print(log_entry.strip())
    
    def _clear_log(self):
        """Clear log text"""
        self.log_text.delete('1.0', tk.END)
        self._log("Log cleared")
    
    def _refresh_devices(self):
        """Refresh available audio devices"""
        self._log("Refreshing audio devices...")
        try:
            devices = self.recorder.get_available_devices()
            device_names = [f"{d['id']}: {d['name']}" for d in devices]
            
            self.mic_combo['values'] = device_names
            
            if device_names:
                self.mic_combo.current(0)
                self.recorder.device_id = devices[0]['id']
                self._log(f"Found {len(device_names)} audio device(s)")
            else:
                self._log("No audio devices found!", "WARNING")
        except Exception as e:
            self._log(f"Error refreshing devices: {e}", "ERROR")
    
    def _on_mic_selected(self, event):
        """Handle microphone selection"""
        selection = self.mic_combo.current()
        if selection >= 0:
            devices = self.recorder.get_available_devices()
            self.recorder.device_id = devices[selection]['id']
            self._log(f"Microphone selected: {devices[selection]['name']}")
    
    def _on_audio_chunk(self, audio_chunk):
        """Handle incoming audio chunks"""
        pass  # Audio is already being added to recorder's buffer
    
    def _browse_model(self):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            initialdir=".",
            filetypes=[("PyTorch Model", "*.pkl *.pth *.pt"), ("All Files", "*.*")]
        )
        if filename:
            self.model_path_var.set(filename)
    
    def _auto_load_model(self):
        """Automatically load model if found"""
        self._log("Searching for model files...")
        
        # Look for model in current directory or common locations
        model_names = ["CoAtNet-1-Best-Phone.pkl", "CoAtNet-1-Phone.pkl"]
        search_paths = [".", "model", "../"]
        
        for search_path in search_paths:
            for model_name in model_names:
                model_path = Path(search_path) / model_name
                if model_path.exists():
                    self._log(f"Found model: {model_path}")
                    self.model_path_var.set(str(model_path))
                    self._load_model()
                    return
        
        self._log("No model found. Please load manually.", "WARNING")
    
    def _load_model(self):
        """Load the CoAtNet model"""
        model_path = self.model_path_var.get()
        
        if not model_path or not os.path.exists(model_path):
            self._log("Invalid model path", "ERROR")
            messagebox.showerror("Error", "Please select a valid model file")
            return
        
        try:
            self._log(f"Loading model from: {model_path}")
            self.status_bar.config(text="Loading model...")
            self.root.update()
            
            # Create CoAtNet-1 model (from phone.ipynb)
            self._log("Creating CoAtNet-1 model...")
            nums_blocks = [2, 2, 3, 5, 2]           # L
            channels = [64, 96, 192, 384, 768]      # D
            self.model = MyCoAtNet(nums_blocks, channels, num_classes=36)
            
            # Load weights
            self._log("Loading model weights...")
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            self._log(f"Model parameters: {total_params:,}")
            
            # Test model
            with torch.no_grad():
                test_input = torch.randn(1, 1, 64, 64).to(self.device)
                test_output = self.model(test_input)
                self._log(f"Model test output shape: {test_output.shape}")
            
            self.model_loaded = True
            self.model_status_label.config(text="✓ Model loaded successfully", fg='green')
            self.record_btn.config(state=tk.NORMAL)
            self.test_wav_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"Model loaded: {os.path.basename(model_path)}")
            
            self._log("Model loaded successfully!")
            messagebox.showinfo("Success", "Model loaded successfully!")
            
        except Exception as e:
            self._log(f"Failed to load model: {e}", "ERROR")
            import traceback
            self._log(traceback.format_exc(), "ERROR")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.model_loaded = False
            self.status_bar.config(text="Model loading failed")
    
    def _test_wav_file(self):
        """Test with a WAV file"""
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load a model first!")
            return
        
        file_path = filedialog.askopenfilename(
            title="Select WAV File",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")]
        )
        
        if not file_path:
            return
        
        self._log(f"Loading WAV file: {os.path.basename(file_path)}")
        
        try:
            # Load audio using soundfile (avoids torchcodec/FFmpeg dependency)
            audio_data, sr = sf.read(file_path, dtype='float32')
            self._log(f"Loaded: {audio_data.shape}, sample rate: {sr} Hz")
            
            # Convert to torch tensor
            if len(audio_data.shape) == 1:
                # Mono audio
                waveform = torch.from_numpy(audio_data).unsqueeze(0)
            else:
                # Stereo - convert to mono
                waveform = torch.from_numpy(audio_data.mean(axis=1)).unsqueeze(0)
                self._log("Converted to mono")
            
            # Resample if needed
            if sr != 44100:
                resampler = torchaudio.transforms.Resample(sr, 44100)
                waveform = resampler(waveform)
                self._log("Resampled to 44100 Hz")
            
            # Process
            self._log("Preprocessing audio...")
            mel_spec = self.preprocessor.process_audio(waveform)
            
            if mel_spec.shape != torch.Size([1, 64, 64]):
                self._log(f"WARNING: Unexpected mel-spec shape: {mel_spec.shape}", "WARNING")
                messagebox.showwarning("Warning", f"Audio might not be properly segmented.\nMel-spec shape: {mel_spec.shape}\nExpected: (1, 64, 64)")
            
            self._log(f"Mel-spectrogram shape: {mel_spec.shape}")
            
            # Save debug if enabled
            if self.save_mel_specs:
                self._save_debug_mel_spec(mel_spec, f"test_{os.path.basename(file_path)}")
            
            # Predict
            self._log("Running inference...")
            with torch.no_grad():
                input_tensor = mel_spec.unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                
                # Handle output shape
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                
                probabilities = F.softmax(output, dim=1)
                top5_probs, top5_indices = torch.topk(probabilities[0], 5)
            
            # Get results
            predicted_key = self.key_labels[top5_indices[0].item()]
            confidence = top5_probs[0].item() * 100
            
            self._log(f"Predicted: '{predicted_key}' (confidence: {confidence:.2f}%)")
            
            # Update UI
            self._update_detection_display(predicted_key, confidence, top5_indices, top5_probs)
            
            # Add to history
            self.history_text.insert(
                tk.END,
                f"[{datetime.now().strftime('%H:%M:%S')}] '{predicted_key}' ({confidence:.1f}%) - {os.path.basename(file_path)}\n"
            )
            self.history_text.see(tk.END)
            
            self.keystroke_count += 1
            self.detection_history.append(confidence)
            self._update_statistics()
            
        except Exception as e:
            self._log(f"Error processing WAV file: {e}", "ERROR")
            import traceback
            self._log(traceback.format_exc(), "ERROR")
            messagebox.showerror("Error", f"Failed to process WAV file:\n{e}")
    
    def _toggle_recording(self):
        """Toggle recording state"""
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()
    
    def _start_recording(self):
        """Start recording"""
        self._log("Starting recording...")
        
        if self.recorder.start_recording():
            self.is_recording = True
            self.stop_processing = False
            self.start_time = time.time()
            
            self.record_btn.config(text="⏹ Stop Recording", bg='#e74c3c')
            self.status_bar.config(text="Recording...")
            
            self._log("Recording started")
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
        else:
            self._log("Failed to start recording", "ERROR")
    
    def _stop_recording(self):
        """Stop recording"""
        self._log("Stopping recording...")
        self.stop_processing = True
        self.recorder.stop_recording()
        self.is_recording = False
        
        self.record_btn.config(text="🎤 Start Recording", bg='#27ae60')
        self.status_bar.config(text="Recording stopped")
        self._log("Recording stopped")
    
    def _processing_loop(self):
        """Background processing loop"""
        self._log("Processing loop started")
        last_process_time = time.time()
        
        while not self.stop_processing:
            current_time = time.time()
            
            if current_time - last_process_time >= 0.1:  # Process every 100ms
                # Detect keystrokes
                keystrokes = self.recorder.detect_keystrokes()
                
                for keystroke in keystrokes:
                    try:
                        self._log(f"Processing keystroke: {len(keystroke)} samples")
                        
                        # Preprocess
                        waveform = torch.from_numpy(keystroke).float()
                        mel_spec = self.preprocessor.process_audio(waveform)
                        
                        if mel_spec.shape != torch.Size([1, 64, 64]):
                            self._log(f"Skipping keystroke with wrong shape: {mel_spec.shape}", "WARNING")
                            continue
                        
                        # Save debug if enabled
                        if self.save_mel_specs:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            self._save_debug_mel_spec(mel_spec, f"keystroke_{timestamp}")
                            # Save audio
                            wav_path = self.debug_dir / f"keystroke_{timestamp}.wav"
                            sf.write(str(wav_path), keystroke, 44100)
                        
                        # Predict
                        with torch.no_grad():
                            input_tensor = mel_spec.unsqueeze(0).to(self.device)
                            output = self.model(input_tensor)
                            
                            if output.dim() == 1:
                                output = output.unsqueeze(0)
                            
                            probabilities = F.softmax(output, dim=1)
                            top5_probs, top5_indices = torch.topk(probabilities[0], 5)
                        
                        # Get results
                        predicted_key = self.key_labels[top5_indices[0].item()]
                        confidence = top5_probs[0].item() * 100
                        
                        self._log(f"Detected: '{predicted_key}' ({confidence:.1f}%)")
                        
                        # Update UI
                        self.root.after(0, self._update_detection_display, predicted_key, confidence, top5_indices, top5_probs)
                        self.root.after(0, self._add_to_history, predicted_key, confidence)
                        
                        self.keystroke_count += 1
                        self.detection_history.append(confidence)
                        self.root.after(0, self._update_statistics)
                        
                    except Exception as e:
                        self._log(f"Error processing keystroke: {e}", "ERROR")
                
                last_process_time = current_time
            
            # Update recording time
            if self.is_recording:
                elapsed = int(current_time - self.start_time)
                self.root.after(0, self._update_recording_time, elapsed)
            
            time.sleep(0.05)
    
    def _update_detection_display(self, predicted_key, confidence, top5_indices, top5_probs):
        """Update detection display"""
        # Current detection
        self.current_key_label.config(text=predicted_key.upper())
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        
        # Top 5 predictions
        for i in range(5):
            key = self.key_labels[top5_indices[i].item()]
            conf = top5_probs[i].item()
            
            self.prediction_labels[i]['key'].config(text=key.upper())
            self.prediction_labels[i]['conf'].config(text=f"{conf*100:.2f}%")
            
            # Draw confidence bar
            canvas = self.prediction_labels[i]['bar']
            canvas.delete('all')
            bar_width = int(190 * conf)
            color = '#27ae60' if i == 0 else '#3498db'
            canvas.create_rectangle(5, 5, 5 + bar_width, 15, fill=color, outline='')
    
    def _add_to_history(self, key, confidence):
        """Add detection to history"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_text.insert(tk.END, f"[{timestamp}] {key.upper()} ({confidence:.1f}%)\n")
        self.history_text.see(tk.END)
    
    def _update_statistics(self):
        """Update statistics"""
        self.stats_labels['keystrokes'].config(text=str(self.keystroke_count))
        
        if self.detection_history:
            avg_conf = sum(self.detection_history) / len(self.detection_history)
            self.stats_labels['avg_conf'].config(text=f"{avg_conf:.1f}%")
    
    def _update_recording_time(self, elapsed):
        """Update recording time"""
        mins, secs = divmod(elapsed, 60)
        self.stats_labels['time'].config(text=f"{mins:02d}:{secs:02d}")
    
    def _clear_results(self):
        """Clear all results"""
        self.current_key_label.config(text="-")
        self.confidence_label.config(text="Confidence: -")
        self.history_text.delete('1.0', tk.END)
        self.keystroke_count = 0
        self.detection_history = []
        self._update_statistics()
        self._log("Results cleared")
    
    def _toggle_save_mel(self):
        """Toggle saving mel-spectrograms"""
        self.save_mel_specs = self.save_mel_var.get()
        if self.save_mel_specs:
            self._log(f"Debug files will be saved to: {self.debug_dir}")
        else:
            self._log("Debug file saving disabled")
    
    def _update_threshold(self, value):
        """Update detection threshold"""
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.recorder.adjust_threshold(threshold)
    
    def _save_debug_mel_spec(self, mel_spec, filename):
        """Save mel-spectrogram visualization"""
        try:
            plt.figure(figsize=(8, 6))
            plt.imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Log Magnitude')
            plt.title(f'Mel Spectrogram - {filename}')
            plt.xlabel('Time')
            plt.ylabel('Mel Frequency')
            plt.tight_layout()
            
            img_path = self.debug_dir / f"{filename}.png"
            plt.savefig(str(img_path), dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self._log(f"Error saving mel-spec: {e}", "ERROR")
    
    def on_closing(self):
        """Handle window closing"""
        if self.is_recording:
            self._stop_recording()
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = KeystrokeDetectionUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
