"""
Keystroke Detection UI - Supporting Both CNN and CoAtNet Models
Auto-detects model architecture from state_dict and loads appropriately
Exact implementation matching the model architectures from CNN.ipynb and phone.ipynb
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
from typing import List, Tuple, Dict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import soundfile as sf
import sounddevice as sd
from collections import deque
import math
from torchvision.ops import SqueezeExcitation
from torchvision.transforms import Compose, ToTensor

# (64-1) * hop_length=256 → exactly 64 time frames matching CNN.ipynb training
TARGET_SAMPLES = 16128
MODEL_SAMPLE_RATE = 44100
MODEL_TARGET_DURATION = TARGET_SAMPLES / MODEL_SAMPLE_RATE


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
# CNN MODEL + PREPROCESSING (imported from cnn_shared.py)
# ============================================================================
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from cnn_shared import ConvBlock, CNN, build_mel_transform, PHONE_MEL_CONFIG


# ============================================================================
# PREPROCESSING (exact pipeline from CNN.ipynb via cnn_shared.PHONE_MEL_CONFIG)
# ============================================================================

class AudioPreprocessor:
    """Audio preprocessing matching CNN.ipynb training pipeline exactly.
    Uses PHONE_MEL_CONFIG from cnn_shared:
        n_mels=64, hop_length=256, n_fft=2048, win_length=1024,
        f_min=298.97, f_max=19569.78, power=1.0 (amplitude + log2)
    Input must be exactly TARGET_SAMPLES=16128 → (1, 64, 64) output.
    """

    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
        self.transforms = build_mel_transform(PHONE_MEL_CONFIG)

    def process_audio(self, waveform):
        """Process audio to mel spectrogram - returns (1, 64, 64) tensor"""
        if isinstance(waveform, np.ndarray):
            waveform = torch.from_numpy(waveform).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        return self.transforms(waveform)


class AudioRecorder:
    """Real-time audio recording with keystroke detection"""

    def __init__(self, sample_rate=44100, device_id=None):
        self.sample_rate = sample_rate
        self.device_id = device_id
        self.stream = None
        self.is_recording = False
        self.audio_buffer = deque(maxlen=int(sample_rate * 5))  # 5 second rolling buffer
        self.callback_func = None
        self.threshold = 0.06
        self.channels = 1

        # segment_duration derived from TARGET_SAMPLES — matches CNN.ipynb exactly
        self.segment_duration = TARGET_SAMPLES / sample_rate  # 0.3657s → 64 frames
        self.detection_buffer_size = int(sample_rate * 1.0)
        self.last_detection_time = 0
        self.min_keystroke_interval = 0.12
        self.processed_sample_index = 0
        self.recent_detection_times = deque(maxlen=5)

    def get_available_devices(self):
        """Get list of available input devices"""
        devices = []
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': int(device.get('default_samplerate', self.sample_rate))
                })
        return devices

    def set_device_config(self, device_id=None, sample_rate=None, channels=1):
        if device_id is not None:
            self.device_id = device_id
        if sample_rate is not None and sample_rate != self.sample_rate:
            self.sample_rate = int(sample_rate)
            self.audio_buffer = deque(maxlen=int(self.sample_rate * 5))
            self.detection_buffer_size = int(self.sample_rate * 1.0)
            self.segment_duration = TARGET_SAMPLES / MODEL_SAMPLE_RATE
        self.channels = max(1, int(channels))

    def set_audio_callback(self, callback):
        self.callback_func = callback

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"Audio callback status: {status}")
        audio_data = indata[:, 0].astype(np.float32, copy=True)
        self.audio_buffer.extend(audio_data)
        if self.callback_func:
            self.callback_func(audio_data)

    def start_recording(self):
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
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.is_recording = False

    def compute_onset_strength(self, audio: np.ndarray,
                                hop_length: int = 512,
                                win_length: int = 2048) -> np.ndarray:
        from scipy import signal as scipy_signal

        f, t, Zxx = scipy_signal.stft(audio, fs=self.sample_rate,
                                      nperseg=win_length,
                                      noverlap=win_length - hop_length)
        magnitude = np.abs(Zxx)

        freq_mask = (f >= 50) & (f <= 5000)
        magnitude_filtered = magnitude[freq_mask, :]

        onset_env = np.zeros(magnitude_filtered.shape[1])
        if magnitude_filtered.shape[1] > 1:
            onset_env[1:] = np.sum(np.maximum(0, magnitude_filtered[:, 1:] - magnitude_filtered[:, :-1]), axis=0)

        if len(onset_env) > 5:
            window = scipy_signal.windows.hann(5)
            onset_env = scipy_signal.convolve(onset_env, window / window.sum(), mode='same')

        return onset_env

    def find_peaks_with_quality(self, onset_env: np.ndarray,
                                  hop_length: int = 512) -> List[Tuple[int, float]]:
        from scipy.ndimage import maximum_filter

        if len(onset_env) < 10:
            return []

        threshold = np.percentile(onset_env, 80)
        min_absolute_threshold = np.max(onset_env) * 0.15
        threshold = max(threshold, min_absolute_threshold)

        min_distance_frames = int(0.10 * self.sample_rate / hop_length)
        size = min_distance_frames * 2 + 1
        local_max = maximum_filter(onset_env, size=size, mode='constant')

        peaks = (onset_env == local_max) & (onset_env > threshold)
        peak_indices = np.where(peaks)[0]

        peaks_with_quality = []
        for peak_idx in peak_indices:
            sample_pos = peak_idx * hop_length
            quality = onset_env[peak_idx] / (np.max(onset_env) + 1e-10)
            peaks_with_quality.append((sample_pos, quality))

        return peaks_with_quality

    def extract_segment_centered(self, audio: np.ndarray, peak_sample: int, target_samples: int = TARGET_SAMPLES) -> np.ndarray:
        """Extract target_samples centered around peak. No padding."""
        start = peak_sample - target_samples // 2
        end = start + target_samples

        start = max(0, start)
        end = min(len(audio), end)

        return audio[start:end]

    def validate_segment_quality(self, segment: np.ndarray, target_samples: int = TARGET_SAMPLES) -> Tuple[bool, float]:
        """Validate segment quality. Segment must be exactly TARGET_SAMPLES."""
        if len(segment) < target_samples:
            return False, 0.0

        segment = segment[:target_samples]

        rms = np.sqrt(np.mean(segment**2))
        if rms < 1e-4:
            return False, 0.0

        peak = np.max(np.abs(segment))
        peak_to_rms = peak / (rms + 1e-10)

        if peak_to_rms < 2.0:
            return False, 0.0

        fft = np.fft.rfft(segment)
        freqs = np.fft.rfftfreq(len(segment), 1 / self.sample_rate)
        magnitude = np.abs(fft)

        freq_mask = (freqs >= 50) & (freqs <= 5000)
        if np.sum(freq_mask) == 0:
            return False, 0.0

        keystroke_energy = np.sum(magnitude[freq_mask]**2)
        total_energy = np.sum(magnitude**2)
        energy_ratio = keystroke_energy / (total_energy + 1e-10)

        if energy_ratio < 0.6:
            return False, 0.0

        quality = min(1.0, (peak_to_rms / 10.0) * energy_ratio)
        return True, quality

    def detect_keystrokes(self):
        """Advanced keystroke detection using onset detection and quality validation."""
        segment_samples = int(round(MODEL_TARGET_DURATION * self.sample_rate))
        if len(self.audio_buffer) < max(self.detection_buffer_size, segment_samples):
            return []

        current_time = time.time()

        if current_time - self.last_detection_time < 0.08:
            return []

        audio = np.array(self.audio_buffer)
        buffer_length = len(audio)
        new_audio_start = max(0, buffer_length - int(self.sample_rate * 1.0))

        if new_audio_start >= buffer_length:
            return []

        audio_to_process = audio[new_audio_start:]

        hop_length = 512
        onset_env = self.compute_onset_strength(audio_to_process, hop_length=hop_length)

        if len(onset_env) == 0:
            return []

        peaks_with_quality = self.find_peaks_with_quality(onset_env, hop_length=hop_length)

        valid_keystrokes = []

        for peak_sample, peak_quality in peaks_with_quality:
            absolute_peak = new_audio_start + peak_sample

            # Skip if segment would go out of buffer bounds
            if absolute_peak < segment_samples // 2:
                continue
            if absolute_peak + segment_samples // 2 > buffer_length:
                continue

            segment = self.extract_segment_centered(audio, absolute_peak, target_samples=segment_samples)

            if len(segment) > segment_samples:
                segment = segment[:segment_samples]

            is_valid, quality_score = self.validate_segment_quality(segment, target_samples=segment_samples)

            if is_valid and quality_score > 0.3:
                time_since_last = current_time - self.last_detection_time
                if time_since_last < self.min_keystroke_interval and len(valid_keystrokes) > 0:
                    continue

                valid_keystrokes.append({
                    'audio': segment,
                    'peak_sample': absolute_peak,
                    'quality': quality_score,
                    'peak_quality': peak_quality
                })

                self.last_detection_time = current_time
                self.recent_detection_times.append(current_time)

                print(f"[Keystroke Detected] Position: {absolute_peak/self.sample_rate:.3f}s, "
                      f"Quality: {quality_score:.2f}, Peak: {peak_quality:.2f}")
                break

        keep_samples = int(self.sample_rate * 2.0)
        if len(self.audio_buffer) > keep_samples:
            buffer_list = list(self.audio_buffer)
            self.audio_buffer = deque(buffer_list[-keep_samples:], maxlen=self.audio_buffer.maxlen)

        return [ks['audio'] for ks in valid_keystrokes]

    def adjust_threshold(self, threshold):
        self.threshold = threshold


# ============================================================================
# UI APPLICATION
# ============================================================================

class KeystrokeDetectionUI:
    """Main UI for keystroke detection"""

    def __init__(self, root):
        self.root = root
        self.root.title("Keystroke Detection System - CNN/CoAtNet")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_loaded = False
        self.model_path = None
        self.num_classes = 36

        self.preprocessor = AudioPreprocessor(sample_rate=44100)
        self.recorder = AudioRecorder(sample_rate=44100)
        self.recorder.set_audio_callback(self._on_audio_chunk)

        self.is_recording = False
        self.processing_thread = None
        self.stop_processing = False
        self.start_time = None

        digits = [str(digit) for digit in range(10)]
        alphabet = [chr(ascii_code) for ascii_code in range(ord('a'), ord('z') + 1)]
        self.key_labels = digits + alphabet
        self.alphabet_only = alphabet

        self.keystroke_count = 0
        self.detection_history = []

        self.save_mel_specs = False
        self.debug_dir = Path("model/debug_output")
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.available_devices = []

        self._create_widgets()
        self._log("Application started")
        self._log(f"Device: {self.device.upper()}")
        self._log(f"Target samples: {TARGET_SAMPLES} ({TARGET_SAMPLES/MODEL_SAMPLE_RATE:.4f}s) → 64 time frames")
        self._log(f"Mel config: PHONE_MEL_CONFIG (n_mels=64, hop=256, power=1.0, log2)")

        self._auto_load_model()

    def _create_widgets(self):
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        title_frame.pack(fill=tk.X, side=tk.TOP)
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame,
            text="🎹 Keystroke Detection - CNN/CoAtNet",
            font=('Arial', 18, 'bold'),
            bg='#2c3e50',
            fg='white'
        ).pack(pady=15)

        main_container = tk.Frame(self.root, padx=20, pady=10)
        main_container.pack(fill=tk.BOTH, expand=True)

        self._create_model_section(main_container)
        self._create_recording_section(main_container)
        self._create_results_section(main_container)
        self._create_log_section(main_container)
        self._create_statistics_section(main_container)
        self._create_status_bar()

    def _create_model_section(self, parent):
        model_frame = tk.LabelFrame(parent, text="Model Configuration", font=('Arial', 11, 'bold'), padx=10, pady=10)
        model_frame.pack(fill=tk.X, pady=(0, 10))

        path_frame = tk.Frame(model_frame)
        path_frame.pack(fill=tk.X, pady=5)

        tk.Label(path_frame, text="Model Path:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))

        self.model_path_var = tk.StringVar()
        tk.Entry(path_frame, textvariable=self.model_path_var, font=('Arial', 10), state='readonly').pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))

        tk.Button(path_frame, text="Browse", command=self._browse_model, font=('Arial', 10)).pack(
            side=tk.LEFT, padx=(0, 5))
        tk.Button(path_frame, text="Load Model", command=self._load_model,
                  font=('Arial', 10, 'bold'), bg='#3498db', fg='white').pack(side=tk.LEFT)

        self.model_status_label = tk.Label(model_frame, text="⚠ No model loaded", font=('Arial', 10), fg='orange')
        self.model_status_label.pack(pady=5)

        tk.Label(model_frame, text=f"Device: {self.device.upper()}", font=('Arial', 9), fg='gray').pack()

    def _create_recording_section(self, parent):
        recording_frame = tk.LabelFrame(parent, text="Recording Control", font=('Arial', 11, 'bold'), padx=10, pady=10)
        recording_frame.pack(fill=tk.X, pady=(0, 10))

        mic_frame = tk.Frame(recording_frame)
        mic_frame.pack(fill=tk.X, pady=(5, 10))

        tk.Label(mic_frame, text="Microphone:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))

        self.mic_var = tk.StringVar()
        self.mic_combo = ttk.Combobox(mic_frame, textvariable=self.mic_var, state='readonly', font=('Arial', 9), width=50)
        self.mic_combo.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.mic_combo.bind('<<ComboboxSelected>>', self._on_mic_selected)

        tk.Button(mic_frame, text="🔄 Refresh", command=self._refresh_devices, font=('Arial', 9)).pack(side=tk.LEFT)

        self._refresh_devices()

        controls_frame = tk.Frame(recording_frame)
        controls_frame.pack(pady=10)

        self.record_btn = tk.Button(
            controls_frame, text="🎤 Start Recording", command=self._toggle_recording,
            font=('Arial', 12, 'bold'), bg='#27ae60', fg='white', width=20, height=2, state=tk.DISABLED)
        self.record_btn.pack(side=tk.LEFT, padx=10)

        tk.Button(controls_frame, text="🗑 Clear Results", command=self._clear_results,
                  font=('Arial', 10), width=15).pack(side=tk.LEFT, padx=10)

        self.test_wav_btn = tk.Button(
            controls_frame, text="📁 Test WAV File", command=self._test_wav_file,
            font=('Arial', 10), bg='#9b59b6', fg='white', width=15, state=tk.DISABLED)
        self.test_wav_btn.pack(side=tk.LEFT, padx=10)

        self.bulk_check_btn = tk.Button(
            controls_frame, text="📊 Bulk Check", command=self._bulk_check_folder,
            font=('Arial', 10), bg='#e67e22', fg='white', width=15, state=tk.DISABLED)
        self.bulk_check_btn.pack(side=tk.LEFT, padx=10)

        debug_frame = tk.Frame(recording_frame)
        debug_frame.pack(fill=tk.X, pady=5)

        self.save_mel_var = tk.BooleanVar(value=False)
        tk.Checkbutton(debug_frame, text="Save audio & mel-spectrograms for debugging",
                       variable=self.save_mel_var, font=('Arial', 9),
                       command=self._toggle_save_mel).pack(side=tk.LEFT)

        threshold_frame = tk.Frame(recording_frame)
        threshold_frame.pack(fill=tk.X, pady=5)

        tk.Label(threshold_frame, text="Detection Sensitivity:", font=('Arial', 10)).pack(side=tk.LEFT, padx=(0, 10))

        self.threshold_var = tk.DoubleVar(value=0.06)
        tk.Scale(threshold_frame, from_=0.01, to=0.2, resolution=0.01, orient=tk.HORIZONTAL,
                 variable=self.threshold_var, command=self._update_threshold, length=200).pack(
            side=tk.LEFT, fill=tk.X, expand=True)

        self.threshold_label = tk.Label(threshold_frame, text="0.06", font=('Arial', 10))
        self.threshold_label.pack(side=tk.LEFT, padx=(10, 0))

    def _create_results_section(self, parent):
        results_frame = tk.LabelFrame(parent, text="Detection Results", font=('Arial', 11, 'bold'), padx=10, pady=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        current_frame = tk.Frame(results_frame)
        current_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Label(current_frame, text="Last Detected Key:", font=('Arial', 11, 'bold')).pack(side=tk.LEFT, padx=(0, 10))

        self.current_key_label = tk.Label(
            current_frame, text="-", font=('Arial', 28, 'bold'),
            bg='#ecf0f1', fg='#2c3e50', width=3, relief=tk.RAISED, borderwidth=2)
        self.current_key_label.pack(side=tk.LEFT, padx=10)

        self.confidence_label = tk.Label(current_frame, text="Confidence: -", font=('Arial', 11))
        self.confidence_label.pack(side=tk.LEFT, padx=20)

        tk.Label(results_frame, text="Top 5 Predictions:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

        predictions_container = tk.Frame(results_frame)
        predictions_container.pack(fill=tk.X)

        self.prediction_labels = []
        for i in range(5):
            pred_frame = tk.Frame(predictions_container, bg='#ecf0f1', relief=tk.GROOVE, borderwidth=1)
            pred_frame.pack(fill=tk.X, pady=2)

            tk.Label(pred_frame, text=f"#{i+1}", font=('Arial', 9, 'bold'), bg='#ecf0f1', width=4).pack(side=tk.LEFT, padx=5)
            key_label = tk.Label(pred_frame, text="-", font=('Arial', 10, 'bold'), bg='#ecf0f1', width=3)
            key_label.pack(side=tk.LEFT, padx=5)
            bar_canvas = tk.Canvas(pred_frame, width=200, height=20, bg='white', highlightthickness=0)
            bar_canvas.pack(side=tk.LEFT, padx=5)
            conf_label = tk.Label(pred_frame, text="0.00%", font=('Arial', 9), bg='#ecf0f1', width=8)
            conf_label.pack(side=tk.LEFT, padx=5)

            self.prediction_labels.append({'key': key_label, 'bar': bar_canvas, 'conf': conf_label})

        tk.Label(results_frame, text="Detection History:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(10, 5))

        history_container = tk.Frame(results_frame)
        history_container.pack(fill=tk.BOTH, expand=True)

        scrollbar = tk.Scrollbar(history_container)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.history_text = tk.Text(history_container, height=6, font=('Courier', 9), yscrollcommand=scrollbar.set)
        self.history_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.history_text.yview)

    def _create_statistics_section(self, parent):
        stats_frame = tk.LabelFrame(parent, text="Statistics", font=('Arial', 11, 'bold'), padx=10, pady=10)
        stats_frame.pack(fill=tk.X)

        stats_container = tk.Frame(stats_frame)
        stats_container.pack(fill=tk.X)

        self.stats_labels = {}

        for label_text, key in [('Total Keystrokes:', 'keystrokes'), ('Recording Time:', 'time'), ('Avg Confidence:', 'avg_conf')]:
            frame = tk.Frame(stats_container)
            frame.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=10)
            tk.Label(frame, text=label_text, font=('Arial', 9)).pack()
            value_label = tk.Label(frame, text="0", font=('Arial', 12, 'bold'), fg='#3498db')
            value_label.pack()
            self.stats_labels[key] = value_label

    def _create_log_section(self, parent):
        log_frame = tk.LabelFrame(parent, text="System Log", font=('Arial', 11, 'bold'), padx=10, pady=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))

        log_container = tk.Frame(log_frame)
        log_container.pack(fill=tk.BOTH, expand=True)

        log_scrollbar = tk.Scrollbar(log_container)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.log_text = tk.Text(log_container, height=8, font=('Courier', 8),
                                yscrollcommand=log_scrollbar.set, bg='#f8f9fa')
        self.log_text.pack(fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.log_text.yview)

        tk.Button(log_frame, text="Clear Log", command=self._clear_log, font=('Arial', 9)).pack(pady=(5, 0))

    def _create_status_bar(self):
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W, font=('Arial', 9))
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _log(self, message, level="INFO"):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        log_entry = f"[{timestamp}] [{level}] {message}\n"
        try:
            self.log_text.insert(tk.END, log_entry)
            self.log_text.see(tk.END)
            print(log_entry.strip())
        except:
            print(log_entry.strip())

    def _clear_log(self):
        self.log_text.delete('1.0', tk.END)
        self._log("Log cleared")

    def _refresh_devices(self):
        self._log("Refreshing audio devices...")
        try:
            devices = self.recorder.get_available_devices()
            self.available_devices = devices
            device_names = [f"{d['id']}: {d['name']}" for d in devices]
            self.mic_combo['values'] = device_names
            if device_names:
                preferred_index = 0
                for idx, device in enumerate(devices):
                    device_name = device['name'].lower()
                    if 'dji' in device_name or 'mic' in device_name:
                        preferred_index = idx
                        break

                self.mic_combo.current(preferred_index)
                selected = devices[preferred_index]
                self.recorder.set_device_config(
                    device_id=selected['id'],
                    sample_rate=selected.get('default_samplerate', 44100),
                    channels=1,
                )
                self._log(
                    f"Found {len(device_names)} audio device(s); using {selected['name']} at {selected.get('default_samplerate', 44100)} Hz"
                )
            else:
                self._log("No audio devices found!", "WARNING")
        except Exception as e:
            self._log(f"Error refreshing devices: {e}", "ERROR")

    def _on_mic_selected(self, event):
        selection = self.mic_combo.current()
        if selection >= 0 and selection < len(self.available_devices):
            device = self.available_devices[selection]
            self.recorder.set_device_config(
                device_id=device['id'],
                sample_rate=device.get('default_samplerate', 44100),
                channels=1,
            )
            self._log(f"Microphone selected: {device['name']} ({device.get('default_samplerate', 44100)} Hz)")

    def _on_audio_chunk(self, audio_chunk):
        pass

    def _prepare_waveform_for_model(self, samples, source_sample_rate):
        if isinstance(samples, np.ndarray):
            waveform = torch.from_numpy(samples).float()
        elif isinstance(samples, torch.Tensor):
            waveform = samples.float()
        else:
            waveform = torch.tensor(samples, dtype=torch.float32)

        if waveform.dim() > 1:
            waveform = waveform.squeeze()
        if waveform.dim() == 0:
            waveform = waveform.unsqueeze(0)

        if source_sample_rate != MODEL_SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform.unsqueeze(0), source_sample_rate, MODEL_SAMPLE_RATE).squeeze(0)

        if waveform.numel() > TARGET_SAMPLES:
            start = (waveform.numel() - TARGET_SAMPLES) // 2
            waveform = waveform[start:start + TARGET_SAMPLES]
        elif waveform.numel() < TARGET_SAMPLES:
            pad_total = TARGET_SAMPLES - waveform.numel()
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            left_pad = waveform[:1].expand(pad_left) if pad_left > 0 else waveform[:0]
            right_pad = waveform[-1:].expand(pad_right) if pad_right > 0 else waveform[:0]
            waveform = torch.cat([left_pad, waveform, right_pad])

        return waveform.unsqueeze(0)

    def _browse_model(self):
        filename = filedialog.askopenfilename(
            title="Select Model File", initialdir=".",
            filetypes=[("PyTorch Model", "*.pkl *.pth *.pt"), ("All Files", "*.*")])
        if filename:
            self.model_path_var.set(filename)

    def _auto_load_model(self):
        self._log("Searching for model files...")
        for search_path in [".", "model", "../"]:
            for model_name in ["CoAtNet-1-Best-Phone.pkl", "CoAtNet-1-Phone.pkl"]:
                model_path = Path(search_path) / model_name
                if model_path.exists():
                    self._log(f"Found model: {model_path}")
                    self.model_path_var.set(str(model_path))
                    self._load_model()
                    return
        self._log("No model found. Please load manually.", "WARNING")

    def _detect_model_type(self, state_dict):
        keys = list(state_dict.keys())

        new_cnn_keys = ['conv1.conv.0.weight', 'conv2.conv.0.weight', 'fc.7.weight']
        if any(key in keys for key in new_cnn_keys):
            self._log(f"Detected new CNN (ConvBlock) keys: {[k for k in new_cnn_keys if k in keys]}")
            if 'fc.7.weight' in state_dict:
                num_classes = state_dict['fc.7.weight'].shape[0]
                self._log(f"Detected {num_classes} classes from fc.7.weight shape")
                return 'CNN', num_classes
            elif 'fc.7.bias' in state_dict:
                num_classes = state_dict['fc.7.bias'].shape[0]
                self._log(f"Detected {num_classes} classes from fc.7.bias shape")
                return 'CNN', num_classes
            return 'CNN', 36

        old_cnn_keys = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'fc1.weight', 'fc2.weight']
        if any(key in keys for key in old_cnn_keys):
            self._log(f"Detected old CNN keys: {[k for k in old_cnn_keys if k in keys]}")
            if 'fc2.weight' in state_dict:
                num_classes = state_dict['fc2.weight'].shape[0]
                self._log(f"Detected {num_classes} classes from fc2.weight shape")
                return 'CNN', num_classes
            return 'CNN', 36

        coatnet_keys = ['0.0.0.weight', '1.0.mb_conv.0.weight', '3.0.0.relative_bias']
        if any(key in keys for key in coatnet_keys):
            self._log(f"Detected CoAtNet keys: {[k for k in coatnet_keys if k in keys]}")
            if '5.fc.weight' in state_dict:
                num_classes = state_dict['5.fc.weight'].shape[0]
                self._log(f"Detected {num_classes} classes from head layer")
                return 'CoAtNet', num_classes
            return 'CoAtNet', 36

        self._log(f"First 5 keys in state_dict: {keys[:5]}")
        self._log("Defaulting to CNN model type")
        return 'CNN', 36

    def _load_model(self):
        model_path = self.model_path_var.get()

        if not model_path or not os.path.exists(model_path):
            self._log("Invalid model path", "ERROR")
            messagebox.showerror("Error", "Please select a valid model file")
            return

        try:
            self._log(f"Loading model from: {model_path}")
            self.status_bar.config(text="Loading model...")
            self.root.update()

            state_dict = torch.load(model_path, map_location=self.device)
            model_type, num_classes = self._detect_model_type(state_dict)
            self._log(f"Detected model type: {model_type}")

            if model_type == 'CNN':
                self._log(f"Creating CNN model with {num_classes} classes...")
                self.model = CNN(num_classes=num_classes)
            else:
                self._log(f"Creating CoAtNet-1 model with {num_classes} classes...")
                self.model = MyCoAtNet([2, 2, 3, 5, 2], [64, 96, 192, 384, 768], num_classes=num_classes)

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.num_classes = num_classes
            if num_classes == 10:
                self.key_labels = [str(d) for d in range(10)]
                self._log("Using digit-only labels (0-9)")
            elif num_classes == 9:
                self.key_labels = [str(d) for d in range(9)]
                self._log("Using digit-only labels (0-8)")
            elif num_classes == 26:
                self.key_labels = self.alphabet_only
                self._log("Using alphabet-only labels (a-z)")
            elif num_classes == 36:
                self.key_labels = [str(d) for d in range(10)] + self.alphabet_only
                self._log("Using alphanumeric labels (0-9, a-z)")
            else:
                if num_classes <= 10:
                    self.key_labels = [str(d) for d in range(num_classes)]
                elif num_classes <= 26:
                    self.key_labels = self.alphabet_only[:num_classes]
                else:
                    self.key_labels = [str(d) for d in range(10)] + self.alphabet_only[:num_classes - 10]
                self._log(f"Using best-guess labels for {num_classes} classes", "WARNING")

            total_params = sum(p.numel() for p in self.model.parameters())
            self._log(f"Model parameters: {total_params:,}")

            with torch.no_grad():
                test_output = self.model(torch.randn(1, 1, 64, 64).to(self.device))
                self._log(f"Model test output shape: {test_output.shape}")

            self.model_loaded = True
            self.model_status_label.config(text=f"✓ {model_type} model loaded successfully", fg='green')
            self.record_btn.config(state=tk.NORMAL)
            self.test_wav_btn.config(state=tk.NORMAL)
            self.bulk_check_btn.config(state=tk.NORMAL)
            self.status_bar.config(text=f"{model_type} model loaded: {os.path.basename(model_path)}")

            self._log(f"{model_type} model loaded successfully!")
            messagebox.showinfo("Success", f"{model_type} model loaded successfully!")

        except Exception as e:
            self._log(f"Failed to load model: {e}", "ERROR")
            import traceback
            self._log(traceback.format_exc(), "ERROR")
            messagebox.showerror("Error", f"Failed to load model:\n{str(e)}")
            self.model_loaded = False
            self.status_bar.config(text="Model loading failed")

    def _bulk_check_folder(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load a model first!")
            return

        folder_path = filedialog.askdirectory(title="Select Segmented Audio Folder", initialdir=".")
        if not folder_path:
            return

        results_window = tk.Toplevel(self.root)
        results_window.title("Bulk Check Results")
        results_window.geometry("800x600")

        title_frame = tk.Frame(results_window, bg='#34495e', height=50)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)
        tk.Label(title_frame, text="📊 Bulk Check Results", font=('Arial', 16, 'bold'),
                 bg='#34495e', fg='white').pack(pady=10)

        progress_frame = tk.Frame(results_window, padx=20, pady=10)
        progress_frame.pack(fill=tk.X)
        progress_label = tk.Label(progress_frame, text="Processing...", font=('Arial', 10))
        progress_label.pack()
        progress_bar = ttk.Progressbar(progress_frame, length=700, mode='determinate')
        progress_bar.pack(pady=5)

        results_frame = tk.LabelFrame(results_window, text="Summary", font=('Arial', 11, 'bold'), padx=20, pady=15)
        results_frame.pack(fill=tk.X, padx=20, pady=10)
        summary_text = tk.Text(results_frame, height=10, font=('Courier', 9), bg='#f8f9fa')
        summary_text.pack(fill=tk.BOTH, expand=True)

        notebook = ttk.Notebook(results_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))

        details_frame = tk.Frame(notebook)
        notebook.add(details_frame, text="Detailed Results")
        details_scrollbar = tk.Scrollbar(details_frame)
        details_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        details_text = tk.Text(details_frame, font=('Courier', 9), yscrollcommand=details_scrollbar.set)
        details_text.pack(fill=tk.BOTH, expand=True)
        details_scrollbar.config(command=details_text.yview)

        stats_frame = tk.Frame(notebook)
        notebook.add(stats_frame, text="Statistics & Analysis")
        stats_scrollbar = tk.Scrollbar(stats_frame)
        stats_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        stats_text = tk.Text(stats_frame, font=('Courier', 9), yscrollcommand=stats_scrollbar.set, bg='#f8f9fa')
        stats_text.pack(fill=tk.BOTH, expand=True)
        stats_scrollbar.config(command=stats_text.yview)

        def process_folder():
            try:
                self._log(f"Starting bulk check on: {folder_path}")
                folder_path_obj = Path(folder_path)
                label_folders = [f for f in folder_path_obj.iterdir() if f.is_dir()]

                if not label_folders:
                    self._log("No subdirectories found!", "ERROR")
                    results_window.after(0, lambda: messagebox.showerror("Error", "No label subdirectories found!"))
                    results_window.after(0, results_window.destroy)
                    return

                correct = 0
                wrong = 0
                errors = 0
                results_by_label = {}
                confusion_data = {}
                all_predictions = []

                all_wav_files = []
                for label_folder in label_folders:
                    for wav_file in label_folder.glob("*.wav"):
                        all_wav_files.append((wav_file, label_folder.name))

                total_files = len(all_wav_files)

                if total_files == 0:
                    self._log("No WAV files found!", "ERROR")
                    results_window.after(0, lambda: messagebox.showerror("Error", "No WAV files found!"))
                    results_window.after(0, results_window.destroy)
                    return

                self._log(f"Found {total_files} files across {len(label_folders)} labels")

                for idx, (wav_file, expected_label) in enumerate(all_wav_files):
                    try:
                        audio_data, sr = sf.read(str(wav_file), dtype='float32')

                        if len(audio_data.shape) == 1:
                            waveform = audio_data
                        else:
                            waveform = audio_data.mean(axis=1)

                        waveform = self._prepare_waveform_for_model(waveform, sr)

                        if waveform.shape[1] != TARGET_SAMPLES:
                            self._log(f"Skipping {wav_file.name}: unexpected prepared shape {waveform.shape}", "WARNING")
                            errors += 1
                            continue

                        mel_spec = self.preprocessor.process_audio(waveform)

                        if mel_spec.shape != torch.Size([1, 64, 64]):
                            self._log(f"Skipping {wav_file.name}: wrong shape {mel_spec.shape}", "WARNING")
                            errors += 1
                            continue

                        with torch.no_grad():
                            input_tensor = mel_spec.unsqueeze(0).to(self.device)
                            output = self.model(input_tensor)
                            if output.dim() == 1:
                                output = output.unsqueeze(0)
                            probabilities = F.softmax(output, dim=1)
                            predicted_idx = torch.argmax(probabilities[0]).item()
                            predicted_label = self.key_labels[predicted_idx]
                            confidence = probabilities[0][predicted_idx].item() * 100

                        is_correct = (predicted_label == expected_label)
                        if is_correct:
                            correct += 1
                        else:
                            wrong += 1

                        all_predictions.append(predicted_label)

                        if expected_label not in results_by_label:
                            results_by_label[expected_label] = {'total': 0, 'correct': 0, 'wrong': 0, 'confidences': []}
                        results_by_label[expected_label]['total'] += 1
                        results_by_label[expected_label]['confidences'].append(confidence)
                        if is_correct:
                            results_by_label[expected_label]['correct'] += 1
                        else:
                            results_by_label[expected_label]['wrong'] += 1

                        if expected_label not in confusion_data:
                            confusion_data[expected_label] = {}
                        confusion_data[expected_label][predicted_label] = confusion_data[expected_label].get(predicted_label, 0) + 1

                        status = "✓" if is_correct else "✗"
                        color = "green" if is_correct else "red"
                        detail_line = f"{status} {wav_file.name:40s} Expected: {expected_label:2s}  Predicted: {predicted_label:2s}  ({confidence:5.1f}%)\n"

                        def update_details(line, tag):
                            details_text.insert(tk.END, line, tag)
                            details_text.tag_config(tag, foreground=color)
                            details_text.see(tk.END)

                        results_window.after(0, update_details, detail_line, f"tag_{idx}")

                    except Exception as e:
                        self._log(f"Error processing {wav_file.name}: {e}", "ERROR")
                        errors += 1

                    progress = (idx + 1) / total_files * 100
                    results_window.after(0, lambda p=progress, i=idx+1, t=total_files: (
                        progress_bar.config(value=p),
                        progress_label.config(text=f"Processing: {i}/{t} files ({p:.1f}%)")
                    ))

                processed = correct + wrong
                success_rate = (correct / processed * 100) if processed > 0 else 0

                from collections import Counter
                prediction_counts = Counter(all_predictions)

                label_stats = []
                for label in sorted(results_by_label.keys()):
                    s = results_by_label[label]
                    label_stats.append({
                        'label': label,
                        'accuracy': (s['correct'] / s['total'] * 100) if s['total'] > 0 else 0,
                        'correct': s['correct'],
                        'total': s['total'],
                        'avg_conf': sum(s['confidences']) / len(s['confidences']) if s['confidences'] else 0
                    })

                best_labels = sorted(label_stats, key=lambda x: x['accuracy'], reverse=True)[:5]
                worst_labels = sorted(label_stats, key=lambda x: x['accuracy'])[:5]
                most_predicted = prediction_counts.most_common(5)

                summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                     BULK CHECK SUMMARY                        ║
╚══════════════════════════════════════════════════════════════╝

Folder: {folder_path_obj.name}

Total Files:        {total_files}
Processed:          {processed}
Errors:             {errors}

✓ Correct:          {correct}
✗ Wrong:            {wrong}

SUCCESS RATE:       {success_rate:.2f}%
Accuracy:           {correct}/{processed}

═══════════════════════════════════════════════════════════════
"""

                stats_report = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DETAILED STATISTICS & ANALYSIS                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

OVERALL PERFORMANCE
───────────────────────────────────────────────────────────────────────────────
Total Samples:          {processed}
Correct Predictions:    {correct} ({success_rate:.2f}%)
Wrong Predictions:      {wrong} ({100-success_rate:.2f}%)
Error Rate:             {(wrong/processed*100) if processed > 0 else 0:.2f}%


PER-LABEL ACCURACY (All Labels)
───────────────────────────────────────────────────────────────────────────────
Label  | Correct | Total | Accuracy | Avg Confidence
────---|---------|-------|----------|----------------
"""
                for item in sorted(label_stats, key=lambda x: x['label']):
                    stats_report += f"{item['label']:^6s} | {item['correct']:^7d} | {item['total']:^5d} | {item['accuracy']:^7.2f}% | {item['avg_conf']:^13.2f}%\n"

                stats_report += f"""
BEST PERFORMING LABELS (Top 5)
───────────────────────────────────────────────────────────────────────────────
Rank | Label | Accuracy | Correct/Total | Avg Confidence
-----|-------|----------|---------------|----------------
"""
                for i, item in enumerate(best_labels, 1):
                    stats_report += f"{i:^4d} | {item['label']:^5s} | {item['accuracy']:^7.2f}% | {item['correct']:^5d}/{item['total']:^5d} | {item['avg_conf']:^13.2f}%\n"

                stats_report += f"""
WORST PERFORMING LABELS (Bottom 5)
───────────────────────────────────────────────────────────────────────────────
Rank | Label | Accuracy | Correct/Total | Avg Confidence
-----|-------|----------|---------------|----------------
"""
                for i, item in enumerate(worst_labels, 1):
                    stats_report += f"{i:^4d} | {item['label']:^5s} | {item['accuracy']:^7.2f}% | {item['correct']:^5d}/{item['total']:^5d} | {item['avg_conf']:^13.2f}%\n"

                stats_report += f"""
MODEL PREDICTION BIAS ANALYSIS
───────────────────────────────────────────────────────────────────────────────
Most Frequently Predicted Labels (Top 5):

Label | Times Predicted | % of All Predictions
------|-----------------|---------------------
"""
                for label, count in most_predicted:
                    pct = (count / len(all_predictions) * 100) if all_predictions else 0
                    stats_report += f"{label:^5s} | {count:^15d} | {pct:^19.2f}%\n"

                actual_dist = {l: s['total'] for l, s in results_by_label.items()}
                pred_dist = dict(prediction_counts)

                bias_analysis = []
                for label in results_by_label:
                    actual = actual_dist.get(label, 0)
                    predicted = pred_dist.get(label, 0)
                    if actual > 0:
                        bias_analysis.append((label, predicted / actual, actual, predicted))

                over_predicted = sorted([x for x in bias_analysis if x[1] > 1.0], key=lambda x: x[1], reverse=True)[:5]
                under_predicted = sorted([x for x in bias_analysis if x[1] < 1.0], key=lambda x: x[1])[:5]

                if over_predicted:
                    stats_report += "\nOVER-PREDICTED LABELS\n───────────────────────────────────────────────────────────────────────────────\nLabel | Bias Ratio | Actual | Predicted\n"
                    for label, ratio, actual, predicted in over_predicted:
                        stats_report += f"{label:^5s} | {ratio:^10.2f} | {actual:^6d} | {predicted:^9d}\n"
                    stats_report += "\nNote: Bias Ratio > 1.0 means model predicts this label more than it appears\n"

                if under_predicted:
                    stats_report += "\nUNDER-PREDICTED LABELS\n───────────────────────────────────────────────────────────────────────────────\nLabel | Bias Ratio | Actual | Predicted\n"
                    for label, ratio, actual, predicted in under_predicted:
                        stats_report += f"{label:^5s} | {ratio:^10.2f} | {actual:^6d} | {predicted:^9d}\n"
                    stats_report += "\nNote: Bias Ratio < 1.0 means model predicts this label less than it appears\n"

                stats_report += "\nCONFUSION PATTERNS (Worst Performing Labels)\n───────────────────────────────────────────────────────────────────────────────\n"
                for item in worst_labels[:3]:
                    label = item['label']
                    if label in confusion_data:
                        stats_report += f"\nLabel '{label}' (Accuracy: {item['accuracy']:.2f}%):\n  Often confused with:\n"
                        confusions = sorted([(p, c) for p, c in confusion_data[label].items() if p != label],
                                           key=lambda x: x[1], reverse=True)[:5]
                        for pred_label, count in confusions:
                            pct = (count / item['total'] * 100) if item['total'] > 0 else 0
                            stats_report += f"    → '{pred_label}': {count} times ({pct:.1f}%)\n"

                stats_report += "\n" + "═" * 79 + "\n"

                results_window.after(0, lambda: (
                    summary_text.delete('1.0', tk.END),
                    summary_text.insert(tk.END, summary),
                    stats_text.delete('1.0', tk.END),
                    stats_text.insert(tk.END, stats_report),
                    progress_label.config(text=f"✓ Completed! Success Rate: {success_rate:.2f}%")
                ))

                self._log(f"Bulk check completed: {success_rate:.2f}% success rate")
                self._log(f"Best: {best_labels[0]['label']} ({best_labels[0]['accuracy']:.1f}%), Worst: {worst_labels[0]['label']} ({worst_labels[0]['accuracy']:.1f}%)")

            except Exception as e:
                self._log(f"Error in bulk check: {e}", "ERROR")
                import traceback
                self._log(traceback.format_exc(), "ERROR")
                results_window.after(0, lambda: messagebox.showerror("Error", f"Bulk check failed:\n{str(e)}"))

        threading.Thread(target=process_folder, daemon=True).start()

    def _test_wav_file(self):
        if not self.model_loaded:
            messagebox.showerror("Error", "Please load a model first!")
            return

        file_path = filedialog.askopenfilename(
            title="Select WAV File", filetypes=[("WAV files", "*.wav"), ("All files", "*.*")])
        if not file_path:
            return

        self._log(f"Loading WAV file: {os.path.basename(file_path)}")

        try:
            audio_data, sr = sf.read(file_path, dtype='float32')
            self._log(f"Loaded: {audio_data.shape}, sample rate: {sr} Hz")

            if len(audio_data.shape) == 1:
                waveform = audio_data
            else:
                waveform = audio_data.mean(axis=1)
                self._log("Converted to mono")

            waveform = self._prepare_waveform_for_model(waveform, sr)

            if waveform.shape[1] != TARGET_SAMPLES:
                self._log(f"Audio could not be normalized to {TARGET_SAMPLES} samples, cannot process", "WARNING")
                messagebox.showwarning("Warning", f"Audio segment too short.\nGot {waveform.shape[1]} samples, need {TARGET_SAMPLES}.")
                return

            self._log("Preprocessing audio...")
            mel_spec = self.preprocessor.process_audio(waveform)
            self._log(f"Mel-spectrogram shape: {mel_spec.shape}")

            if self.save_mel_specs:
                self._save_debug_mel_spec(mel_spec, f"test_{os.path.basename(file_path)}")

            self._log("Running inference...")
            with torch.no_grad():
                input_tensor = mel_spec.unsqueeze(0).to(self.device)
                output = self.model(input_tensor)
                if output.dim() == 1:
                    output = output.unsqueeze(0)
                probabilities = F.softmax(output, dim=1)
                top5_probs, top5_indices = torch.topk(probabilities[0], 5)

            predicted_key = self.key_labels[top5_indices[0].item()]
            confidence = top5_probs[0].item() * 100
            self._log(f"Predicted: '{predicted_key}' (confidence: {confidence:.2f}%)")

            self._update_detection_display(predicted_key, confidence, top5_indices, top5_probs)
            self.history_text.insert(
                tk.END,
                f"[{datetime.now().strftime('%H:%M:%S')}] '{predicted_key}' ({confidence:.1f}%) - {os.path.basename(file_path)}\n")
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
        if not self.model_loaded:
            messagebox.showwarning("Warning", "Please load a model first")
            return
        if not self.is_recording:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._log("Starting recording...")
        if self.recorder.start_recording():
            self.is_recording = True
            self.stop_processing = False
            self.start_time = time.time()
            self.record_btn.config(text="⏹ Stop Recording", bg='#e74c3c')
            self.status_bar.config(text="Recording...")
            self._log("Recording started")
            self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processing_thread.start()
        else:
            self._log("Failed to start recording", "ERROR")

    def _stop_recording(self):
        self._log("Stopping recording...")
        self.stop_processing = True
        self.recorder.stop_recording()
        self.is_recording = False
        self.record_btn.config(text="🎤 Start Recording", bg='#27ae60')
        self.status_bar.config(text="Recording stopped")
        self._log("Recording stopped")

    def _processing_loop(self):
        self._log("Processing loop started with advanced detection")
        last_process_time = time.time()

        while not self.stop_processing:
            current_time = time.time()

            if current_time - last_process_time >= 0.10:
                try:
                    keystrokes = self.recorder.detect_keystrokes()

                    if keystrokes:
                        self._log(f"Detected {len(keystrokes)} keystroke(s) in buffer")

                    for keystroke in keystrokes:
                        try:
                            self._log(f"Processing keystroke: {len(keystroke)} samples ({len(keystroke)/self.recorder.sample_rate:.4f}s)")

                            waveform = self._prepare_waveform_for_model(keystroke, self.recorder.sample_rate)

                            mel_spec = self.preprocessor.process_audio(waveform)

                            if mel_spec.shape != torch.Size([1, 64, 64]):
                                self._log(f"Invalid mel-spec shape: {mel_spec.shape}, expected [1, 64, 64]", "WARNING")
                                continue

                            if self.save_mel_specs:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                self._save_debug_mel_spec(mel_spec, f"live_keystroke_{timestamp}")
                                sf.write(str(self.debug_dir / f"live_keystroke_{timestamp}.wav"), np.asarray(keystroke, dtype=np.float32), self.recorder.sample_rate)
                                self._log(f"Saved debug files: {timestamp}", "DEBUG")

                            with torch.no_grad():
                                input_tensor = mel_spec.unsqueeze(0).to(self.device)
                                output = self.model(input_tensor)
                                if output.dim() == 1:
                                    output = output.unsqueeze(0)
                                probabilities = F.softmax(output, dim=1)
                                top5_probs, top5_indices = torch.topk(probabilities[0], 5)

                            predicted_key = self.key_labels[top5_indices[0].item()]
                            confidence = top5_probs[0].item() * 100

                            if confidence > 10.0:
                                self._log(f"✓ Detected: '{predicted_key}' (confidence: {confidence:.1f}%)")
                                self.root.after(0, self._update_detection_display, predicted_key, confidence, top5_indices, top5_probs)
                                self.root.after(0, self._add_to_history, predicted_key, confidence)
                                self.keystroke_count += 1
                                self.detection_history.append(confidence)
                                self.root.after(0, self._update_statistics)
                            else:
                                self._log(f"Low confidence: '{predicted_key}' ({confidence:.1f}%) - skipped", "DEBUG")

                        except Exception as e:
                            self._log(f"Error processing keystroke: {e}", "ERROR")
                            import traceback
                            traceback.print_exc()

                    last_process_time = current_time

                except Exception as e:
                    self._log(f"Error in detection loop: {e}", "ERROR")
                    import traceback
                    traceback.print_exc()

            if self.is_recording:
                elapsed = int(current_time - self.start_time)
                self.root.after(0, self._update_recording_time, elapsed)

            time.sleep(0.05)

    def _update_detection_display(self, predicted_key, confidence, top5_indices, top5_probs):
        self.current_key_label.config(text=predicted_key.upper())
        self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")

        for i in range(5):
            key = self.key_labels[top5_indices[i].item()]
            conf = top5_probs[i].item()
            self.prediction_labels[i]['key'].config(text=key.upper())
            self.prediction_labels[i]['conf'].config(text=f"{conf*100:.2f}%")
            canvas = self.prediction_labels[i]['bar']
            canvas.delete('all')
            bar_width = int(190 * conf)
            color = '#27ae60' if i == 0 else '#3498db'
            canvas.create_rectangle(5, 5, 5 + bar_width, 15, fill=color, outline='')

    def _add_to_history(self, key, confidence):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.history_text.insert(tk.END, f"[{timestamp}] {key.upper()} ({confidence:.1f}%)\n")
        self.history_text.see(tk.END)

    def _update_statistics(self):
        self.stats_labels['keystrokes'].config(text=str(self.keystroke_count))
        if self.detection_history:
            avg_conf = sum(self.detection_history) / len(self.detection_history)
            self.stats_labels['avg_conf'].config(text=f"{avg_conf:.1f}%")

    def _update_recording_time(self, elapsed):
        mins, secs = divmod(elapsed, 60)
        self.stats_labels['time'].config(text=f"{mins:02d}:{secs:02d}")

    def _clear_results(self):
        self.current_key_label.config(text="-")
        self.confidence_label.config(text="Confidence: -")
        self.history_text.delete('1.0', tk.END)
        self.keystroke_count = 0
        self.detection_history = []
        self._update_statistics()
        self._log("Results cleared")

    def _toggle_save_mel(self):
        self.save_mel_specs = self.save_mel_var.get()
        if self.save_mel_specs:
            self._log(f"Debug files will be saved to: {self.debug_dir}")
        else:
            self._log("Debug file saving disabled")

    def _update_threshold(self, value):
        threshold = float(value)
        self.threshold_label.config(text=f"{threshold:.2f}")
        self.recorder.adjust_threshold(threshold)

    def _save_debug_mel_spec(self, mel_spec, filename):
        try:
            plt.figure(figsize=(8, 6))
            plt.imshow(mel_spec.squeeze().numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Log Magnitude')
            plt.title(f'Mel Spectrogram - {filename}')
            plt.xlabel('Time')
            plt.ylabel('Mel Frequency')
            plt.tight_layout()
            plt.savefig(str(self.debug_dir / f"{filename}.png"), dpi=100, bbox_inches='tight')
            plt.close()
        except Exception as e:
            self._log(f"Error saving mel-spec: {e}", "ERROR")

    def on_closing(self):
        if self.is_recording:
            self._stop_recording()
        self.root.destroy()


def main():
    root = tk.Tk()
    app = KeystrokeDetectionUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()