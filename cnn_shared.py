"""
Shared classes and helpers used by CNN.ipynb and CNN-HyperparamTuning.ipynb.
"""

import os
import random

import numpy as np
import torch
import torch.nn as nn
import torchaudio
import soundfile as sf
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor
from torchaudio.transforms import TimeMasking, FrequencyMasking  # re-exported for convenience


# ── Audio helpers ─────────────────────────────────────────────────────────────

def calculate_rms(waveform):
    return torch.sqrt(torch.mean(waveform ** 2))


def add_noise_snr(signal, noise, target_snr_db):
    signal_rms = calculate_rms(signal)
    noise_rms  = calculate_rms(noise)
    snr_linear = 10 ** (target_snr_db / 20.0)
    scaling_factor = (signal_rms / snr_linear) / (noise_rms + 1e-10)
    return signal + noise * scaling_factor


def add_white_noise_snr(signal, target_snr_db):
    return add_noise_snr(signal, torch.randn_like(signal), target_snr_db)


def add_gaussian_noise_snr(signal, target_snr_db):
    return add_noise_snr(signal, torch.normal(mean=0, std=0.1, size=signal.shape), target_snr_db)


def reduce_volume_db(signal, db_reduction):
    return signal * (10 ** (-db_reduction / 20.0))


# ── Noise library ─────────────────────────────────────────────────────────────

class NoiseLibrary:
    def __init__(self, noise_dir):
        self.noise_dir     = noise_dir
        self.noises        = {'city': [], 'classroom': [], 'office': [], 'common': []}
        self.noise_metadata = []
        self._load()

    def _load(self):
        categories = {
            'city': 'city', 'classroom': 'classroom',
            'office': 'office', 'Common Noises': 'common'
        }
        for folder, key in categories.items():
            path = os.path.join(self.noise_dir, folder)
            if not os.path.exists(path):
                print(f"Warning: {path} not found")
                continue
            for fname in os.listdir(path):
                if not fname.endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    continue
                fpath = os.path.join(path, fname)
                try:
                    try:
                        waveform, sr = torchaudio.load(fpath)
                    except Exception:
                        data, sr = sf.read(fpath)
                        waveform = torch.from_numpy(data).float()
                        waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.T
                    if waveform.shape[0] > 1:
                        waveform = torch.mean(waveform, dim=0, keepdim=True)
                    if sr != 44100:
                        waveform = torchaudio.transforms.Resample(sr, 44100)(waveform)
                    rms = torch.sqrt(torch.mean(waveform ** 2)).item()
                    info = {'category': key, 'waveform': waveform,
                            'length': waveform.shape[1], 'rms': rms, 'filename': fname}
                    self.noises[key].append(info)
                    self.noise_metadata.append(info)
                except Exception as e:
                    print(f"Error loading {fpath}: {e}")

        total = len(self.noise_metadata)
        print(f"\n✅ Loaded {total} noise samples:")
        for cat, noises in self.noises.items():
            if noises:
                avg_len = np.mean([n['length'] for n in noises])
                avg_rms = np.mean([n['rms'] for n in noises])
                print(f"  - {cat}: {len(noises)} samples | Avg length: {avg_len/16000:.2f}s | Avg RMS: {avg_rms:.4f}")

    def get_random_noise(self, category=None):
        if category and self.noises.get(category):
            return random.choice(self.noises[category])
        elif self.noise_metadata:
            return random.choice(self.noise_metadata)
        return None

    def get_noise_segment(self, target_length, category=None):
        info = self.get_random_noise(category)
        if info is None:
            return None
        w = info['waveform']
        n = w.shape[1]
        if n >= target_length:
            start = random.randint(0, n - target_length)
            return w[:, start:start + target_length]
        return w.repeat(1, (target_length // n) + 1)[:, :target_length]


# ── Augmentation ──────────────────────────────────────────────────────────────

class AggressiveMultiVariationAugmentation:
    def __init__(self, noise_library, snr_range=(5, 30),
                 volume_reduction_prob=0.4, volume_reduction_range=(2, 15),
                 multi_aug_prob=0.3):
        self.noise_library          = noise_library
        self.snr_range              = snr_range
        self.volume_reduction_prob  = volume_reduction_prob
        self.volume_reduction_range = volume_reduction_range
        self.multi_aug_prob         = multi_aug_prob
        self.strategies = [
            'real_city', 'real_office', 'real_classroom', 'real_common',
            'white', 'gaussian', 'mixed_real_white', 'mixed_real_gaussian', 'real_random',
        ]

    def apply_augmentation(self, signal, strategy, snr):
        signal_length = signal.shape[1] if signal.dim() > 1 else signal.shape[0]
        _cat = {'real_city': 'city', 'real_office': 'office',
                'real_classroom': 'classroom', 'real_common': 'common',
                'real_random': None}

        if strategy in _cat or strategy in ('mixed_real_white', 'mixed_real_gaussian'):
            seg = self.noise_library.get_noise_segment(signal_length, _cat.get(strategy))
            if seg is not None:
                if seg.shape[0] != signal.shape[0]:
                    seg = seg[:signal.shape[0], :]
                signal = add_noise_snr(signal, seg, snr)
            if strategy == 'mixed_real_white':
                signal = add_white_noise_snr(signal, snr + 5)
            elif strategy == 'mixed_real_gaussian':
                signal = add_gaussian_noise_snr(signal, snr + 5)
        elif strategy == 'white':
            signal = add_white_noise_snr(signal, snr)
        elif strategy == 'gaussian':
            signal = add_gaussian_noise_snr(signal, snr)
        return signal

    def __call__(self, signal):
        signal     = signal.clone()
        target_snr = random.uniform(*self.snr_range)
        strategy   = random.choice(self.strategies)
        signal     = self.apply_augmentation(signal, strategy, target_snr)

        if random.random() < self.multi_aug_prob:
            second_snr      = random.uniform(max(target_snr, 10), 30)
            second_strategy = random.choice(['white', 'gaussian'])
            signal          = self.apply_augmentation(signal, second_strategy, second_snr)

        if random.random() < self.volume_reduction_prob:
            db = random.uniform(*self.volume_reduction_range)
            signal = reduce_volume_db(signal, db)

        return signal


class TimeShifting:
    def __init__(self, shift_ratio=0.3):
        self.shift_ratio = shift_ratio

    def __call__(self, samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.numpy()
        shift        = int(samples.shape[1] * self.shift_ratio)
        random_shift = random.randint(0, shift)
        data_roll    = np.zeros_like(samples)
        for ch in range(samples.shape[0]):
            data_roll[ch] = np.roll(samples[ch], random_shift)
        return torch.tensor(data_roll)


# ── Datasets ──────────────────────────────────────────────────────────────────

class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir  = data_dir
        self.transform = transform
        self.n_samples = 0
        self.dataset   = []
        self.labels    = set()
        self.label_map = {}
        self._load(data_dir)

    def _load(self, path):
        raw_labels = set()
        for dirname, _, filenames in os.walk(path):
            lbl = os.path.basename(dirname)
            if filenames and (('0' <= lbl <= '9') or ('a' <= lbl <= 'z')):
                raw_labels.add(lbl)
        for idx, lbl in enumerate(sorted(raw_labels)):
            self.label_map[lbl] = idx
        for dirname, _, filenames in os.walk(path):
            for fname in filenames:
                fpath = os.path.join(dirname, fname)
                lbl   = os.path.basename(dirname)
                if lbl not in self.label_map:
                    continue
                label_tensor = torch.tensor(self.label_map[lbl])
                self.labels.add(label_tensor.item())
                try:
                    waveform, _ = torchaudio.load(fpath)
                except Exception:
                    data, _ = sf.read(fpath)
                    waveform = torch.from_numpy(data).float()
                    waveform = waveform.unsqueeze(0) if waveform.ndim == 1 else waveform.T
                self.n_samples += 1
                self.dataset.append((waveform, label_tensor))

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        waveform, label = self.dataset[idx]
        if self.transform is not None:
            waveform = self.transform(waveform)
        return waveform, label

    def num_classes(self):
        return len(self.labels)


class TrainingDataset(Dataset):
    def __init__(self, base_dataset, transformations):
        super().__init__()
        self.base            = base_dataset
        self.transformations = transformations

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        waveform, label = self.base[idx]
        return self.transformations(waveform), label


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class CNN(nn.Module):
    """
    dropout_conv and dropout_fc are kept optional (default = original hardcoded values)
    so CNN.ipynb can continue to instantiate CNN() without any changes.
    """
    def __init__(self, num_classes=36, dropout_conv=0.25, dropout_fc=0.5):
        super().__init__()
        dc = dropout_conv
        self.conv1 = ConvBlock(1,   64);  self.pool1 = nn.MaxPool2d(2, 2); self.dropout1 = nn.Dropout2d(dc)
        self.conv2 = ConvBlock(64,  128); self.pool2 = nn.MaxPool2d(2, 2); self.dropout2 = nn.Dropout2d(dc)
        self.conv3 = ConvBlock(128, 256); self.pool3 = nn.MaxPool2d(2, 2); self.dropout3 = nn.Dropout2d(min(dc + 0.05, 0.5))
        self.conv4 = ConvBlock(256, 512); self.pool4 = nn.MaxPool2d(2, 2); self.dropout4 = nn.Dropout2d(min(dc + 0.05, 0.5))
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Dropout(dropout_fc),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(dropout_fc),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.dropout1(self.pool1(self.conv1(x)))
        x = self.dropout2(self.pool2(self.conv2(x)))
        x = self.dropout3(self.pool3(self.conv3(x)))
        x = self.dropout4(self.pool4(self.conv4(x)))
        x = self.global_pool(x)
        return self.fc(x)


# ── Utilities ─────────────────────────────────────────────────────────────────

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def get_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return 'cuda'
        except Exception as e:
            print(f"⚠️  CUDA available but unusable ({e}), falling back to CPU")
    return 'cpu'


# ── Mel spectrogram config ─────────────────────────────────────────────────────

from dataclasses import dataclass, field

@dataclass
class MelConfig:
    """
    All mel-spectrogram parameters in one place for systematic experimentation.
    Pass an instance to build_mel_transform() to get a ready-made transform pipeline.

    Fields:
        n_mels      – mel frequency bins           (higher = finer freq resolution)
        hop_length  – time step in samples         (lower  = finer time grid)
        n_fft       – FFT window size              (larger = finer freq, coarser time)
        win_length  – analysis window length       (shorter = better transient capture)
        f_min       – lowest frequency in Hz
        f_max       – highest frequency in Hz
        power       – 2.0 = power spectrogram, 1.0 = magnitude spectrogram
        sample_rate – audio sample rate (must match training data)
    """
    n_mels      : int   = 128
    hop_length  : int   = 256
    n_fft       : int   = 1024
    win_length  : int   = 512
    f_min       : float = 50.0
    f_max       : float = 16000.0
    power       : float = 2.0
    sample_rate : int   = 44100

    def label(self) -> str:
        return (f"mels={self.n_mels} hop={self.hop_length} "
                f"fft={self.n_fft} win={self.win_length} "
                f"f=[{int(self.f_min)},{int(self.f_max)}] p={self.power}")


def build_mel_transform(cfg: MelConfig):
    """
    Return a Compose pipeline: waveform Tensor → (1, n_mels, T) Tensor in dB scale.

    Example::
        cfg = MelConfig(n_mels=256, hop_length=128)
        tfm = build_mel_transform(cfg)
        spec = tfm(waveform)   # shape: (1, 256, T)
    """
    to_mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=cfg.sample_rate,
        n_mels=cfg.n_mels,
        hop_length=cfg.hop_length,
        n_fft=cfg.n_fft,
        win_length=cfg.win_length,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
        power=cfg.power,
    )
    to_db_numpy = lambda s: (10 * s.clamp(min=1e-10).log10())[0, :, :].numpy()
    return Compose([to_mel, to_db_numpy, ToTensor()])
