"""
Audio Processing Module
Handles audio segmentation and preprocessing for keystroke detection
EXACTLY matching phone.ipynb preprocessing pipeline
"""

import numpy as np
import librosa
import torch
from collections import deque
import threading


def to_mel_spectrogram(samples, sample_rate=44100):
    """
    Convert audio samples to mel spectrogram
    EXACTLY matching notebook: torchaudio.transforms.MelSpectrogram + log2()
    
    Notebook transform:
    to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=64, hop_length=300, n_fft=2048, win_length=1024)
    mel_spectrogram_to_numpy = lambda spectrogram: spectrogram.log2()[0,:,:].numpy()
    
    Args:
        samples: Audio samples array (numpy)
        sample_rate: Sample rate (default 44100 Hz)
    
    Returns:
        Mel spectrogram with log2 transformation (64 x time_frames)
    """
    # librosa.feature.melspectrogram returns power spectrogram (like torchaudio)
    # Parameters MUST match notebook exactly
    mel_power_spec = librosa.feature.melspectrogram(
        y=samples, 
        sr=sample_rate, 
        n_mels=64,
        n_fft=2048,
        win_length=1024,
        hop_length=300
    )
    
    # Apply log2 to power spectrogram (matching notebook: spectrogram.log2())
    mel_log2 = np.log2(mel_power_spec + 1e-9)  # Add epsilon to avoid log(0)
    
    return mel_log2


class RealtimeAudioProcessor:
    """
    Process audio in real-time, detecting and segmenting keystrokes
    Preprocessing matches phone.ipynb exactly
    """
    
    def __init__(self, sample_rate=44100, buffer_duration=1.0):
        """
        Initialize the processor
        
        Args:
            sample_rate: Audio sample rate (44100 Hz matching notebook)
            buffer_duration: Duration of audio buffer in seconds
        """
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.buffer_size = int(sample_rate * buffer_duration)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.lock = threading.Lock()
        
        # Detection parameters
        self.pre_trigger_samples = int(0.1 * self.sample_rate)  # 4410 samples at 44100Hz
        self.post_trigger_samples = int(0.33 * self.sample_rate)  # 14553 samples at 44100Hz
        
        # For isolator (energy detection)
        self.fft_size = 48
        self.hop_length = 24
        self.threshold = 0.06
        
        # Optional filtering (disabled by default to match notebook)
        self.filter_enabled = False
        self.filter_low = 50
        self.filter_high = 5000
        
        # Track processed timestamps to avoid duplicates
        self.last_keystroke_time = -1.0
        self.min_keystroke_spacing = 0.1  # seconds
    
    def add_audio_data(self, audio_chunk):
        """
        Add new audio data to the buffer
        
        Args:
            audio_chunk: Numpy array of audio samples
        """
        with self.lock:
            self.audio_buffer.extend(audio_chunk.flatten())
    
    def get_buffer_as_array(self):
        """Get current buffer as numpy array"""
        with self.lock:
            return np.array(self.audio_buffer, dtype=np.float32)
    
    def apply_bandpass_filter(self, audio_data):
        """
        Apply bandpass filter (50-5000 Hz)
        
        Args:
            audio_data: Audio signal array
        
        Returns:
            Filtered audio signal
        """
        from scipy import signal as scipy_signal
        
        # Design Butterworth bandpass filter
        nyquist = self.sample_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist
        
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        filtered = scipy_signal.filtfilt(b, a, audio_data)
        
        return filtered
    
    def isolator(self, signal, before, after, threshold):
        """
        Isolate keystrokes from continuous audio signal
        
        Args:
            signal: Audio signal array
            before: Samples to include before keystroke peak
            after: Samples to include after keystroke peak
            threshold: Energy threshold for keystroke detection
        
        Returns:
            List of isolated keystroke arrays
        """
        strokes = []
        
        # Compute FFT energy
        fft = librosa.stft(signal, n_fft=self.fft_size, hop_length=self.hop_length)
        energy = np.abs(np.sum(fft, axis=0)).astype(float)
        
        # Find peaks above threshold
        threshed = energy > threshold
        peaks = np.where(threshed)[0]
        peak_count = len(peaks)
        
        # Extract keystrokes with minimum spacing
        prev_end = self.sample_rate * 0.1 * (-1)
        for i in range(peak_count):
            this_peak = peaks[i]
            timestamp = (this_peak * self.hop_length) + self.fft_size // 2
            
            # Ensure minimum spacing between keystrokes
            if timestamp > prev_end + (0.1 * self.sample_rate):
                keystroke = signal[timestamp - before:timestamp + after]
                if len(keystroke) > 0:
                    strokes.append(keystroke)
                prev_end = timestamp + after
        
        return strokes
    
    def detect_and_segment_keystrokes(self):
        """
        Detect and segment keystrokes from current buffer
        
        Returns:
            List of keystroke audio arrays
        """
        buffer_array = self.get_buffer_as_array()
        
        # Need at least minimum samples
        min_samples = self.pre_trigger_samples + self.post_trigger_samples
        if len(buffer_array) < min_samples:
            return []
        
        # Apply bandpass filter if enabled
        if self.filter_enabled:
            buffer_array = self.apply_bandpass_filter(buffer_array)
        
        # Use isolator to detect keystrokes with pre/post trigger
        strokes = self.isolator(
            buffer_array,
            self.pre_trigger_samples,
            self.post_trigger_samples,
            self.threshold
        )
        
        # Filter out invalid strokes
        valid_strokes = []
        for stroke in strokes:
            if len(stroke) >= 1000:  # At least 1000 samples
                valid_strokes.append(stroke)
        
        return valid_strokes
    
    def preprocess_keystroke(self, keystroke_audio):
        """
        Preprocess a single keystroke for model input
        EXACTLY matches notebook preprocessing:
        
        Notebook pipeline:
        1. torchaudio.load(file) -> waveform
        2. to_mel_spectrogram(waveform) -> mel spectrogram (power)
        3. spectrogram.log2()[0,:,:].numpy() -> log2 transform
        4. ToTensor() -> convert to torch tensor with channel dim
        5. Ensure shape is (1, 64, 64)
        
        Args:
            keystroke_audio: Numpy array or torch tensor of keystroke audio
        
        Returns:
            Preprocessed mel spectrogram tensor ready for model (1, 64, 64)
        """
        # Convert to numpy and flatten
        if isinstance(keystroke_audio, np.ndarray):
            samples = keystroke_audio.flatten()
        else:
            samples = keystroke_audio.numpy().flatten()
        
        # Apply bandpass filter if enabled (disabled by default to match notebook)
        if self.filter_enabled:
            samples = self.apply_bandpass_filter(samples)
        
        # Convert to mel spectrogram with log2 transformation
        # This matches: to_mel_spectrogram(waveform) -> spectrogram.log2()[0,:,:].numpy()
        mel_spec = to_mel_spectrogram(samples, self.sample_rate)
        
        # mel_spec shape is (64, time_frames)
        # Ensure time dimension is exactly 64
        mel_height, mel_width = mel_spec.shape
        
        if mel_width < 64:
            # Pad width to 64
            pad_width = 64 - mel_width
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='edge')
        elif mel_width > 64:
            # Crop to 64 (center crop to preserve keystroke)
            start = (mel_width - 64) // 2
            mel_spec = mel_spec[:, start:start + 64]
        
        # Height should always be 64 with n_mels=64, but verify
        if mel_height != 64:
            if mel_height < 64:
                pad_height = 64 - mel_height
                mel_spec = np.pad(mel_spec, ((0, pad_height), (0, 0)), mode='edge')
            else:
                mel_spec = mel_spec[:64, :]
        
        # Convert to tensor and add channel dimension: (64, 64) -> (1, 64, 64)
        # This matches: ToTensor() in notebook which adds channel dimension
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
        
        return mel_tensor
    
    def clear_buffer(self):
        """Clear the audio buffer"""
        with self.lock:
            self.audio_buffer.clear()
    
    def adjust_threshold(self, new_threshold):
        """
        Adjust the energy threshold for keystroke detection
        
        Args:
            new_threshold: New threshold value (typically 0.01-0.2)
        """
        self.threshold = new_threshold
    
    def set_filter_params(self, enabled, low_freq=50, high_freq=5000):
        """
        Configure bandpass filter parameters
        
        Args:
            enabled: Enable/disable filtering
            low_freq: Low cutoff frequency in Hz
            high_freq: High cutoff frequency in Hz
        """
        self.filter_enabled = enabled
        self.filter_low = low_freq
        self.filter_high = high_freq
