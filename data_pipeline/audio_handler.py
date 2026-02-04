"""
Audio handling module with efficient recording and playback.
Uses sounddevice for recording and pygame for smooth playback.
"""
import numpy as np
import sounddevice as sd
import soundfile as sf
from scipy import signal
from scipy.ndimage import gaussian_filter1d
import threading
import queue
from typing import Optional, Callable, Tuple, List
import pygame.mixer as mixer


class AudioHandler:
    """Handles all audio recording and playback operations."""
    
    def __init__(self, sample_rate: int = 44100, channels: int = 2):
        self.sample_rate = sample_rate
        self.channels = channels
        self.is_recording = False
        self.recording_thread = None
        self.audio_queue = queue.Queue()
        
        # Initialize pygame mixer for smooth playback
        try:
            mixer.init(frequency=sample_rate, size=-16, channels=channels)
        except:
            print("Warning: pygame mixer init failed, playback may be limited")
        
        # Noise profile
        self.noise_profile = None
        self.noise_reduction_enabled = True
        self.noise_reduction_strength = 1.5
    
    @staticmethod
    def get_audio_devices() -> Tuple[List[dict], List[dict]]:
        """Get available input and output audio devices."""
        devices = sd.query_devices()
        input_devices = []
        output_devices = []
        
        for idx, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'index': idx,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'default_samplerate': device['default_samplerate']
                })
            if device['max_output_channels'] > 0:
                output_devices.append({
                    'index': idx,
                    'name': device['name'],
                    'channels': device['max_output_channels'],
                    'default_samplerate': device['default_samplerate']
                })
        
        return input_devices, output_devices
    
    def set_device(self, input_device: Optional[int] = None, 
                   output_device: Optional[int] = None) -> None:
        """Set audio devices."""
        if input_device is not None:
            sd.default.device[0] = input_device
        if output_device is not None:
            sd.default.device[1] = output_device
    
    def set_channels(self, channels: int) -> None:
        """Update channel configuration."""
        self.channels = channels
        # Reinitialize pygame mixer with new channel count
        try:
            mixer.quit()
            mixer.init(frequency=self.sample_rate, size=-16, channels=channels)
        except Exception as e:
            print(f"Warning: pygame mixer reinit failed: {e}")
    
    def set_sample_rate(self, sample_rate: int) -> None:
        """Update sample rate configuration."""
        self.sample_rate = sample_rate
        # Reinitialize pygame mixer with new sample rate
        try:
            mixer.quit()
            mixer.init(frequency=sample_rate, size=-16, channels=self.channels)
        except Exception as e:
            print(f"Warning: pygame mixer reinit failed: {e}")
    
    def get_device_info(self, device_index: int) -> dict:
        """Get information about a specific device."""
        devices = sd.query_devices()
        if 0 <= device_index < len(devices):
            return devices[device_index]
        return {}
    
    def get_audio_level(self, audio: np.ndarray) -> float:
        """Calculate RMS audio level as percentage."""
        return np.sqrt(np.mean(audio**2)) * 100
    
    def calibrate_noise(self, duration: float = 2.0) -> bool:
        """Calibrate background noise profile."""
        try:
            print(f"Calibrating noise for {duration} seconds...")
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=self.channels, 
                          dtype='float32')
            sd.wait()
            
            # Compute noise profile (average spectrum)
            if len(audio.shape) == 2 and audio.shape[1] == 2:
                mono = np.mean(audio, axis=1)
            else:
                mono = audio
            
            # Compute FFT of noise
            noise_fft = np.fft.rfft(mono)
            self.noise_profile = np.abs(noise_fft)
            print("Noise calibration complete")
            return True
            
        except Exception as e:
            print(f"Noise calibration failed: {e}")
            return False
    
    def apply_noise_reduction(self, audio: np.ndarray) -> np.ndarray:
        """Apply spectral subtraction noise reduction."""
        if not self.noise_reduction_enabled or self.noise_profile is None:
            return audio
        
        try:
            original_shape = audio.shape
            is_stereo = len(original_shape) == 2 and original_shape[1] == 2
            
            if is_stereo:
                # Process each channel
                processed = np.zeros_like(audio)
                for ch in range(2):
                    processed[:, ch] = self._reduce_noise_channel(audio[:, ch])
                return processed
            else:
                return self._reduce_noise_channel(audio)
                
        except Exception as e:
            print(f"Noise reduction failed: {e}")
            return audio
    
    def _reduce_noise_channel(self, channel: np.ndarray) -> np.ndarray:
        """Reduce noise for a single channel."""
        # FFT
        fft = np.fft.rfft(channel)
        magnitude = np.abs(fft)
        phase = np.angle(fft)
        
        # Match noise profile length
        if len(self.noise_profile) != len(magnitude):
            noise_profile = np.interp(
                np.linspace(0, 1, len(magnitude)),
                np.linspace(0, 1, len(self.noise_profile)),
                self.noise_profile
            )
        else:
            noise_profile = self.noise_profile
        
        # Spectral subtraction
        reduced_magnitude = magnitude - (noise_profile * self.noise_reduction_strength)
        reduced_magnitude = np.maximum(reduced_magnitude, 0.0)
        
        # Reconstruct
        fft_reduced = reduced_magnitude * np.exp(1j * phase)
        return np.fft.irfft(fft_reduced, n=len(channel))
    
    def apply_bandpass_filter(self, audio: np.ndarray, 
                             lowcut: float = 50, highcut: float = 5000) -> np.ndarray:
        """Apply bandpass filter to audio."""
        try:
            nyquist = self.sample_rate / 2
            low = max(lowcut / nyquist, 0.001)
            high = min(highcut / nyquist, 0.999)
            
            if low >= high:
                return audio
            
            sos = signal.butter(4, [low, high], btype='band', output='sos')
            
            if len(audio.shape) == 2 and audio.shape[1] == 2:
                filtered = np.zeros_like(audio)
                for ch in range(2):
                    filtered[:, ch] = signal.sosfilt(sos, audio[:, ch])
                return filtered
            else:
                return signal.sosfilt(sos, audio)
                
        except Exception as e:
            print(f"Filter failed: {e}")
            return audio
    
    def play_audio(self, audio: np.ndarray, blocking: bool = False) -> bool:
        """Play audio using pygame mixer for smooth playback."""
        try:
            # Ensure correct format
            if len(audio.shape) == 1:
                audio = np.stack([audio, audio], axis=1)
            
            # Normalize to int16
            audio = np.clip(audio, -1.0, 1.0)
            audio_int16 = (audio * 32767).astype(np.int16)
            
            # Create sound object
            sound = mixer.Sound(buffer=audio_int16)
            sound.play()
            
            if blocking:
                while mixer.get_busy():
                    pass
            
            return True
            
        except Exception as e:
            print(f"Playback failed: {e}")
            # Fallback to sounddevice
            try:
                sd.play(audio, self.sample_rate)
                if blocking:
                    sd.wait()
                return True
            except:
                return False
    
    def stop_playback(self) -> None:
        """Stop all audio playback."""
        try:
            mixer.stop()
        except:
            pass
        try:
            sd.stop()
        except:
            pass
    
    def record_stream(self, duration: float, 
                     callback: Optional[Callable[[np.ndarray], None]] = None) -> np.ndarray:
        """Record audio for specified duration with optional callback."""
        try:
            audio = sd.rec(int(duration * self.sample_rate), 
                          samplerate=self.sample_rate, 
                          channels=self.channels, 
                          dtype='float32')
            
            if callback:
                # Monitor recording in background
                def monitor():
                    while sd.get_stream().active:
                        if callback:
                            callback(audio)
                
                monitor_thread = threading.Thread(target=monitor, daemon=True)
                monitor_thread.start()
            
            sd.wait()
            return audio
            
        except Exception as e:
            print(f"Recording failed: {e}")
            return np.zeros((int(duration * self.sample_rate), self.channels))
    
    def save_audio(self, filepath: str, audio: np.ndarray) -> bool:
        """Save audio to WAV file."""
        try:
            sf.write(filepath, audio, self.sample_rate)
            return True
        except Exception as e:
            print(f"Save failed: {e}")
            return False
    
    def load_audio(self, filepath: str) -> Tuple[Optional[np.ndarray], int]:
        """Load audio from WAV file."""
        try:
            audio, sr = sf.read(filepath, dtype='float32')
            return audio, sr
        except Exception as e:
            print(f"Load failed: {e}")
            return None, 0
    
    def get_audio_level(self, audio: np.ndarray) -> float:
        """Calculate RMS level of audio."""
        return np.sqrt(np.mean(audio ** 2))
    
    def detect_peak(self, audio: np.ndarray, threshold: float = 0.1) -> Optional[int]:
        """Detect main peak in audio signal."""
        try:
            # Convert to mono if stereo
            if len(audio.shape) == 2:
                mono = np.mean(np.abs(audio), axis=1)
            else:
                mono = np.abs(audio)
            
            # Smooth
            smoothed = gaussian_filter1d(mono, sigma=10)
            
            # Find peak
            peak_idx = np.argmax(smoothed)
            
            if smoothed[peak_idx] > threshold:
                return peak_idx
            return None
            
        except Exception as e:
            print(f"Peak detection failed: {e}")
            return None
