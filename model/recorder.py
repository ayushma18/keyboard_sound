"""
Real-time Audio Recorder
Handles continuous audio recording with background processing
"""

import sounddevice as sd
import numpy as np
import threading
import queue
import time


class RealtimeRecorder:
    """
    Record audio in real-time with background processing
    """
    
    def __init__(self, sample_rate=44100, channels=1, dtype='float32'):
        """
        Initialize the recorder
        
        Args:
            sample_rate: Audio sample rate (44100 Hz to match notebook training data)
            channels: Number of audio channels (1 for mono)
            dtype: Data type for audio samples
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.device_id = None  # Audio device to use (None = default)
        
        # Recording state
        self.is_recording = False
        self.recording_thread = None
        self.stream = None
        
        # Audio data queue
        self.audio_queue = queue.Queue()
        
        # Callbacks
        self.on_audio_callback = None
    
    def set_audio_callback(self, callback):
        """
        Set callback function to process audio chunks
        
        Args:
            callback: Function that takes audio chunk as argument
        """
        self.on_audio_callback = callback
    
    def _audio_callback(self, indata, frames, time_info, status):
        """
        Internal callback for sounddevice stream
        
        Args:
            indata: Input audio data
            frames: Number of frames
            time_info: Timing information
            status: Stream status
        """
        if status:
            print(f"Audio stream status: {status}")
        
        # Copy data and put in queue
        audio_chunk = indata.copy()
        self.audio_queue.put(audio_chunk)
    
    def _processing_loop(self):
        """Background thread loop for processing audio"""
        while self.is_recording:
            try:
                # Get audio chunk from queue with timeout
                audio_chunk = self.audio_queue.get(timeout=0.1)
                
                # Call user callback if set
                if self.on_audio_callback:
                    self.on_audio_callback(audio_chunk)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing loop: {e}")
    
    def start_recording(self):
        """Start recording audio"""
        if self.is_recording:
            print("Already recording")
            return False
        
        try:
            # Create input stream
            self.stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=self.dtype,
                callback=self._audio_callback,
                blocksize=2400  # Match the before_samples from isolator
            )
            
            # Start stream
            self.stream.start()
            self.is_recording = True
            
            # Start processing thread
            self.recording_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.recording_thread.start()
            
            print("Recording started")
            return True
            
        except Exception as e:
            print(f"Error starting recording: {e}")
            self.is_recording = False
            return False
    
    def stop_recording(self):
        """Stop recording audio"""
        if not self.is_recording:
            print("Not recording")
            return False
        
        try:
            # Stop recording flag
            self.is_recording = False
            
            # Wait for processing thread to finish
            if self.recording_thread:
                self.recording_thread.join(timeout=2.0)
            
            # Stop and close stream
            if self.stream:
                self.stream.stop()
                self.stream.close()
                self.stream = None
            
            # Clear queue
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            print("Recording stopped")
            return True
            
        except Exception as e:
            print(f"Error stopping recording: {e}")
            return False
    
    def is_active(self):
        """Check if recording is active"""
        return self.is_recording
    
    def get_available_devices(self):
        """Get list of available audio input devices"""
        devices = sd.query_devices()
        input_devices = []
        
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append({
                    'id': i,
                    'name': device['name'],
                    'channels': device['max_input_channels'],
                    'sample_rate': device['default_samplerate']
                })
        
        return input_devices
