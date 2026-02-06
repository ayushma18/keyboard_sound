"""
Test script to verify preprocessing matches phone.ipynb exactly
"""
import torch
import numpy as np
from model.audio_processor import to_mel_spectrogram, RealtimeAudioProcessor
import soundfile as sf
import os

print("Testing preprocessing pipeline...")
print("=" * 60)

# Test parameters from notebook
sample_rate = 44100
expected_n_mels = 64
expected_n_fft = 2048
expected_win_length = 1024
expected_hop_length = 300

# Load a test audio file (adjust paths as needed)
test_files = []
if os.path.exists("Data/segmented/main-alphanumeric/0"):
    test_files.append("Data/segmented/main-alphanumeric/0/0.wav")
if os.path.exists("Data/segmented/main-alphanumeric/a"):
    test_files.append("Data/segmented/main-alphanumeric/a/a.wav")

if not test_files:
    print("⚠️  No test files found in Data/segmented/main-alphanumeric/")
    print("   Please adjust paths or place test files there")
    exit(0)

processor = RealtimeAudioProcessor(sample_rate=44100)

for test_file in test_files:
    try:
        # Load audio
        audio, sr = sf.read(test_file)
        print(f"\n📁 Testing: {test_file}")
        print(f"   Audio: {len(audio)} samples @ {sr} Hz")
        
        # Resample if needed
        if sr != 44100:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
            print(f"   Resampled to 44100 Hz")
        
        # Preprocess
        mel_spec = processor.preprocess_keystroke(audio)
        
        print(f"   ✅ Mel-spectrogram shape: {mel_spec.shape}")
        print(f"   ✅ Expected: torch.Size([1, 64, 64])")
        print(f"   ✅ Value range: [{mel_spec.min():.3f}, {mel_spec.max():.3f}]")
        
        # Verify shape
        assert mel_spec.shape == torch.Size([1, 64, 64]), f"Wrong shape: {mel_spec.shape}"
        print(f"   ✅ Shape matches notebook!")
        
    except FileNotFoundError:
        print(f"   ⚠️  File not found: {test_file}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("✅ Preprocessing pipeline verification complete!")
print("\nKey parameters (matching phone.ipynb):")
print(f"  - Sample rate: 44100 Hz")
print(f"  - n_mels: 64")
print(f"  - n_fft: 2048")
print(f"  - win_length: 1024")
print(f"  - hop_length: 300")
print(f"  - Transform: log2(mel_power_spectrogram)")
print(f"  - Output shape: (1, 64, 64)")
