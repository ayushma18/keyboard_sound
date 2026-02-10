# 🎵 Hybrid Noise Augmentation - Quick Reference

## 📊 Augmentation Distribution

```
Real Ambient Noise: ████████████████████████████████ 55.0%
White Noise:        █████████████                     22.5%
Gaussian Noise:     ███████                           12.5%
Clean Audio:        ██████                            10.0%
```

## 🎚️ SNR Levels

| SNR (dB) | Description | Use Case |
|----------|-------------|----------|
| 5 dB | Very Noisy | Challenging conditions (loud office, cafe) |
| 10 dB | Realistic | Normal office/home environment |
| 15 dB | Moderate | Quiet office |
| 20 dB | Light Noise | Very quiet environment |

**Training Range:** 5-20 dB (randomly selected per sample)

## 🔊 Augmentation Parameters

### Noise Augmentation
- **Probability:** 90% (10% clean)
- **SNR Range:** 5-20 dB
- **Noise Types:** Real ambient, White, Gaussian

### Volume Reduction
- **Probability:** 25%
- **Reduction Range:** 3-12 dB
- **Purpose:** Simulates far microphone

### Time Shifting
- **Shift Range:** 0-50% of audio length
- **Previous:** 30% (increased by 66%)

## 🎯 Code Snippets

### Apply Noise to Single Sample
```python
# Real ambient noise at 10 dB SNR
noisy = add_noise_snr(
    signal, 
    noise_library.get_noise_segment(signal.shape[1]), 
    target_snr_db=10
)

# White noise at 15 dB SNR
noisy = add_white_noise_snr(signal, target_snr_db=15)

# Gaussian noise at 20 dB SNR
noisy = add_gaussian_noise_snr(signal, target_snr_db=20)

# Reduce volume by 10 dB
quieter = reduce_volume_db(signal, db_reduction=10)
```

### Modify Augmentation Probabilities
```python
hybrid_noise_aug = HybridNoiseAugmentation(
    noise_library=noise_library,
    real_noise_prob=0.60,      # 60% real noise
    white_noise_prob=0.20,     # 20% white noise
    gaussian_noise_prob=0.10,  # 10% Gaussian
    clean_prob=0.10,           # 10% clean
    snr_range=(8, 18),         # SNR 8-18 dB
    volume_reduction_prob=0.30,# 30% volume reduction
    volume_reduction_range=(5, 15)  # 5-15 dB reduction
)
```

### Change Time Shifting Range
```python
# More aggressive time shifting
TimeShifting(shift_ratio=0.6)  # 60% shift

# Less aggressive time shifting
TimeShifting(shift_ratio=0.4)  # 40% shift
```

## 📂 Noise Library Structure

```
model/Noises/
├── city/           (3 samples)  - Traffic, urban sounds
├── classroom/      (13 samples) - Talking, footsteps
├── office/         (3 samples)  - Typing, printers
└── Common Noises/  (17 samples) - Fans, mouse clicks, breathing
```

**Total:** 36 noise samples, all converted to 16kHz mono

## 🧪 Testing Commands

### Test Noise Loading
```python
# Check loaded noises
print(f"Total noises: {len(noise_library.noise_metadata)}")
for cat, noises in noise_library.noises.items():
    print(f"{cat}: {len(noises)} samples")
```

### Test SNR Addition
```python
# Get test sample
waveform, label = init_train_set[0]

# Test real noise at 10 dB
noise_segment = noise_library.get_noise_segment(waveform.shape[1])
noisy = add_noise_snr(waveform, noise_segment, 10)

# Check RMS
print(f"Original RMS: {calculate_rms(waveform):.6f}")
print(f"Noisy RMS: {calculate_rms(noisy):.6f}")
```

### Visualize Augmentation Effect
```python
import matplotlib.pyplot as plt

# Original
plt.subplot(2, 1, 1)
plt.plot(waveform[0].numpy())
plt.title("Original")

# With noise
noisy = hybrid_noise_aug(waveform)
plt.subplot(2, 1, 2)
plt.plot(noisy[0].numpy())
plt.title("With Augmentation")
plt.tight_layout()
plt.show()
```

## 🔄 Pipeline Order

```
1. Raw Waveform (1D audio)
   ↓
2. Hybrid Noise Augmentation (Real/White/Gaussian + Volume Reduction)
   ↓
3. Time Shifting (50% range)
   ↓
4. Mel Spectrogram Conversion (2D representation)
   ↓
5. Frequency Masking (×2)
   ↓
6. Time Masking (×2)
   ↓
7. Final Augmented Spectrogram
```

## 🎓 Understanding SNR

### SNR Formula
```
SNR (dB) = 20 × log₁₀(Signal_RMS / Noise_RMS)
```

### Example Calculations

**Signal RMS = 0.1, Noise RMS = 0.01**
- SNR = 20 × log₁₀(0.1 / 0.01) = 20 × log₁₀(10) = 20 dB

**Signal RMS = 0.1, Noise RMS = 0.0316**
- SNR = 20 × log₁₀(0.1 / 0.0316) = 20 × log₁₀(3.16) = 10 dB

**Signal RMS = 0.1, Noise RMS = 0.0562**
- SNR = 20 × log₁₀(0.1 / 0.0562) = 20 × log₁₀(1.78) = 5 dB

## 🚨 Common Issues & Solutions

### Issue: Noise too loud/quiet
**Solution:** Adjust SNR range
```python
snr_range=(10, 25)  # Lighter noise
snr_range=(3, 15)   # Heavier noise
```

### Issue: Too much augmentation
**Solution:** Increase clean probability
```python
clean_prob=0.20  # 20% clean samples
```

### Issue: Model not learning
**Solution:** Reduce augmentation initially
```python
real_noise_prob=0.30  # Start lighter
snr_range=(15, 25)    # Easier SNR
```

### Issue: Not enough variety
**Solution:** Add more noise samples to Noises folder
- Ensure 16kHz sample rate
- Supported formats: .wav, .mp3, .flac, .ogg

## 📈 Tuning Guidelines

### For Better Accuracy
- ↑ Increase clean_prob to 15-20%
- ↑ Increase SNR range to (10, 25)
- ↓ Decrease volume reduction probability

### For Better Robustness
- ↑ Increase real_noise_prob to 60-70%
- ↓ Decrease SNR range to (3, 15)
- ↑ Increase volume reduction probability to 30-40%

### For Faster Training
- ↓ Use fewer augmentation types
- ↑ Higher SNR (easier samples)
- ↓ Remove volume reduction temporarily

## ✅ Validation Checklist

Before training:
- [ ] Noise library loads successfully (36 samples)
- [ ] SNR functions work correctly
- [ ] Hybrid augmentation initializes
- [ ] Test sample shows proper augmentation
- [ ] RMS values are reasonable (0.01-0.3 range)

During training:
- [ ] Training loss decreases steadily
- [ ] Validation accuracy improves
- [ ] No NaN or Inf values
- [ ] Model generalizes well

After training:
- [ ] Test on noisy real-world samples
- [ ] Compare with baseline (no augmentation)
- [ ] Verify robustness in different environments

## 🎉 Ready to Train!

Your augmentation pipeline is configured and ready. Start training with:
```python
# Train with augmentation
train_dataloader = DataLoader(train_set, batch_size=32, shuffle=True)

# Monitor both augmented and non-augmented performance
train_dataloader_no_aug = DataLoader(train_set_no_aug, batch_size=32, shuffle=True)
```

Good luck! 🚀
