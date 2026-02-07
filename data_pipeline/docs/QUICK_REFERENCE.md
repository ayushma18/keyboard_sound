# Quick Reference Card 📋

## Start Application
```bash
python main.py
# or
run.bat  # Windows
./run.sh # Linux/Mac
```

---

## Tab Shortcuts

| Tab | Icon | Purpose | Key Action |
|-----|------|---------|------------|
| 1 | 📹 | Data Collection | Start Recording → Type → Stop |
| 2 | ✂️ | Data Segmentation | Browse → Configure → Process |
| 3 | 🧹 | Data Cleanup | Launch Tool → Review → Delete |
| 4 | 📊 | Data Analyzer | Launch Tool → Load → Analyze |

---

## Quick Workflows

### 🎙️ Record New Dataset
1. **File → Settings** → Select devices → Calibrate noise
2. **Tab 1** → Enter session name → Start Recording
3. Type naturally
4. Stop Recording

### ✂️ Segment Recording
1. **Tab 2** → Browse session folder
2. Adjust parameters (duration: 0.43s recommended)
3. Enable peak centering ✅
4. Apply filtering (50-5000 Hz) ✅
5. Start Segmentation

### 🧹 Clean Dataset
1. **Tab 3** → Launch Cleanup Tool
2. Browse segmented folder
3. Set concentration threshold (0.3 default)
4. Mark noisy files
5. Delete marked files

### 📊 Analyze Audio
1. **Tab 4** → Launch Analyzer
2. Load single file or folder
3. View waveform/spectrograms
4. Play audio (Space to play/pause)

---

## File Locations

```
recordings/
  └── mic1-kb1/
      ├── continuous/
      │   └── session_20260204_143022/
      │       ├── audio.wav
      │       └── keystroke_log.csv
      └── segmented_20260204_150000/
          ├── metadata.csv
          └── [key folders]/
```

---

## Common Parameters

### Recording
- Sample Rate: **44100 Hz** ✅
- Channels: **2 (Stereo)** ✅
- Format: WAV (32-bit float)

### Segmentation
- Duration: **0.430s** (18963 samples)
- Pre-trigger: **0.1s**
- Post-trigger: **0.33s**
- Peak centering: **Enabled** ✅
- Filter: **50-5000 Hz** ✅

### Cleanup
- Concentration threshold: **0.3** (balanced)
- Lower = more lenient
- Higher = more strict

---

## Keyboard Shortcuts (Analyzer)

| Key | Action |
|-----|--------|
| `Space` | Play/Pause |
| `Esc` | Stop |
| `←` | Back 0.1s |
| `→` | Forward 0.1s |

---

## Troubleshooting Quick Fixes

| Problem | Solution |
|---------|----------|
| No devices | Check connections, restart app |
| Choppy audio | Already fixed with pygame! |
| No keystrokes | Run as administrator |
| Everything is noise | Lower thresholds |

---

## Settings (File → Settings)

✅ Select input/output devices  
✅ Calibrate background noise (2 seconds)  
✅ Set sample rate (44100 Hz)  
✅ Set channels (2 stereo)  

---

## Best Practices

1. ✅ **Calibrate before recording** - File → Settings
2. ✅ **Use peak centering** - Tab 2
3. ✅ **Apply filtering** - 50-5000 Hz
4. ✅ **Review with cleanup** - Tab 3
5. ✅ **Keep consistent setup** - Same mic position

---

## Quick Commands

### Install pygame (if missing)
```bash
pip install pygame
```

### Reinstall all dependencies
```bash
pip install -r requirements.txt
```

### Check audio devices
```bash
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

## Module Import (for scripting)

```python
from config import Config
from audio_handler import AudioHandler
from keystroke_logger import KeystrokeLogger

# Initialize
config = Config()
audio = AudioHandler(44100, 2)
logger = KeystrokeLogger()
```

---

## Files Overview

| File | Purpose |
|------|---------|
| `main.py` | Main application |
| `config.py` | Configuration |
| `audio_handler.py` | Audio operations |
| `keystroke_logger.py` | Keyboard logging |
| `data_collector.py` | Recording tab |
| `data_segmenter.py` | Segmentation tab |
| `data_cleanup.py` | Cleanup tool |
| `analyze.py` | Analyzer tool |

---

## Common Errors

### "No module named 'pygame'"
```bash
pip install pygame
```

### "Could not find audio device"
- Check connections
- File → Settings → Select device

### "Permission denied"
- Run as administrator
- Check antivirus

---

## Support

📖 **Full Documentation**:
- [README.md](README.md) - Overview
- [USER_GUIDE.md](USER_GUIDE.md) - Complete guide
- [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md) - Technical details

---

**Happy Recording! 🎹🎵**

*Print this card and keep it handy!*
