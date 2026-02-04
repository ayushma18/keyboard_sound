# Keyboard Acoustic Research Tool - Modular Edition

## 🎯 Overview

This is a **completely refactored** and **modular** version of the keyboard acoustic research tool. The application now uses an **OOP design** with separate modules for different functionalities, making it easier to maintain, extend, and understand.

## 🆕 What's New

### Major Changes:
1. **Continuous Recording**: Records audio continuously and logs keystrokes with timestamps (no more file-per-keystroke)
2. **Separate Segmentation**: Process recordings later with different parameters
3. **Modular Architecture**: Split into multiple files with clear responsibilities
4. **Better Audio Playback**: Uses pygame for smooth, non-choppy playback
5. **Tabbed Interface**: All tools in one application
6. **Improved Performance**: More efficient processing and threading

## 📁 Project Structure

```
keyboard_sound/
├── main.py                  # Main application entry point (NEW - tabbed interface)
├── main_old.py              # Backup of original monolithic version
├── config.py                # Configuration management
├── audio_handler.py         # Audio recording/playback (uses pygame)
├── keystroke_logger.py      # Keyboard event logging
├── data_collector.py        # Continuous recording tab
├── data_segmenter.py        # Segmentation tab
├── data_cleanup.py          # Data cleanup tool (integrated)
├── analyze.py               # Audio analyzer (integrated)
└── requirements.txt         # Updated dependencies (includes pygame)
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install pygame for better audio playback
pip install pygame

# Or reinstall all requirements
pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

## 📋 Tabs/Features

### 📹 Tab 1: Data Collection (Continuous Recording)
- **Record continuous audio** instead of individual files
- **Log keystrokes** with precise timestamps
- More flexible for later processing
- Saves:
  - `audio.wav` - Full continuous recording
  - `keystroke_log.csv` - Timestamp and key for each press
  - `session_info.txt` - Session metadata

**Workflow:**
1. Set microphone and keyboard IDs
2. Click "Start Recording"
3. Type naturally
4. Click "Stop Recording"
5. Data saved to `recordings/micX-kbY/continuous/session_name/`

### ✂️ Tab 2: Data Segmentation
- **Load continuous recordings**
- **Extract individual keystroke segments**
- Adjust parameters without re-recording:
  - Segment duration
  - Pre/post trigger times
  - Peak centering
  - Frequency filtering
- Create datasets with different configurations

**Workflow:**
1. Click "Browse Session Folder"
2. Select a continuous recording session
3. Adjust segmentation parameters
4. Click "Start Segmentation"
5. Segmented files saved in standard format

### 🧹 Tab 3: Data Cleanup
- Integrated version of `data_cleanup.py`
- Review and remove noisy recordings
- Energy concentration detection
- Bulk operations

### 📊 Tab 4: Data Analyzer
- Integrated version of `analyze.py`
- Waveform visualization
- Spectrograms (linear, log, mel)
- Audio playback with position indicator
- Batch analysis

## 🎵 Audio Improvements

### Smooth Playback
- **Uses pygame.mixer** for buffer-based playback
- No more choppy audio
- Better real-time performance

### Better Recording
- Continuous recording in 1-second chunks
- Proper threading
- Lower latency

## 🏗️ Architecture Benefits

### Modular Design (OOP)
- **Separation of Concerns**: Each module has one responsibility
- **Reusable Components**: AudioHandler, KeystrokeLogger can be used independently
- **Easy Testing**: Test individual modules
- **Better Maintenance**: Find and fix bugs faster

### Key Classes:
- `Config`: Centralized configuration management
- `AudioHandler`: All audio operations (recording, playback, processing)
- `KeystrokeLogger`: Keyboard event capture
- `DataCollectorTab`: Continuous recording UI
- `DataSegmenterTab`: Segmentation UI

## 📊 Comparison: Old vs New Workflow

### Old Workflow (main_old.py):
```
Start Recording → Press Key → Save File → Press Key → Save File → ...
```
- ❌ No flexibility to change parameters
- ❌ File I/O overhead on every keystroke
- ❌ Can't adjust segmentation after recording

### New Workflow (main.py):
```
1. Continuous Recording:
   Start Recording → Type naturally → Stop Recording
   Result: 1 audio file + 1 keystroke log

2. Segmentation (can run multiple times):
   Load Recording → Adjust Parameters → Segment
   Result: Individual keystroke files
```
- ✅ Record once, segment many times
- ✅ No file I/O during recording
- ✅ Adjust parameters without re-recording
- ✅ Faster, more flexible

## 🔧 Configuration

Settings are stored in `recording_config.json`:
```json
{
  "mic_id": "mic1",
  "keyboard_id": "kb1",
  "sample_rate": 44100,
  "channels": 2,
  "input_device": 0,
  "output_device": 1
}
```

Access via: **File → Settings**

## 📦 New Dependencies

- **pygame**: Smooth audio playback (non-choppy)
- Existing: sounddevice, soundfile, librosa, matplotlib, etc.

## 🐛 Known Issues Fixed

1. ✅ **Choppy audio playback** - Fixed with pygame
2. ✅ **Laggy UI** - Fixed with better threading
3. ✅ **Single file bottleneck** - Split into modules
4. ✅ **Hard to modify** - Now modular and extensible

## 🔮 Future Enhancements

Ideas for further improvement:
- [ ] Real-time waveform visualization during recording
- [ ] Automatic noise detection and removal
- [ ] Machine learning model training tab
- [ ] Export to different formats (HDF5, TFRecord)
- [ ] Cloud sync capabilities

## 📝 Migration Guide

If you have existing data from `main_old.py`:
1. Data format is compatible (WAV files + metadata CSV)
2. Can use **Data Cleanup** and **Data Analyzer** tabs directly
3. For new recordings, use the new **Data Collection** tab

## 💡 Tips

### For Best Results:
1. **Calibrate noise** before recording (Settings menu)
2. **Use peak centering** in segmentation (enabled by default)
3. **Apply bandpass filtering** (50-5000 Hz recommended)
4. **Review with cleanup tool** before training models

### Performance:
- Continuous recording uses less CPU than old method
- Segmentation can process hundreds of keystrokes quickly
- Cleanup tool caches analysis for faster navigation

## 🤝 Contributing

The modular design makes it easy to add new features:
1. Create new module (e.g., `my_feature.py`)
2. Add tab in `main.py`
3. Use shared `AudioHandler` and `Config`

## 📄 License

Same as original project

## 🙏 Acknowledgments

Built upon the original keyboard acoustic research tool, now with:
- Better architecture
- Improved performance
- More flexibility
- Easier maintenance

---

**Enjoy the new modular design! 🎉**
