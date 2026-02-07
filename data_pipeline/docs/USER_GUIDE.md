# User Guide - Keyboard Acoustic Research Tool (Modular Edition)

## Table of Contents
1. [Getting Started](#getting-started)
2. [Tab 1: Data Collection](#tab-1-data-collection)
3. [Tab 2: Data Segmentation](#tab-2-data-segmentation)
4. [Tab 3: Data Cleanup](#tab-3-data-cleanup)
5. [Tab 4: Data Analyzer](#tab-4-data-analyzer)
6. [Settings](#settings)
7. [Tips & Best Practices](#tips--best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Getting Started

### Installation

1. **Create virtual environment** (if not already done):
   ```bash
   python -m venv myenv
   ```

2. **Activate virtual environment**:
   - Windows: `myenv\Scripts\activate`
   - Linux/Mac: `source myenv/bin/activate`

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```
   
   Or use the launcher:
   - Windows: Double-click `run.bat`
   - Linux/Mac: `./run.sh`

### First Time Setup

1. Go to **File → Settings**
2. Select your **Input Device** (microphone)
3. Select your **Output Device** (speakers/headphones)
4. Click **Calibrate Background Noise** (stay quiet for 2 seconds)
5. Click **Save Settings**

---

## Tab 1: Data Collection

### Purpose
Record continuous audio with keystroke timestamps. This is more flexible than the old method of saving files per keystroke.

### Workflow

#### Step 1: Configure Session
- **Session Name**: Auto-generated or enter custom name
  - Format: `session_YYYYMMDD_HHMMSS`
  - Example: `session_20260204_143022`
  
- **Mic ID**: Identifier for your microphone
  - Examples: `mic1`, `blue_yeti`, `laptop_mic`
  
- **Keyboard ID**: Identifier for your keyboard
  - Examples: `kb1`, `mechanical`, `laptop_kb`

#### Step 2: Start Recording
1. Click **"Start Recording"**
2. Timer starts (format: `HH:MM:SS`)
3. Start typing naturally
4. Keystroke counter updates in real-time

#### Step 3: Stop Recording
1. Click **"Stop Recording"**
2. Application saves:
   - `audio.wav` - Full recording
   - `keystroke_log.csv` - Timestamps and keys
   - `session_info.txt` - Session metadata

#### Output Location
```
recordings/
  └── micX-kbY/
      └── continuous/
          └── session_name/
              ├── audio.wav
              ├── keystroke_log.csv
              └── session_info.txt
```

### Tips
- **Natural typing**: Type as you normally would
- **Stay consistent**: Keep mic position constant
- **No rush**: Recording is continuous, no need to hurry
- **Check keystroke count**: Verify keys are being detected

---

## Tab 2: Data Segmentation

### Purpose
Extract individual keystroke samples from continuous recordings. You can run this multiple times with different parameters without re-recording.

### Workflow

#### Step 1: Load Session
1. Click **"Browse Session Folder"**
2. Navigate to: `recordings/micX-kbY/continuous/session_name/`
3. Select the folder
4. Application displays:
   - Audio duration
   - Sample rate
   - Number of keystrokes

#### Step 2: Configure Parameters

##### Segment Duration
- **Default**: 0.430 seconds (18963 samples @ 44.1kHz)
- **Range**: 0.1 - 1.0 seconds
- **Recommendation**: 0.43s matches original dataset format

##### Pre/Post Trigger
- **Pre-trigger**: Time before keystroke peak
  - Default: 0.1 seconds
  - Captures attack phase
  
- **Post-trigger**: Time after keystroke peak
  - Default: 0.33 seconds
  - Captures decay phase

##### Peak Centering
- **Enabled (Recommended)**: Automatically finds and centers the keystroke peak
- **Disabled**: Uses fixed window around timestamp

##### Bandpass Filtering
- **Enabled**: Apply frequency filtering
- **Presets**:
  - **Optimal (50-5kHz)**: Best for ML models
  - **Dataset Match (50-3kHz)**: Matches original dataset
  - **Extended (50-8kHz)**: More frequency information
- **Custom**: Enter your own low/high cutoff frequencies

#### Step 3: Set Output Name
- Default: `segmented_YYYYMMDD_HHMMSS`
- Custom: Enter any folder name

#### Step 4: Process
1. Click **"Start Segmentation"**
2. Progress bar shows status
3. Wait for completion
4. Review results in text area

#### Output Location
```
recordings/
  └── micX-kbY/
      └── continuous/
          └── segmented_name/
              ├── metadata.csv
              ├── a/
              │   ├── a_0001.wav
              │   ├── a_0002.wav
              │   └── ...
              ├── b/
              │   ├── b_0001.wav
              │   └── ...
              └── ...
```

### Parameter Guidelines

| Use Case | Duration | Pre | Post | Peak Center | Filter |
|----------|----------|-----|------|-------------|--------|
| ML Training | 0.43s | 0.1s | 0.33s | ✅ | 50-5kHz |
| Analysis | 0.5s | 0.15s | 0.35s | ✅ | 50-8kHz |
| Raw Data | 0.3s | 0.1s | 0.2s | ❌ | None |

---

## Tab 3: Data Cleanup

### Purpose
Review segmented data and remove noise/invalid recordings.

### Workflow

#### Step 1: Launch Tool
1. Click **"Launch Data Cleanup Tool"**
2. New window opens with cleanup interface

#### Step 2: Load Session
1. Click **"Browse Folder"**
2. Select a segmented folder (e.g., `recordings/mic1-kb1/segmented_...`)
3. Tool analyzes all files

#### Step 3: Review & Filter
- **Energy Concentration**: Distinguishes keystrokes from noise
  - Higher score = concentrated energy (keystroke)
  - Lower score = distributed energy (noise)
  
- **Adjust Thresholds**:
  - **Primary**: Energy Concentration Threshold (0.3 default)
  - **Secondary**: RMS Threshold (0.0001 default)

#### Step 4: Navigate
- **Next/Previous**: Review files sequentially
- **Next/Previous Noise**: Jump to files flagged as noise
- **Filter**: Show only noise files

#### Step 5: Mark & Delete
1. **Toggle Mark**: Mark current file for deletion
2. **Mark All Below**: Mark all noise files
3. **Delete Marked**: Remove marked files permanently
4. **Export Report**: Save cleanup statistics

### Detection Methods

#### Energy Concentration (Recommended)
- Measures how concentrated energy is in time
- Good keystrokes have sharp peaks
- Noise has distributed energy

#### RMS Threshold (Fallback)
- Based on volume level
- Very quiet sounds = noise
- Works for obviously invalid recordings

---

## Tab 4: Data Analyzer

### Purpose
Visualize and analyze audio files with spectrograms, waveforms, and more.

### Workflow

#### Step 1: Launch Tool
1. Click **"Launch Audio Analyzer"**
2. New window opens with analyzer interface

#### Step 2: Load Audio
- **Single File**: Click "Load Audio File"
- **Folder**: Click "Load Folder (Recursive)" for batch analysis

#### Step 3: View Analysis
Multiple visualization tabs:
1. **Main Analysis**: Waveform, spectrogram, mel spectrogram, energy, MFCC
2. **Detailed Spectrogram**: Linear, log, mel, chromagram
3. **Comparison View**: Side-by-side comparison of multiple files
4. **Bulk View**: All waveforms in one view

#### Step 4: Playback
- **Play**: Listen to current file
- **Pause/Stop**: Control playback
- **Seek**: Jump to position
- **Keyboard Shortcuts**:
  - `Space`: Play/Pause
  - `Esc`: Stop
  - `←`: Back 0.1s
  - `→`: Forward 0.1s

#### Step 5: Adjust Parameters
- **FFT Size**: 512, 1024, 2048, 4096
- **Window**: Hann, Hamming, Blackman
- **Toggle Views**: Enable/disable individual plots

---

## Settings

### Audio Devices
- **Input Device**: Select microphone
- **Output Device**: Select speakers/headphones
- **Tip**: Use same device consistently for best results

### Audio Parameters
- **Sample Rate**: 44100 Hz (recommended, matches dataset)
- **Channels**: 2 (stereo) - mono mics automatically converted

### Noise Calibration
- Run before each session
- Stay quiet for 2 seconds
- Captures ambient noise profile
- Used for noise reduction

---

## Tips & Best Practices

### Recording Quality
1. **Quiet environment**: Minimize background noise
2. **Consistent position**: Keep mic in same place
3. **Calibrate first**: Always calibrate noise before recording
4. **Test audio**: Use mic test before starting session

### Segmentation
1. **Try different parameters**: Segment same recording multiple ways
2. **Use peak centering**: Almost always beneficial
3. **Apply filtering**: Improves ML model performance
4. **Check metadata**: Review RMS/peak levels

### Cleanup
1. **Use concentration detection**: More reliable than RMS alone
2. **Listen before deleting**: Verify files are actually noise
3. **Export report**: Keep record of cleanup statistics
4. **Backup first**: Make copy before mass deletion

### Analysis
1. **Compare similar keys**: Look for consistency
2. **Check frequency content**: Verify filtering worked
3. **Identify patterns**: Look for typing style characteristics
4. **Use bulk view**: Spot outliers quickly

---

## Troubleshooting

### Issue: No audio devices found
**Solution**: 
- Check microphone is connected
- Run `python -c "import sounddevice; print(sounddevice.query_devices())"`
- Restart application

### Issue: Choppy audio playback
**Solution**:
- Pygame should fix this automatically
- If still choppy, check CPU usage
- Try different output device

### Issue: Keystrokes not detected
**Solution**:
- Check keyboard listener is active
- Try running as administrator
- Verify no other apps are blocking keyboard input

### Issue: Segmentation takes too long
**Solution**:
- Normal for large sessions (1000+ keys)
- Progress bar shows status
- Don't close window while processing

### Issue: Cleanup tool shows everything as noise
**Solution**:
- Lower concentration threshold (try 0.2-0.25)
- Lower RMS threshold
- Verify recordings have actual keystroke sounds

### Issue: Files won't play in analyzer
**Solution**:
- Check file format (must be WAV)
- Verify file isn't corrupted
- Try loading single file first

---

## Keyboard Shortcuts

### Data Analyzer
- `Space`: Play/Pause
- `Esc`: Stop
- `←`: Skip backward 0.1s
- `→`: Skip forward 0.1s

---

## File Formats

### Audio Files
- **Format**: WAV (uncompressed)
- **Sample Rate**: 44100 Hz
- **Channels**: 2 (stereo)
- **Bit Depth**: 32-bit float

### Keystroke Log (CSV)
```csv
timestamp,relative_time,key,datetime
1707055802.123,0.0,a,2026-02-04T14:30:02.123
1707055803.456,1.333,b,2026-02-04T14:30:03.456
```

### Metadata (CSV)
```csv
key,filename,timestamp,relative_time,rms,peak,samples,file_number
a,a_0001.wav,1707055802.123,0.0,0.0123,0.456,18963,1
```

---

## Advanced Usage

### Batch Processing
Process multiple sessions:
```python
from data_segmenter import DataSegmenterTab
from audio_handler import AudioHandler
from config import Config

config = Config()
audio = AudioHandler()

sessions = ['session1', 'session2', 'session3']
for session in sessions:
    # Load and segment each session
    ...
```

### Custom Filtering
Implement custom filters in `audio_handler.py`:
```python
def apply_custom_filter(self, audio):
    # Your filter implementation
    return filtered_audio
```

### Export to Other Formats
Convert WAV to other formats:
```python
import librosa
import soundfile as sf

# Load
audio, sr = librosa.load('file.wav')

# Save as different format
sf.write('file.flac', audio, sr, format='FLAC')
```

---

## FAQ

**Q: Can I use this with the old data?**  
A: Yes! The segmented data format is compatible. You can use cleanup and analyzer tabs with old recordings.

**Q: How long can I record continuously?**  
A: Limited only by disk space. ~10MB per minute for stereo 44.1kHz.

**Q: Can I change parameters after recording?**  
A: Yes! That's the main advantage. Segment the same recording multiple times with different parameters.

**Q: What if I make a mistake during recording?**  
A: Just keep recording. You can remove bad segments during cleanup.

**Q: Can I use a different sample rate?**  
A: Yes, but 44.1kHz is recommended to match original dataset format.

---

## Support

For issues or questions:
1. Check this guide first
2. Review [README.md](README.md)
3. Check error messages carefully
4. Verify all dependencies installed

---

**Happy Research! 🎹🎵**
