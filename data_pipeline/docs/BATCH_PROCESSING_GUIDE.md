# Batch Processing Guide

## Overview

The Data Segmentation tool now supports **recursive batch processing**, allowing you to process multiple recording sessions at once. This is perfect for processing all sessions from a specific microphone/keyboard combination.

## Features

### 1. Recursive Folder Detection
- Automatically finds all valid sessions within a parent folder
- Searches up to 3 levels deep
- Validates each session (must contain `audio.wav` and `keystroke_log.csv`)

### 2. Batch Processing
- Process multiple sessions in one operation
- Maintains consistent output structure
- Tracks statistics per session and overall

### 3. Detailed Logging
- Session-by-session progress reporting
- Key distribution across all sessions
- Per-session statistics (saved, skipped, errors)
- Comprehensive summary report

## Usage

### Folder Structure

Your recording sessions should be organized like this:

```
recordings/
  └── dji2-kumara/          # Mic ID - Keyboard ID
      └── continuous/       # Recording type
          ├── 0/            # Session 1
          │   ├── audio.wav
          │   └── keystroke_log.csv
          ├── 1/            # Session 2
          │   ├── audio.wav
          │   └── keystroke_log.csv
          └── 2/            # Session 3
              ├── audio.wav
              └── keystroke_log.csv
```

### Steps

1. **Open Data Segmentation Tab**
   - Navigate to the "Data Segmentation" tab in the application

2. **Select Parent Folder**
   - Click "Browse Session Folder"
   - Navigate to either:
     - A single session folder (e.g., `recordings/dji2-kumara/continuous/0`)
     - A parent folder containing multiple sessions (e.g., `recordings/dji2-kumara/continuous`)

3. **Batch Confirmation**
   - If multiple sessions are found, you'll see a popup listing all detected sessions
   - Click "Yes" to process all sessions in batch mode
   - Click "No" to cancel and select a single session instead

4. **Configure Parameters** (optional)
   - Segment Duration: Length of each extracted audio clip
   - Pre-trigger: Time before peak to include
   - Post-trigger: Time after peak to include
   - Peak Centering: Align segments to actual audio peaks
   - Filtering: Apply bandpass filter to extracted segments

5. **Start Processing**
   - Click "Start Segmentation"
   - Monitor progress in the UI
   - Console shows detailed per-session logging

## Output Structure

After batch processing, your output folder will contain:

```
segmented_20260205_120000/    # Output folder
  ├── a/                       # Key folders
  │   ├── 0.wav
  │   ├── 1.wav
  │   └── ...
  ├── b/
  │   ├── 0.wav
  │   └── ...
  ├── batch_summary.txt        # Comprehensive summary
  └── metadata.csv             # Per-sample metadata (if single session)
```

## Batch Summary Report

The `batch_summary.txt` file contains:

### Overall Statistics
- Total sessions processed
- Success/failure counts
- Total keystrokes across all sessions
- Total segments saved
- Overall success rate

### Key Distribution
Shows how many samples were collected for each key across all sessions:
```
KEY DISTRIBUTION (across all sessions):
----------------------------------------
  a         : 102 samples
  b         :  89 samples
  c         :  95 samples
  ...
```

### Session Details
For each session:
- Status (SUCCESS/FAILED)
- Number of keystrokes processed
- Number of segments saved
- Number skipped (silent segments)
- Key-by-key breakdown

Example:
```
Session: 0
  Status: SUCCESS
  Keystrokes: 260
  Saved: 258
  Skipped: 2
  Keys found:
    a: 10
    b: 8
    c: 9
    ...
```

## Console Logging

During batch processing, the console displays:

```
================================================================================
BATCH SEGMENTATION: 10 sessions
Output: C:/temp/keyboard_sound/recordings/dji2-kumara/continuous/segmented_...
================================================================================

[1/10] Processing: 0
------------------------------------------------------------
  ✓ Success: 258 segments saved

[2/10] Processing: 1
------------------------------------------------------------
  ✓ Success: 245 segments saved

...

KEY DISTRIBUTION:
  a         : 102 samples
  b         :  89 samples
  ...
```

## Tips

### Best Practices
1. **Consistent Naming**: Use descriptive folder names (mic-id/keyboard-id)
2. **Pre-check Sessions**: Ensure all sessions have valid audio and log files
3. **Backup First**: Keep original recordings safe before processing
4. **Monitor Console**: Watch for warnings about skipped segments

### Performance
- Processing time depends on:
  - Number of sessions
  - Audio file sizes
  - Number of keystrokes per session
  - Peak centering enabled/disabled
- Typical rate: ~100-500 keystrokes per minute

### Troubleshooting

**"No valid sessions found"**
- Check that subfolders contain both `audio.wav` and `keystroke_log.csv`
- Verify folder depth (max 3 levels)

**Some sessions failed**
- Check `batch_summary.txt` for error messages
- Common causes: corrupted audio, missing timestamps, file permissions

**Low success rate**
- Try adjusting segmentation parameters
- Check audio quality (silent segments are skipped)
- Verify keystroke log timestamps are accurate

## Example Workflow

Process all sessions from a DJI2 microphone with Kumara keyboard:

1. Navigate to `recordings/dji2-kumara/continuous`
2. This folder contains sessions 0-9
3. Select the `continuous` folder in the browser
4. Confirm batch processing (10 sessions)
5. Click "Start Segmentation"
6. Wait for completion
7. Check `batch_summary.txt` for results
8. Verify key distribution meets your needs

Result: All 10 sessions processed into one consolidated dataset!

## Advanced Usage

### Partial Processing
To process only some sessions:
1. Create a temporary folder
2. Copy/move desired session folders into it
3. Process that folder in batch mode

### Re-processing
- Safe to re-run with different parameters
- Output numbering continues from existing files
- Consider backing up previous output first

### Quality Control
After batch processing:
1. Review `batch_summary.txt`
2. Check key distribution balance
3. Spot-check a few audio samples from each key
4. Verify segment durations are consistent
