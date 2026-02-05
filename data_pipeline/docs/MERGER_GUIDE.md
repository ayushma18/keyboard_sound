# Dataset Merger Guide

## Overview

The Dataset Merger tool allows you to combine multiple segmented keystroke datasets into a single unified dataset. This is useful for:

- Combining recordings from different sessions
- Merging data from different users/keyboards
- Building larger training datasets
- Consolidating partial datasets

## Features

### 🎯 Smart Merging
- Automatically combines files from the same key categories
- Preserves chronological ordering within each key
- Sequential numbering (0.wav, 1.wav, 2.wav, ...) in output

### 📊 Statistics
- Tracks total files merged
- Shows key distribution
- Records source dataset information
- Creates detailed merge summary report

### ⚡ Easy to Use
- Simple UI with dataset selection
- Visual progress tracking
- Validation to ensure dataset integrity

## How to Use

### Step 1: Select Datasets

1. Click **"Add Dataset"** button
2. Navigate to `recordings/segmented/` folder
3. Select a segmented dataset folder
4. Repeat to add more datasets (minimum 2 required)

💡 **Tip**: You can select datasets from different recording sessions or users.

### Step 2: Configure Output

1. **Merged Dataset Name**: Enter a name for the output dataset
   - Default: `merged_YYYYMMDD_HHMMSS`
   - Will be created in `recordings/segmented/` folder

2. **Options**:
   - ✅ **Preserve chronological order**: Maintains file order from source datasets
   - ✅ **Create merged summary report**: Generates detailed statistics file

### Step 3: Merge

1. Click **"Start Merge"** button
2. Monitor progress bar
3. Review results in the results panel

## Output Structure

The merged dataset will have the same structure as segmented datasets:

```
recordings/segmented/merged_20260206_143000/
├── a/
│   ├── 0.wav
│   ├── 1.wav
│   ├── 2.wav
│   └── ...
├── b/
│   ├── 0.wav
│   ├── 1.wav
│   └── ...
├── ... (other keys)
└── merge_summary.txt
```

### Merge Summary File

The `merge_summary.txt` file contains:
- Merge date and time
- Number of source datasets
- Total files merged
- Source dataset details
- Complete key distribution

Example:
```
DATASET MERGE SUMMARY
================================================================================

Merge Date: 2026-02-06 14:30:00
Number of source datasets: 2
Total files merged: 1500
Keys with data: 36

SOURCE DATASETS:
--------------------------------------------------------------------------------
  1. dji-kumara_continuous_main-data (750 files)
  2. dji2-kumara_continuous_num (750 files)

KEY DISTRIBUTION:
--------------------------------------------------------------------------------
  0     :   45 samples
  1     :   45 samples
  a     :   80 samples
  ...
```

## File Ordering

Files are merged and renumbered sequentially:

**Dataset 1 (source1/):**
- a/0.wav, a/1.wav, a/2.wav

**Dataset 2 (source2/):**
- a/0.wav, a/1.wav

**Merged Output (merged/):**
- a/0.wav (from source1/a/0.wav)
- a/1.wav (from source1/a/1.wav)
- a/2.wav (from source1/a/2.wav)
- a/3.wav (from source2/a/0.wav)
- a/4.wav (from source2/a/1.wav)

## Best Practices

### ✅ Do:
- Merge datasets with similar recording conditions
- Check that datasets use the same audio settings (sample rate, channels)
- Review merge summary after completion
- Keep source datasets as backups

### ⚠️ Caution:
- Merging datasets with very different audio quality may affect model training
- Very large merges (10,000+ files) may take several minutes
- Ensure sufficient disk space (approximately sum of source datasets)

## Use Cases

### Example 1: Combining Daily Sessions
```
Session 1 (Monday): 500 samples
Session 2 (Tuesday): 500 samples
Session 3 (Wednesday): 500 samples
→ Merged: 1,500 samples
```

### Example 2: Multi-User Dataset
```
User A keyboard: 800 samples
User B keyboard: 800 samples
User C keyboard: 800 samples
→ Merged: 2,400 samples (diverse keyboard sounds)
```

### Example 3: Balanced Dataset Creation
```
Numbers dataset: 370 samples (0-9)
Letters dataset: 1,040 samples (a-z)
→ Merged: 1,410 samples (balanced alphanumeric)
```

## Troubleshooting

### "Invalid Dataset" Error
- Ensure the selected folder contains key folders (a-z, 0-9)
- Check that key folders contain .wav files
- Verify this is a segmented dataset, not a continuous recording

### "Output folder already exists" Error
- Change the output dataset name
- Or delete/rename the existing output folder

### Merge Takes Too Long
- Normal for large datasets (>5,000 files)
- Progress bar shows current status
- Do not close the application during merge

### Missing Keys in Output
- Some keys may not exist in all source datasets
- Only keys present in at least one source will appear in output
- Check merge_summary.txt for complete distribution

## Technical Details

### File Validation
- Validates dataset structure before merging
- Checks for .wav files in key folders
- Ensures minimum 2 datasets selected

### Threading
- Merge runs in background thread
- UI remains responsive during merge
- Progress updates in real-time

### File Operations
- Uses `shutil.copy2()` to preserve file metadata
- Creates new sequential filenames
- Original files remain unchanged

## Integration with Pipeline

The merged dataset can be used with other tools:

1. **Data Cleanup**: Clean the merged dataset to remove noise
2. **Data Analyzer**: Analyze merged dataset statistics
3. **Model Training**: Use merged dataset for ML model training

## Example Workflow

```
1. Record Session A → Segment → 500 samples
2. Record Session B → Segment → 500 samples  
3. Record Session C → Segment → 500 samples
4. Merge A + B + C → 1,500 samples
5. Cleanup merged dataset → Remove bad samples
6. Analyze final dataset → Generate spectrograms
7. Train ML model → Use cleaned merged dataset
```

---

**Questions or Issues?**
Check the main documentation or create an issue in the project repository.
