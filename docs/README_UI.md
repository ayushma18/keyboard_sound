# Keystroke Detection System

A real-time keystroke detection and classification system using CoAtNet deep learning model.

## Features

- **Real-time Detection**: Continuously monitors audio and detects keystrokes as you type
- **High Accuracy**: Uses CoAtNet-1 architecture with mel-spectrogram preprocessing
- **Detailed Results**: Shows top 5 predictions with confidence scores
- **Visual Feedback**: Interactive UI with confidence bars and detection history
- **Adjustable Sensitivity**: Fine-tune detection threshold for your keyboard
- **Statistics Tracking**: Monitor keystroke count, average confidence, and recording time

## Setup

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Install PyTorch with CUDA (Optional but Recommended)

For GPU acceleration, install PyTorch with CUDA support:

**Automatic Installation (Recommended):**
```powershell
.\install_pytorch_cuda.bat
```

**Manual Installation:**
```powershell
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Verify installation:
```powershell
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 3. Place Model File

Place your trained model file (e.g., `CoAtNet-1-Best-Zoom.pkl`) in the `model/` folder.

The UI will automatically detect and load models with these names:
- `CoAtNet-1-Best-Zoom.pkl`
- `CoAtNet-1-Zoom.pkl`

Alternatively, you can browse for any model file using the UI.

## Usage

### Running the Application

```powershell
python keystroke_detector_ui.py
```

### Using the UI

1. **Load Model**
   - The application will auto-load if a model is found in the `model/` folder
   - Or click "Browse" to select a model file manually
   - Click "Load Model" to load the selected model

2. **Start Recording**
   - Click "🎤 Start Recording" button
   - The system will continuously listen for keystrokes
   - Detected keys will appear in real-time

3. **View Results**
   - **Last Detected Key**: Shows the most recent keystroke with confidence
   - **Top 5 Predictions**: Displays alternative predictions with probability bars
   - **Detection History**: Scrollable log of all detected keystrokes with timestamps

4. **Adjust Sensitivity**
   - Use the "Detection Sensitivity" slider to adjust the threshold
   - Lower values = more sensitive (may detect more false positives)
   - Higher values = less sensitive (may miss some keystrokes)
   - Default: 0.06

5. **Statistics**
   - Total Keystrokes: Count of detected keystrokes
   - Recording Time: Duration of current recording session
   - Avg Confidence: Average prediction confidence

6. **Stop Recording**
   - Click "⏹ Stop Recording" to stop detection
   - Use "🗑 Clear Results" to reset all results and history

## Model Architecture

The system uses CoAtNet-1 architecture with:
- **Input**: Mel-spectrogram (64x64) from keystroke audio
- **Layers**: 
  - Stage 0: Stem (Convolution)
  - Stage 1-2: MBConv blocks (Mobile Inverted Bottleneck)
  - Stage 3-4: Transformer blocks with relative attention
- **Output**: 36 classes (0-9, a-z)
- **Parameters**: ~5M parameters

## Audio Processing

### Segmentation
- **FFT Size**: 48
- **Hop Length**: 24
- **Before Peak**: 2400 samples (~50ms at 48kHz)
- **After Peak**: 12000 samples (~250ms at 48kHz)

### Preprocessing
- **Mel Spectrogram**:
  - n_mels: 64
  - n_fft: 2048
  - win_length: 1024
  - hop_length: 226
- **Power to dB conversion**

## Technical Details

### Components

1. **model/model_architecture.py**
   - Complete CoAtNet model implementation
   - Includes Stem, MBConv, Relative Attention, and Transformer blocks

2. **model/audio_processor.py**
   - Audio segmentation using energy-based keystroke detection
   - Mel-spectrogram preprocessing
   - Real-time buffer management

3. **model/recorder.py**
   - Real-time audio recording with sounddevice
   - Background processing thread
   - Queue-based audio chunk handling

4. **keystroke_detector_ui.py**
   - Main UI application using tkinter
   - Threading for non-blocking audio processing
   - Real-time result visualization

### Performance

- **Sample Rate**: 48kHz
- **Processing Latency**: ~100ms per keystroke
- **Device Support**: CUDA (GPU) or CPU

## Troubleshooting

### Model Not Loading
- Ensure the model file is in PyTorch format (.pkl, .pth, or .pt)
- Check that the model was trained with the same architecture
- Verify the file path is correct

### No Audio Recording
- Check microphone permissions
- Ensure sounddevice can access your audio device
- Try adjusting the detection sensitivity slider

### Poor Detection Accuracy
- Adjust the "Detection Sensitivity" slider
- Ensure quiet recording environment
- Use the same keyboard type as training data
- Check that model is properly trained

### High CPU/Memory Usage
- Use GPU if available (CUDA)
- Reduce processing frequency (modify `process_interval` in code)
- Close other applications

## Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA (optional, for GPU acceleration)
- Microphone access
- ~2GB RAM minimum
- ~500MB disk space for model

## License

This project is part of a keyboard acoustic side-channel research system.
