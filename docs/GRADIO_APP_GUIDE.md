# 🎹 AI Keystroke Detection - Gradio App Guide

## ✅ Status: Running Successfully!

Your beautiful Gradio web interface is now running at:
**http://127.0.0.1:7865**

---

## 🎯 Features

### ✨ Beautiful Modern UI
- **Gradient header** with purple/pink theme
- **Real-time predictions** with confidence scores
- **Interactive visualizations** (confidence bars + spectrogram)
- **Responsive design** that looks great on all screens

### 🎤 DJI Mic Support
- Works with **any microphone** including your DJI mic
- **44.1 kHz sample rate** for high-quality audio
- **Automatic resampling** if your mic uses different rates
- **Stereo to mono conversion** automatic

### 🤖 AI Predictions
- **Top 5 predictions** with confidence scores
- **Beautiful gradient bar charts**
- **Real-time processing** (< 1 second)
- **Mel-spectrogram visualization** showing audio features

---

## 📝 How to Use

### Method 1: Using the Web Interface

1. **Open your browser** and go to: http://127.0.0.1:7865

2. **Click the microphone icon** in the "Record Keystroke Sound" section

3. **Press a key** on your keyboard (the DJI mic will capture the sound)

4. **Stop recording** by clicking the microphone icon again

5. **Click "🔍 Analyze Keystroke"** to get the prediction

6. **View results**:
   - Large predicted key with confidence
   - Top 5 predictions bar chart
   - Audio spectrogram visualization

### Method 2: Upload Audio File

- Instead of recording, you can also **upload a WAV file**
- Click the upload area and select an audio file
- Then click "Analyze Keystroke"

---

## 🚀 Running the App

### Start the Server

```bash
# Basic usage (auto-detects model)
python app_gradio.py

# Specify model
python app_gradio.py --model model/CNN-Final.pkl

# Custom port
python app_gradio.py --port 8080

# Create public share link (accessible from internet)
python app_gradio.py --share

# Full example
python app_gradio.py --model model/CNN-Final.pkl --port 7865 --share
```

### Stop the Server

Press **Ctrl+C** in the terminal

---

## 🎨 UI Improvements

### What's New in This Version:

1. **Modern Gradient Design**
   - Purple/pink gradients for headers
   - Smooth animations and transitions
   - Professional color scheme

2. **Enhanced Visualizations**
   - Gradient bar charts with emerald green for top prediction
   - Magma colormap for spectrogram (better than viridis)
   - Larger, clearer plots with better labels

3. **Better User Experience**
   - Clear instructions at the top
   - Status updates (real-time feedback)
   - System information panel
   - HTML-formatted results with icons

4. **Improved Error Handling**
   - Checks for audio length
   - Validates input data
   - Displays helpful error messages

5. **Smart Model Detection**
   - Auto-finds available models
   - Supports CNN and CoAtNet architectures
   - Shows model info in UI

---

## 🎤 DJI Mic Configuration

### Setup Tips:

1. **Connect your DJI mic** to your computer
2. **Set as default input** in Windows Sound Settings
3. **Position near keyboard** (5-15 cm recommended)
4. **Test the recording** in Gradio before collecting data

### Optimal Recording Settings:

- **Distance**: 5-15 cm from keyboard
- **Environment**: Quiet room (minimal background noise)
- **Duration**: 0.5-2 seconds per keystroke
- **Format**: WAV, 44.1 kHz (auto-converted if different)

---

## 📊 Model Information

### Current Model: CNN-Final.pkl

- **Architecture**: Convolutional Neural Network (CNN)
- **Classes**: 26 (a-z alphabet keys)
- **Parameters**: 407,066
- **Input**: 64x64 Mel-spectrogram
- **Device**: CPU (can use GPU if available)

### Available Models:

- `model/CNN-Final.pkl` - CNN for alphabet (a-z)
- `model/CNN-Best.pkl` - Best CNN checkpoint
- `model/CoAtNet-1-Phone.pkl` - CoAtNet architecture
- `model/CoAtNet-1-Best-Phone.pkl` - Best CoAtNet

---

## 🔧 Troubleshooting

### Issue: Can't hear audio in browser
**Solution**: Check if your DJI mic is set as default input device

### Issue: Low confidence predictions
**Solution**: 
- Move mic closer to keyboard
- Reduce background noise
- Record longer audio (1-2 seconds)

### Issue: Port already in use
**Solution**: Use a different port with `--port 8080`

### Issue: Model not found
**Solution**: Specify model path explicitly:
```bash
python app_gradio.py --model model/CNN-Final.pkl
```

---

## 🌐 Sharing the App

### Create Public URL:

```bash
python app_gradio.py --share
```

This creates a temporary public URL (valid for 72 hours) that you can share with anyone!

Example output:
```
📍 Local URL:      http://127.0.0.1:7865
🌐 Public URL:     https://[random-id].gradio.live
```

---

## 📱 Mobile Access

### Access from Phone/Tablet on Same Network:

1. Find your computer's IP address:
   ```bash
   ipconfig  # On Windows
   ```

2. Open browser on phone and go to:
   ```
   http://[YOUR-IP]:7865
   ```
   Example: `http://192.168.1.100:7865`

---

## 🎉 Next Steps

### Ideas for Improvement:

1. **Record Training Data**
   - Use the interface to collect more samples
   - Save recordings with labels

2. **Test Different Models**
   - Try CoAtNet models for comparison
   - Test with different datasets

3. **Customize UI**
   - Modify colors in `custom_css`
   - Add more features to the interface

4. **Deploy Online**
   - Use Hugging Face Spaces (free hosting)
   - Deploy to cloud services

---

## 📞 Support

### Need Help?

- Check console output for error messages
- Verify model file exists
- Ensure all dependencies are installed:
  ```bash
  pip install gradio librosa sounddevice soundfile torch torchvision torchaudio
  ```

---

## 🎊 Enjoy Your Beautiful AI Keystroke Detector!

**Your app is now ready to use with your DJI mic!**

Open: **http://127.0.0.1:7865** 🚀
