"""
⌨️ Gradio Web Interface for Keyboard Keystroke Detection
Beautiful, modern interface for real-time keystroke prediction from audio
Auto-detects CNN or CoAtNet model architectures
"""

import gradio as gr
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image
import librosa
import sounddevice as sd
import warnings
warnings.filterwarnings('ignore')

# Import components from existing UI
from keystroke_detector_ui import AudioPreprocessor, CNN, MyCoAtNet

# Set matplotlib style for better-looking plots
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    plt.style.use('default')
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = '#f8f9fa'


class ModelLoader:
    """Handles automatic model detection and loading"""
    
    def __init__(self, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.num_classes = None
        self.model_type = None
        self.key_labels = None
        
    def detect_model_type(self, state_dict):
        """Detect model type and num_classes from state_dict keys"""
        keys = list(state_dict.keys())
        
        # Check for CNN model keys
        cnn_keys = ['conv1.weight', 'conv2.weight', 'conv3.weight', 'fc1.weight', 'fc2.weight']
        if any(key in keys for key in cnn_keys):
            print(f"✓ Detected CNN architecture")
            if 'fc2.weight' in state_dict:
                num_classes = state_dict['fc2.weight'].shape[0]
                print(f"✓ Detected {num_classes} output classes")
                return 'CNN', num_classes
            return 'CNN', 36
        
        # Check for CoAtNet model keys
        coatnet_keys = ['0.0.0.weight', '1.0.mb_conv.0.weight', '3.0.0.relative_bias']
        if any(key in keys for key in coatnet_keys):
            print(f"✓ Detected CoAtNet architecture")
            if '5.fc.weight' in state_dict:
                num_classes = state_dict['5.fc.weight'].shape[0]
                print(f"✓ Detected {num_classes} output classes")
                return 'CoAtNet', num_classes
            return 'CoAtNet', 36
        
        print(f"⚠ Warning: Unknown model type, defaulting to CNN")
        print(f"First 5 keys: {keys[:5]}")
        return 'CNN', 36
    
    def load_model(self, model_path):
        """Load model from path with automatic architecture detection"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"\n{'='*60}")
        print(f"Loading model from: {model_path}")
        print(f"Device: {self.device.upper()}")
        
        # Load state dict and detect architecture
        state_dict = torch.load(model_path, map_location=self.device)
        self.model_type, self.num_classes = self.detect_model_type(state_dict)
        
        # Create appropriate model
        if self.model_type == 'CNN':
            print(f"Creating CNN model with {self.num_classes} classes...")
            self.model = CNN(num_classes=self.num_classes)
        else:
            print(f"Creating CoAtNet-1 model with {self.num_classes} classes...")
            nums_blocks = [2, 2, 3, 5, 2]
            channels = [64, 96, 192, 384, 768]
            self.model = MyCoAtNet(nums_blocks, channels, num_classes=self.num_classes)
        
        # Load weights
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Setup labels
        if self.num_classes == 26:
            self.key_labels = [chr(i) for i in range(ord('a'), ord('z') + 1)]
            print("Using alphabet-only labels (a-z)")
        elif self.num_classes == 36:
            digits = [str(i) for i in range(10)]
            alphabet = [chr(i) for i in range(ord('a'), ord('z') + 1)]
            self.key_labels = digits + alphabet
            print("Using alphanumeric labels (0-9, a-z)")
        else:
            # Generate generic labels for other cases
            self.key_labels = [f"class_{i}" for i in range(self.num_classes)]
            print(f"Using generic labels for {self.num_classes} classes")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameters: {total_params:,}")
        
        # Test model
        with torch.no_grad():
            test_input = torch.randn(1, 1, 64, 64).to(self.device)
            test_output = self.model(test_input)
            print(f"Model test successful - output shape: {test_output.shape}")
        
        print(f"✓ Model loaded successfully!")
        print(f"{'='*60}\n")
        
        return self.model


class KeystrokeDetector:
    """Main detection class for Gradio interface"""
    
    def __init__(self, model_path="model.pth", sample_rate=44100):
        self.sample_rate = sample_rate
        
        # Initialize components
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        self.model_loader = ModelLoader()
        
        # Load model
        try:
            self.model = self.model_loader.load_model(model_path)
            self.device = self.model_loader.device
            self.key_labels = self.model_loader.key_labels
            self.model_info = f"{self.model_loader.model_type} model with {self.model_loader.num_classes} classes"
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def process_audio(self, audio_tuple):
        """
        Process audio from Gradio microphone input
        
        Args:
            audio_tuple: Tuple of (sample_rate, audio_data) from Gradio
        
        Returns:
            predicted_key: str with HTML formatting
            confidence_plot: matplotlib figure
            spectrogram_plot: matplotlib figure
            status_message: str
        """
        try:
            if audio_tuple is None:
                return "⚠️ No audio recorded. Please record a keystroke sound.", None, None, "❌ No audio input"
            
            sample_rate, audio_data = audio_tuple
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif audio_data.dtype == np.int32:
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            
            # Handle stereo: take first channel
            if len(audio_data.shape) > 1:
                audio_data = audio_data[:, 0]
            
            # Check if audio is too short
            if len(audio_data) < 1000:
                return "⚠️ Audio too short. Please record a longer keystroke sound.", None, None, "❌ Audio too short"
            
            # Resample if needed
            if sample_rate != self.sample_rate:
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.sample_rate)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(audio_data).float()
            
            # Process to mel-spectrogram
            mel_spec = self.preprocessor.process_audio(waveform)
            
            # Add batch dimension and move to device
            mel_spec_batch = mel_spec.unsqueeze(0).to(self.device)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(mel_spec_batch)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                probs_np = probabilities.cpu().numpy()[0]
            
            # Get top prediction
            predicted_idx = np.argmax(probs_np)
            predicted_key = self.key_labels[predicted_idx]
            confidence = probs_np[predicted_idx]
            
            # Get top 3 predictions for display
            top_3_indices = np.argsort(probs_np)[-3:][::-1]
            top_3_text = " | ".join([f"{self.key_labels[i].upper()}: {probs_np[i]*100:.1f}%" 
                                     for i in top_3_indices])
            
            # Generate visualizations
            confidence_plot = self.create_confidence_plot(probs_np, self.key_labels)
            spectrogram_plot = self.create_spectrogram_plot(mel_spec)
            
            # Format result with HTML/Markdown
            confidence_color = "#10b981" if confidence > 0.7 else "#f59e0b" if confidence > 0.4 else "#ef4444"
            
            result_text = f"""
<div style="text-align: center; padding: 30px; background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%); border-radius: 20px; color: white; box-shadow: 0 15px 40px rgba(37, 99, 235, 0.3);">
    <h1 style="font-size: 4em; margin: 10px 0; text-shadow: 3px 3px 6px rgba(0,0,0,0.3); letter-spacing: 4px;">
        ⌨️ {predicted_key.upper()}
    </h1>
    <div style="font-size: 2em; font-weight: bold; margin: 15px 0;">
        <span style="background-color: rgba(255,255,255,0.25); padding: 10px 25px; border-radius: 30px; backdrop-filter: blur(10px);">
            {confidence*100:.2f}% Confidence
        </span>
    </div>
</div>

<div style="margin-top: 20px; padding: 20px; background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%); border-radius: 12px; border-left: 5px solid #14b8a6; box-shadow: 0 4px 15px rgba(20, 184, 166, 0.1);">
    <p style="margin: 8px 0; font-size: 1.1em;"><strong>🎯 Top 3 Predictions:</strong> {top_3_text}</p>
    <p style="margin: 8px 0;"><strong>🤖 Model:</strong> {self.model_info}</p>
    <p style="margin: 8px 0;"><strong>⚡ Device:</strong> {self.device.upper()}</p>
    <p style="margin: 8px 0;"><strong>🎤 Sample Rate:</strong> 44.1 kHz</p>
    <p style="margin: 8px 0;"><strong>📊 Audio Duration:</strong> {len(audio_data)/self.sample_rate:.2f}s</p>
</div>
"""
            
            status_msg = f"✅ Predicted: {predicted_key.upper()} ({confidence*100:.1f}%)"
            
            return result_text, confidence_plot, spectrogram_plot, status_msg
            
        except Exception as e:
            error_msg = f"❌ Error processing audio: {str(e)}"
            return error_msg, None, None, error_msg
    
    def create_confidence_plot(self, probabilities, labels, top_k=5):
        """Create beautiful bar chart of top K predictions"""
        # Get top K predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probs = probabilities[top_indices]
        top_labels = [labels[i].upper() for i in top_indices]
        
        # Create plot with modern styling
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        ax.set_facecolor('#f0fdfa')
        
        # Create gradient colors matching the new theme
        colors = []
        for i in range(len(top_probs)):
            if i == 0:
                colors.append('#14b8a6')  # teal for top prediction
            elif i == 1:
                colors.append('#2563eb')  # blue
            elif i == 2:
                colors.append('#06b6d4')  # cyan
            else:
                colors.append('#3b82f6')  # lighter blue
        
        # Create horizontal bars
        bars = ax.barh(range(len(top_probs)), top_probs, height=0.6, color=colors, edgecolor='white', linewidth=2)
        
        # Style the plot
        ax.set_yticks(range(len(top_probs)))
        ax.set_yticklabels(top_labels, fontsize=16, fontweight='bold', fontfamily='monospace')
        ax.set_xlabel('Confidence Score', fontsize=14, fontweight='bold', color='#374151')
        ax.set_title('🎯 Top 5 Predictions', fontsize=18, fontweight='bold', color='#1f2937', pad=20)
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.2, linestyle='--', linewidth=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#9ca3af')
        ax.spines['bottom'].set_color('#9ca3af')
        
        # Add percentage labels with dynamic positioning
        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            # Add background box for better readability
            label_x = prob + 0.02 if prob < 0.85 else prob - 0.02
            ha = 'left' if prob < 0.85 else 'right'
            ax.text(label_x, bar.get_y() + bar.get_height()/2, 
                   f'{prob*100:.1f}%', 
                   va='center', ha=ha, fontsize=13, fontweight='bold', 
                   color='#1f2937' if prob < 0.85 else 'white')
        
        plt.tight_layout()
        return fig
    
    def create_spectrogram_plot(self, mel_spec):
        """Create beautiful mel-spectrogram visualization"""
        # Convert to numpy and remove batch dimension if present
        if isinstance(mel_spec, torch.Tensor):
            mel_spec_np = mel_spec.squeeze().cpu().numpy()
        else:
            mel_spec_np = mel_spec.squeeze()
        
        # Create plot with modern styling
        fig, ax = plt.subplots(figsize=(12, 7), facecolor='white')
        
        # Use modern colormap (magma, viridis, plasma)
        im = ax.imshow(mel_spec_np, aspect='auto', origin='lower', 
                      cmap='magma', interpolation='bilinear')
        
        # Style the plot
        ax.set_xlabel('Time Frames', fontsize=14, fontweight='bold', color='#374151')
        ax.set_ylabel('Mel Frequency Bins', fontsize=14, fontweight='bold', color='#374151')
        ax.set_title('🎵 Mel-Spectrogram (Log Scale)', fontsize=18, fontweight='bold', 
                    color='#1f2937', pad=20)
        ax.set_facecolor('#f8f9fa')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#9ca3af')
        ax.spines['bottom'].set_color('#9ca3af')
        
        # Add colorbar with modern styling
        cbar = plt.colorbar(im, ax=ax, pad=0.02)
        cbar.set_label('Log Magnitude', fontsize=12, fontweight='bold', color='#374151')
        cbar.outline.set_edgecolor('#9ca3af')
        cbar.ax.tick_params(colors='#374151')
        
        plt.tight_layout()
        return fig


def create_gradio_interface(model_path="model.pth"):
    """Create and configure beautiful Gradio interface"""
    
    # Initialize detector
    try:
        detector = KeystrokeDetector(model_path=model_path)
    except Exception as e:
        print(f"❌ Failed to initialize detector: {e}")
        print("Please ensure the model file exists and is valid.")
        raise
    
    # Custom CSS for beautiful styling
    custom_css = """
    .gradio-container {
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    
    .main-header {
        text-align: center;
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%);
        padding: 50px;
        border-radius: 25px;
        color: white;
        margin-bottom: 30px;
        box-shadow: 0 15px 50px rgba(37, 99, 235, 0.4);
        animation: headerGlow 3s ease-in-out infinite alternate;
    }
    
    @keyframes headerGlow {
        from { box-shadow: 0 15px 50px rgba(37, 99, 235, 0.4); }
        to { box-shadow: 0 15px 60px rgba(6, 182, 212, 0.5); }
    }
    
    .main-header h1 {
        font-size: 3.5em;
        margin: 0;
        text-shadow: 3px 3px 6px rgba(0,0,0,0.3);
        letter-spacing: 2px;
        font-weight: 800;
    }
    
    .main-header p {
        font-size: 1.3em;
        margin: 15px 0 0 0;
        opacity: 0.98;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .instruction-box {
        background: linear-gradient(135deg, #f97316 0%, #ec4899 100%);
        padding: 30px;
        border-radius: 20px;
        color: white;
        margin: 20px 0;
        box-shadow: 0 10px 40px rgba(249, 115, 22, 0.4);
        transition: transform 0.3s ease;
    }
    
    .instruction-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 50px rgba(236, 72, 153, 0.5);
    }
    
    .instruction-box h3 {
        margin-top: 0;
        font-size: 1.5em;
    }
    
    .status-box {
        padding: 15px;
        border-radius: 10px;
        background: #f0fdfa;
        border-left: 5px solid #14b8a6;
        font-weight: bold;
        font-size: 1.1em;
    }
    
    .record-button {
        background: linear-gradient(135deg, #2563eb 0%, #06b6d4 100%) !important;
        border: none !important;
        padding: 15px 30px !important;
        font-size: 1.2em !important;
        font-weight: bold !important;
        color: white !important;
        border-radius: 12px !important;
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .record-button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.6) !important;
    }
    
    #audio-input {
        border: 3px dashed #2563eb !important;
        border-radius: 15px !important;
        padding: 20px !important;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.05) 0%, rgba(6, 182, 212, 0.05) 100%) !important;
    }
    
    .footer-info {
        text-align: center;
        padding: 25px;
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: white;
        border-radius: 15px;
        margin-top: 30px;
        box-shadow: 0 8px 30px rgba(15, 23, 42, 0.3);
    }
    """
    
    # Create Gradio interface with Blocks
    with gr.Blocks(css=custom_css, title="⌨️ Keyboard Keystroke Detection", theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="cyan",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "Arial", "sans-serif"],
    )) as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>⌨️ Keyboard Keystroke Detection</h1>
            <p>🎵 Real-time keyboard keystroke identification from audio using deep learning 🤖</p>
        </div>
        """)
        
        # Instructions
        gr.HTML("""
        <div class="instruction-box">
            <h3>📝 Quick Start Guide</h3>
            <ol style="font-size: 1.1em; line-height: 1.8;">
                <li><strong>🎤 Click the microphone</strong> icon below to start recording</li>
                <li><strong>⌨️ Type a single key</strong> on your keyboard (works with DJI mic or any microphone)</li>
                <li><strong>🛑 Stop recording</strong> after the keystroke</li>
                <li><strong>🔍 Click "Analyze Keystroke"</strong> to get instant prediction</li>
            </ol>
            <p style="font-size: 0.95em; margin-top: 15px; opacity: 0.9;">
                💡 <strong>Tip:</strong> For best results, ensure your microphone is close to the keyboard and the environment is relatively quiet.
            </p>
        </div>
        """)
        
        # Main content area
        with gr.Row():
            # Left column - Audio input
            with gr.Column(scale=1):
                gr.Markdown("### 🎤 Audio Recording")
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="Record Keystroke Sound",
                    format="wav",
                    elem_id="audio-input",
                    show_label=False
                )
                
                submit_btn = gr.Button(
                    "🔍 Analyze Keystroke", 
                    variant="primary", 
                    size="lg",
                    elem_classes="record-button"
                )
                
                status_output = gr.Textbox(
                    label="Status",
                    placeholder="Waiting for audio input...",
                    interactive=False,
                    elem_classes="status-box"
                )
                
                # System info
                gr.Markdown(f"""
                <div style="background: linear-gradient(135deg, #f0fdfa 0%, #ecfeff 100%); padding: 20px; border-radius: 12px; margin-top: 20px; border: 2px solid #99f6e4; box-shadow: 0 4px 15px rgba(20, 184, 166, 0.1);">
                    <h4 style="margin-top: 0; color: #0f766e;">⚙️ System Information</h4>
                    <p style="color: #134e4a;"><strong>🤖 Model:</strong> {detector.model_info}</p>
                    <p style="color: #134e4a;"><strong>⚡ Device:</strong> {detector.device.upper()}</p>
                    <p style="color: #134e4a;"><strong>🎵 Sample Rate:</strong> 44.1 kHz</p>
                    <p style="color: #134e4a;"><strong>📊 Classes:</strong> {detector.model_loader.num_classes}</p>
                </div>
                """)
            
            # Right column - Prediction result
            with gr.Column(scale=1):
                gr.Markdown("### 🎯 Prediction Result")
                output_html = gr.HTML(
                    label="Detected Key",
                    value="""
                    <div style="text-align: center; padding: 60px; background: #f8f9fa; border-radius: 15px; border: 2px dashed #cbd5e1;">
                        <p style="font-size: 1.5em; color: #64748b;">
                            🎤 Record audio to see predictions here
                        </p>
                    </div>
                    """
                )
        
        # Visualization area
        gr.Markdown("---")
        gr.Markdown("## 📊 Detailed Analysis")
        
        with gr.Row():
            with gr.Column():
                confidence_plot = gr.Plot(label="📊 Confidence Distribution", show_label=True)
            
            with gr.Column():
                spectrogram_plot = gr.Plot(label="🎵 Audio Spectrogram", show_label=True)
        
        # Footer
        gr.HTML(f"""
        <div class="footer-info">
            <p style="margin: 5px 0; font-size: 1.1em;">
                <strong>🚀 Powered by Deep Learning</strong>
            </p>
            <p style="margin: 5px 0; opacity: 0.8;">
                {detector.model_info} | Running on {detector.device.upper()}
            </p>
            <p style="margin: 5px 0; opacity: 0.7; font-size: 0.9em;">
                Supports CNN and CoAtNet architectures | Auto-detects model type
            </p>
        </div>
        """)
        
        # Connect events
        submit_btn.click(
            fn=detector.process_audio,
            inputs=[audio_input],
            outputs=[output_html, confidence_plot, spectrogram_plot, status_output]
        )
        
        # Also trigger on audio change (real-time effect)
        audio_input.upload(
            fn=lambda: "✅ Audio uploaded! Click 'Analyze Keystroke' to predict.",
            inputs=None,
            outputs=status_output
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    import datetime
    
    parser = argparse.ArgumentParser(
        description="🎹 AI Keystroke Detection - Beautiful Gradio Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app_gradio.py --model model/CNN-Final.pkl
  python app_gradio.py --model model/CoAtNet-1-Phone.pkl --share
  python app_gradio.py --port 8080
        """
    )
    parser.add_argument("--model", type=str, default=None, 
                       help="Path to trained model file (.pth or .pkl)")
    parser.add_argument("--share", action="store_true", 
                       help="Create public share link (accessible from anywhere)")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the server on (default: 7860)")
    
    args = parser.parse_args()
    
    # Search for model if not specified or doesn't exist
    model_path = args.model
    
    if model_path is None or not Path(model_path).exists():
        if model_path:
            print(f"⚠️  Model not found at: {model_path}")
        print("\n🔍 Searching for model files...")
        
        # Search in common locations
        search_locations = [
            "model.pth",
            "model.pkl",
            "model/CNN-Final.pkl",
            "model/CNN-Best.pkl",
            "model/CoAtNet-1-Phone.pkl",
            "model/CoAtNet-1-Best-Phone.pkl",
        ]
        
        found_models = []
        for loc in search_locations:
            path = Path(loc)
            if path.exists():
                found_models.append(path)
        
        # Also search with glob
        for pattern in ["*.pth", "*.pkl", "model/*.pth", "model/*.pkl"]:
            found_models.extend(list(Path(".").glob(pattern)))
        
        # Remove duplicates
        found_models = list(set(found_models))
        
        if found_models:
            # Prefer CNN-Final or Best models
            priority_names = ["CNN-Final", "CNN-Best", "CoAtNet-1-Phone", "CoAtNet"]
            for priority in priority_names:
                for model in found_models:
                    if priority in str(model):
                        model_path = str(model)
                        break
                if model_path:
                    break
            
            # If no priority match, use first found
            if not model_path or not Path(model_path).exists():
                model_path = str(found_models[0])
            
            print(f"✅ Found and using model: {model_path}")
            print(f"   Available models: {len(found_models)}")
            for i, m in enumerate(found_models[:5], 1):
                print(f"   {i}. {m}")
        else:
            print("\n❌ No model files found!")
            print("\n💡 Please either:")
            print("   1. Specify model path: python app_gradio.py --model path/to/model.pth")
            print("   2. Place a model file in the current directory or 'model/' folder")
            print("\n📋 Supported formats: .pth, .pkl")
            exit(1)
    
    # Display launch info
    print("\n" + "="*70)
    print(" " * 20 + "🚀 LAUNCHING GRADIO APP")
    print("="*70)
    print(f"\n📦 Model File:     {model_path}")
    print(f"🌐 Server Port:    {args.port}")
    print(f"🔗 Public Share:   {'Yes ✅' if args.share else 'No (local only)'}")
    print(f"⏰ Launch Time:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n" + "="*70)
    
    # Create and launch interface
    try:
        print("\n🔧 Initializing model and components...")
        demo = create_gradio_interface(model_path=str(model_path))
        
        # Display URLs
        local_url = f"http://127.0.0.1:{args.port}"
        
        print("\n" + "🎉" * 35)
        print("\n" + " " * 20 + "✅ READY TO USE!")
        print("\n" + "🎉" * 35)
        print(f"\n📍 Local URL:      {local_url}")
        
        if args.share:
            print("🌐 Public URL:     Will be displayed below...")
        
        print("\n💡 To stop the server, press Ctrl+C")
        print("\n" + "-"*70 + "\n")
        
        # Save URL to file
        with open("gradio_url.txt", "w", encoding="utf-8") as f:
            f.write("⌨️ Keyboard Keystroke Detection - Gradio Interface\n")
            f.write("="*60 + "\n\n")
            f.write(f"Local URL:  {local_url}\n")
            f.write(f"Port:       {args.port}\n")
            f.write(f"Model:      {model_path}\n")
            if args.share:
                f.write("\nPublic share link will be displayed in the terminal.\n")
            f.write(f"\nStarted:    {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("\nInstructions:\n")
            f.write("1. Open the URL in your browser\n")
            f.write("2. Click the microphone icon to record\n")
            f.write("3. Press a key on your keyboard\n")
            f.write("4. Click 'Analyze Keystroke' to get prediction\n")
        
        print("📝 Info saved to: gradio_url.txt\n")
        
        # Launch the app
        demo.launch(
            share=args.share,
            server_port=args.port,
            server_name="0.0.0.0",  # Allow access from network
            show_error=True,
            quiet=False,
            favicon_path=None
        )
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Server stopped by user")
        print("👋 Thanks for using Keyboard Keystroke Detection!\n")
    except Exception as e:
        print(f"\n\n❌ Error: {e}")
        print("Please check the model file and try again.\n")
        raise
