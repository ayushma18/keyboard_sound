"""
Gradio Web Interface for Keyboard Keystroke Detection
Auto-detects CNN or CoAtNet model architectures and provides real-time predictions
"""

import gradio as gr
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import io
from PIL import Image

# Import components from existing UI
from keystroke_detector_ui import AudioPreprocessor, CNN, MyCoAtNet


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
            predicted_key: str
            confidence_plot: matplotlib figure
            spectrogram_plot: matplotlib figure
        """
        if audio_tuple is None:
            return "No audio recorded", None, None
        
        sample_rate, audio_data = audio_tuple
        
        # Convert to float32 and normalize
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        # Handle stereo: take first channel
        if len(audio_data.shape) > 1:
            audio_data = audio_data[:, 0]
        
        # Resample if needed
        if sample_rate != self.sample_rate:
            import librosa
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
        
        # Generate visualizations
        confidence_plot = self.create_confidence_plot(probs_np, self.key_labels)
        spectrogram_plot = self.create_spectrogram_plot(mel_spec)
        
        # Format result
        result_text = f"""
# Predicted Key: **{predicted_key.upper()}**

**Confidence:** {confidence*100:.2f}%

**Model:** {self.model_info}  
**Device:** {self.device.upper()}
"""
        
        return result_text, confidence_plot, spectrogram_plot
    
    def create_confidence_plot(self, probabilities, labels, top_k=5):
        """Create bar chart of top K predictions"""
        # Get top K predictions
        top_indices = np.argsort(probabilities)[-top_k:][::-1]
        top_probs = probabilities[top_indices]
        top_labels = [labels[i] for i in top_indices]
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#2ecc71' if i == 0 else '#3498db' for i in range(len(top_probs))]
        bars = ax.barh(range(len(top_probs)), top_probs, color=colors)
        
        ax.set_yticks(range(len(top_probs)))
        ax.set_yticklabels([label.upper() for label in top_labels], fontsize=14, fontweight='bold')
        ax.set_xlabel('Confidence', fontsize=12)
        ax.set_title('Top 5 Predictions', fontsize=14, fontweight='bold')
        ax.set_xlim(0, 1.0)
        ax.grid(axis='x', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, prob) in enumerate(zip(bars, top_probs)):
            ax.text(prob + 0.01, bar.get_y() + bar.get_height()/2, 
                   f'{prob*100:.1f}%', 
                   va='center', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_spectrogram_plot(self, mel_spec):
        """Create mel-spectrogram visualization"""
        # Convert to numpy and remove batch dimension if present
        if isinstance(mel_spec, torch.Tensor):
            mel_spec_np = mel_spec.squeeze().cpu().numpy()
        else:
            mel_spec_np = mel_spec.squeeze()
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(mel_spec_np, aspect='auto', origin='lower', cmap='viridis')
        ax.set_xlabel('Time Frames', fontsize=12)
        ax.set_ylabel('Mel Frequency Bins', fontsize=12)
        ax.set_title('Mel-Spectrogram (Log Scale)', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Log Magnitude', fontsize=11)
        
        plt.tight_layout()
        return fig


def create_gradio_interface(model_path="model.pth"):
    """Create and configure Gradio interface"""
    
    # Initialize detector
    try:
        detector = KeystrokeDetector(model_path=model_path)
    except Exception as e:
        print(f"Failed to initialize detector: {e}")
        print("Please ensure the model file exists and is valid.")
        raise
    
    # Create Gradio interface
    with gr.Blocks(title="Keystroke Detection") as demo:
        
        gr.Markdown(
            """
            # 🎹 Keyboard Keystroke Detection using CNN
            
            This system uses deep learning to identify keyboard keystrokes from audio recordings.
            The model automatically detects whether you're using a CNN or CoAtNet architecture.
            
            ### How to Use:
            1. Click the **microphone icon** to record a keystroke sound
            2. The system will process the audio and display:
               - **Predicted key** with confidence score
               - **Top 5 predictions** as a bar chart
               - **Mel-spectrogram** showing the audio features used by the model
            
            ### For Demo Defense:
            - The Mel-spectrogram visualizes how audio is transformed into a 2D image (64×64) for CNN processing
            - The confidence scores show the model's certainty for each prediction
            - The system works with both CNN and CoAtNet architectures
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="numpy",
                    label="🎤 Record Keystroke Audio",
                    format="wav"
                )
                
                submit_btn = gr.Button("🔍 Analyze Keystroke", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_text = gr.Markdown(label="Prediction Result")
        
        with gr.Row():
            confidence_plot = gr.Plot(label="📊 Top 5 Predictions")
        
        with gr.Row():
            spectrogram_plot = gr.Plot(label="🎵 Mel-Spectrogram Visualization")
        
        # Add footer with info
        gr.Markdown(
            f"""
            ---
            **System Info:** {detector.model_info} | **Device:** {detector.device.upper()} | **Sample Rate:** 44.1 kHz
            
            *This interface integrates with the existing AudioPreprocessor and ModelLoader components.*
            """
        )
        
        # Connect button to processing function
        submit_btn.click(
            fn=detector.process_audio,
            inputs=[audio_input],
            outputs=[output_text, confidence_plot, spectrogram_plot]
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Keystroke Detection Gradio Interface")
    parser.add_argument("--model", type=str, default="model.pth", 
                       help="Path to trained model file (default: model.pth)")
    parser.add_argument("--share", action="store_true", 
                       help="Create public share link")
    parser.add_argument("--port", type=int, default=7860,
                       help="Port to run the server on (default: 7860)")
    
    args = parser.parse_args()
    
    # Search for model if default doesn't exist
    model_path = args.model
    if not Path(model_path).exists():
        print(f"Model not found at: {model_path}")
        print("Searching for model files...")
        
        # Common model locations and names
        search_patterns = [
            "model.pth",
            "model.pkl",
            "*.pth",
            "*.pkl",
            "model/*.pth",
            "model/*.pkl",
        ]
        
        found_models = []
        for pattern in search_patterns:
            found_models.extend(list(Path(".").glob(pattern)))
        
        if found_models:
            model_path = str(found_models[0])
            print(f"✓ Found model: {model_path}")
        else:
            print("❌ No model files found!")
            print("Please specify model path with --model argument")
            exit(1)
    
    # Create and launch interface
    print("\n" + "="*60)
    print("Starting Gradio Interface...")
    print("="*60 + "\n")
    
    demo = create_gradio_interface(model_path=model_path)
    
    # Display URLs prominently BEFORE launch
    local_url = f"http://127.0.0.1:{args.port}"
    
    print("\n" + "🚀"*30)
    print("   GRADIO APP LAUNCHING...")
    print("🚀"*30)
    print(f"\n📍 LOCAL URL:  {local_url}")
    if args.share:
        print(f"🌐 PUBLIC URL: Will be generated and shown below...")
    print("\n" + "="*60 + "\n")
    
    # Save URL to file for easy access
    with open("gradio_url.txt", "w") as f:
        f.write(f"Gradio Keyboard Keystroke Detection App\n")
        f.write(f"="*50 + "\n\n")
        f.write(f"Local URL: {local_url}\n")
        f.write(f"Port: {args.port}\n")
        if args.share:
            f.write(f"\nPublic share link will be displayed in terminal\n")
        f.write(f"\nModel: {model_path}\n")
        f.write(f"Started: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print("📝 URL info saved to: gradio_url.txt\n")
    
    # Launch the app (this will block and show Gradio's URLs)
    demo.launch(
        share=args.share,
        server_port=args.port,
        show_error=True,
        theme=gr.themes.Soft()
    )
