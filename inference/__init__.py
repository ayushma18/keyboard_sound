"""
Inference module for keyboard keystroke detection
Contains Gradio app and keystroke detector UI
"""

from .app_gradio import create_gradio_interface, KeystrokeDetector

__all__ = ['create_gradio_interface', 'KeystrokeDetector']
