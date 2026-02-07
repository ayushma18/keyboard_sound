"""
Keystroke logger module for capturing keyboard events with timestamps.
"""
import threading
import queue
from datetime import datetime
from pynput import keyboard
from typing import Callable, Optional


class KeystrokeLogger:
    """Handles keyboard event logging with precise timestamps."""
    
    def __init__(self, on_key_press: Optional[Callable[[str, float], None]] = None):
        self.on_key_press = on_key_press
        self.listener = None
        self.is_listening = False
        self.keystroke_queue = queue.Queue()
        self.pressed_keys = {}
        self.debounce_time = 0.15  # seconds
        
    def start(self) -> bool:
        """Start listening to keyboard events."""
        if self.is_listening:
            return False
        
        try:
            self.is_listening = True
            self.listener = keyboard.Listener(on_press=self._on_key_press)
            self.listener.start()
            return True
        except Exception as e:
            print(f"Failed to start keyboard listener: {e}")
            self.is_listening = False
            return False
    
    def stop(self) -> None:
        """Stop listening to keyboard events."""
        if self.listener:
            self.listener.stop()
            self.is_listening = False
    
    def _on_key_press(self, key) -> None:
        """Handle key press event."""
        try:
            timestamp = datetime.now().timestamp()
            
            # Get key label
            try:
                key_label = key.char
            except AttributeError:
                key_label = str(key).replace('Key.', '')
            
            # Debouncing
            last_press = self.pressed_keys.get(key_label, 0)
            if timestamp - last_press < self.debounce_time:
                return
            
            self.pressed_keys[key_label] = timestamp
            
            # Add to queue
            self.keystroke_queue.put((key_label, timestamp))
            
            # Call callback if provided
            if self.on_key_press:
                self.on_key_press(key_label, timestamp)
                
        except Exception as e:
            print(f"Error processing key press: {e}")
    
    def get_keystroke(self, block: bool = True, timeout: Optional[float] = None):
        """Get next keystroke from queue."""
        try:
            return self.keystroke_queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def clear_queue(self) -> None:
        """Clear all pending keystrokes."""
        while not self.keystroke_queue.empty():
            try:
                self.keystroke_queue.get_nowait()
            except queue.Empty:
                break
