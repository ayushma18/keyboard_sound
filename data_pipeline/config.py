"""
Configuration management module for keyboard acoustic research tool.
"""
import json
import os
from typing import Any, Optional


class Config:
    """Manage application configuration with JSON persistence."""
    
    def __init__(self, config_file: str = "recording_config.json"):
        self.config_file = config_file
        self.config = {}
        self.load()
    
    def load(self) -> None:
        """Load configuration from file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    self.config = json.load(f)
            except Exception as e:
                print(f"Error loading config: {e}")
                self.config = {}
        else:
            self.set_defaults()
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def set_defaults(self) -> None:
        """Set default configuration values."""
        self.config = {
            'sample_rate': 44100,
            'channels': 2,
            'base_output_dir': 'recordings',
            'mic_id': 'mic1',
            'keyboard_id': 'kb1',
            'input_device': None,
            'output_device': None
        }
        self.save()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value and save."""
        self.config[key] = value
        self.save()
    
    def update(self, updates: dict) -> None:
        """Update multiple configuration values."""
        self.config.update(updates)
        self.save()
