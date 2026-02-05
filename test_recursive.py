"""Test script to verify recursive session folder detection."""
import os
import sys

# Add data_pipeline to path
sys.path.insert(0, os.path.dirname(__file__))

from data_pipeline.data_segmenter import DataSegmenterTab

class MockAudioHandler:
    """Mock audio handler for testing."""
    pass

class MockConfig:
    """Mock config for testing."""
    pass

class MockParent:
    """Mock parent for testing."""
    def after(self, delay, func):
        func()

def test_find_sessions():
    """Test the find_session_folders method."""
    # Create mock objects
    parent = MockParent()
    config = MockConfig()
    audio = MockAudioHandler()
    
    # Create segmenter (without building UI)
    segmenter = DataSegmenterTab.__new__(DataSegmenterTab)
    segmenter.parent = parent
    segmenter.config = config
    segmenter.audio = audio
    
    # Test with recordings folder
    test_folder = r"c:\temp\keyboard_sound\recordings"
    if os.path.exists(test_folder):
        print(f"Testing folder: {test_folder}\n")
        sessions = segmenter.find_session_folders(test_folder)
        
        print(f"Found {len(sessions)} session(s):\n")
        for session in sessions:
            # Get relative path for cleaner display
            rel_path = os.path.relpath(session, test_folder)
            print(f"  • {rel_path}")
            
            # Verify it has required files
            audio_file = os.path.join(session, 'audio.wav')
            log_file = os.path.join(session, 'keystroke_log.csv')
            
            has_audio = os.path.exists(audio_file)
            has_log = os.path.exists(log_file)
            
            print(f"    - audio.wav: {'✓' if has_audio else '✗'}")
            print(f"    - keystroke_log.csv: {'✓' if has_log else '✗'}")
            print()
    else:
        print(f"Test folder not found: {test_folder}")
    
    # Test with backups folder
    test_folder2 = r"c:\temp\keyboard_sound\backups"
    if os.path.exists(test_folder2):
        print(f"\nTesting folder: {test_folder2}\n")
        sessions = segmenter.find_session_folders(test_folder2)
        
        print(f"Found {len(sessions)} session(s):\n")
        for session in sessions:
            rel_path = os.path.relpath(session, test_folder2)
            print(f"  • {rel_path}")

if __name__ == "__main__":
    test_find_sessions()
