"""
Main application - Keyboard Acoustic Research Tool - Data Pipeline Edition
Integrated tool with all features in one tabbed interface.
"""
import sys
import os

# Add data_pipeline to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data_pipeline'))

import tkinter as tk
from tkinter import ttk, messagebox
from data_pipeline.config import Config
from data_pipeline.audio_handler import AudioHandler
from data_pipeline.data_collector import DataCollectorTab
from data_pipeline.data_segmenter import DataSegmenterTab
from data_pipeline.data_cleanup import DataCleanupApp
from data_pipeline.data_analyzer import AudioAnalyzer
from data_pipeline.data_merger import DataMergerTab


class KeyboardAcousticApp:
    """Main application with integrated tabbed interface."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Keyboard Acoustic Research Tool - Data Pipeline v2.0")
        self.root.geometry("1400x900")
        
        # Initialize core components
        self.config = Config()
        self.audio = AudioHandler(
            sample_rate=self.config.get('sample_rate', 44100),
            channels=self.config.get('channels', 2)
        )
        
        # Set audio devices if configured
        input_dev = self.config.get('input_device')
        output_dev = self.config.get('output_device')
        if input_dev is not None or output_dev is not None:
            self.audio.set_device(input_dev, output_dev)
        
        # Reference to current tab modules
        self.cleanup_app = None
        self.analyzer_app = None
        self.merger_tab = None
        
        self.build_ui()
    
    def build_ui(self):
        """Build the main UI with tabs."""
        # Menu bar
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Settings", command=self.show_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        
        # Main container
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create notebook (tabbed interface)
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Bind tab change event to lazy load heavy tabs
        self.notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
        # Tab 1: Data Collection (Continuous Recording)
        self.collector_frame = tk.Frame(self.notebook)
        self.notebook.add(self.collector_frame, text="📹 Data Collection")
        self.collector_tab = DataCollectorTab(self.collector_frame, self.config, self.audio)
        
        # Tab 2: Data Segmentation
        self.segmenter_frame = tk.Frame(self.notebook)
        self.notebook.add(self.segmenter_frame, text="✂️ Data Segmentation")
        self.segmenter_tab = DataSegmenterTab(self.segmenter_frame, self.config, self.audio)
        
        # Tab 3: Dataset Merger (lazy loaded)
        self.merger_frame = tk.Frame(self.notebook)
        self.notebook.add(self.merger_frame, text="🔀 Dataset Merger")
        self.merger_loaded = False
        
        # Tab 4: Data Cleanup (lazy loaded)
        self.cleanup_frame = tk.Frame(self.notebook)
        self.notebook.add(self.cleanup_frame, text="🧹 Data Cleanup")
        self.cleanup_loaded = False
        
        # Tab 5: Data Analyzer (lazy loaded)
        self.analyzer_frame = tk.Frame(self.notebook)
        self.notebook.add(self.analyzer_frame, text="📊 Data Analyzer")
        self.analyzer_loaded = False
        
        # Status bar
        self.status_bar = tk.Label(self.root, text="Ready", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def on_tab_changed(self, event):
        """Handle tab changes - lazy load heavy modules."""
        selected_tab = self.notebook.index(self.notebook.select())
        
        # Tab 2 (index 2) = Dataset Merger
        if selected_tab == 2 and not self.merger_loaded:
            self.load_merger_tab()
        
        # Tab 3 (index 3) = Data Cleanup
        elif selected_tab == 3 and not self.cleanup_loaded:
            self.load_cleanup_tab()
        
        # Tab 4 (index 4) = Data Analyzer
        elif selected_tab == 4 and not self.analyzer_loaded:
            self.load_analyzer_tab()
    
    def load_cleanup_tab(self):
        """Lazy load the cleanup module."""
        try:
            self.status_bar.config(text="Loading Data Cleanup tool...")
            self.root.update()
            
            # Embed cleanup app in frame
            self.cleanup_app = DataCleanupApp(self.cleanup_frame)
            self.cleanup_loaded = True
            
            self.status_bar.config(text="Data Cleanup tool loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load cleanup tool:\n{e}")
            self.status_bar.config(text="Error loading cleanup tool")
    
    def load_merger_tab(self):
        """Lazy load the merger module."""
        try:
            self.status_bar.config(text="Loading Dataset Merger tool...")
            self.root.update()
            
            # Embed merger app in frame
            self.merger_tab = DataMergerTab(self.merger_frame, self.config, self.audio)
            self.merger_loaded = True
            
            self.status_bar.config(text="Dataset Merger tool loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load merger tool:\n{e}")
            self.status_bar.config(text="Error loading merger tool")
    
    def load_analyzer_tab(self):
        """Lazy load the analyzer module."""
        try:
            self.status_bar.config(text="Loading Data Analyzer tool...")
            self.root.update()
            
            # Embed analyzer app in frame
            self.analyzer_app = AudioAnalyzer(self.analyzer_frame)
            self.analyzer_loaded = True
            
            self.status_bar.config(text="Data Analyzer tool loaded successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load analyzer tool:\n{e}")
            self.status_bar.config(text="Error loading analyzer tool")
    
    def show_settings(self):
        """Show settings dialog."""
        settings_window = tk.Toplevel(self.root)
        settings_window.title("Settings")
        settings_window.geometry("500x400")
        settings_window.transient(self.root)
        settings_window.grab_set()
        
        main_frame = tk.Frame(settings_window, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        tk.Label(main_frame, text="Audio Settings", font=("Arial", 14, "bold")).pack(pady=(0, 15))
        
        # Sample rate
        rate_frame = tk.Frame(main_frame)
        rate_frame.pack(fill=tk.X, pady=5)
        tk.Label(rate_frame, text="Sample Rate:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        rate_var = tk.StringVar(value=str(self.config.get('sample_rate', 44100)))
        rate_combo = ttk.Combobox(rate_frame, textvariable=rate_var, 
                                  values=["22050", "44100", "48000"], state="readonly")
        rate_combo.pack(side=tk.LEFT, padx=10)
        
        # Channels
        ch_frame = tk.Frame(main_frame)
        ch_frame.pack(fill=tk.X, pady=5)
        tk.Label(ch_frame, text="Channels:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        ch_var = tk.StringVar(value=str(self.config.get('channels', 2)))
        ch_combo = ttk.Combobox(ch_frame, textvariable=ch_var,
                               values=["1", "2"], state="readonly")
        ch_combo.pack(side=tk.LEFT, padx=10)
        
        # Output directory
        dir_frame = tk.Frame(main_frame)
        dir_frame.pack(fill=tk.X, pady=5)
        tk.Label(dir_frame, text="Output Directory:", width=20, anchor=tk.W).pack(side=tk.LEFT)
        dir_var = tk.StringVar(value=self.config.get('base_output_dir', 'recordings'))
        tk.Entry(dir_frame, textvariable=dir_var, width=30).pack(side=tk.LEFT, padx=10)
        
        # Save button
        def save_settings():
            self.config.update({
                'sample_rate': int(rate_var.get()),
                'channels': int(ch_var.get()),
                'base_output_dir': dir_var.get()
            })
            messagebox.showinfo("Settings", "Settings saved! Restart app for changes to take effect.")
            settings_window.destroy()
        
        tk.Button(main_frame, text="Save", command=save_settings,
                 bg="#4CAF50", fg="white", font=("Arial", 10, "bold")).pack(pady=20)
    
    def show_about(self):
        """Show about dialog."""
        about_text = """Keyboard Acoustic Research Tool - Data Pipeline v2.0
        
A comprehensive tool for keyboard acoustic research:
• Continuous data collection with keystroke logging
• Flexible data segmentation
• Intelligent data cleanup with noise detection
• Advanced audio analysis with spectrograms

Modular architecture with integrated tabs for efficiency.

© 2026 - Research Tool"""
        
        messagebox.showinfo("About", about_text)


def main():
    """Main entry point."""
    try:
        print("Creating Tkinter root window...")
        root = tk.Tk()
        print("Root window created successfully")
        print("Initializing KeyboardAcousticApp...")
        app = KeyboardAcousticApp(root)
        print("App initialized successfully")
        print("Starting mainloop...")
        root.mainloop()
        print("Mainloop ended")
    except Exception as e:
        print(f"ERROR: Application failed to start!")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        import traceback
        traceback.print_exc()
        input("Press Enter to exit...")


if __name__ == "__main__":
    main()
