"""
Data merger module - merges multiple segmented datasets into one.

This module allows combining multiple segmented keystroke datasets while:
- Maintaining proper ordering within each key category
- Preserving audio quality
- Creating merged summary statistics
- Handling duplicate handling and validation
"""
import os
import shutil
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
from collections import defaultdict


class DataMergerTab:
    """Tab for merging multiple segmented datasets."""
    
    def __init__(self, parent, config, audio_handler):
        self.parent = parent
        self.config = config
        self.audio = audio_handler
        
        self.selected_datasets = []
        self.is_merging = False
        
        self.build_ui()
    
    def build_ui(self):
        """Build the data merger UI."""
        # Create canvas with scrollbar
        canvas = tk.Canvas(self.parent)
        scrollbar = tk.Scrollbar(self.parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        main_frame = tk.Frame(scrollable_frame, padx=20, pady=20)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title = tk.Label(main_frame, text="Dataset Merger",
                        font=("Arial", 16, "bold"), fg="#1976D2")
        title.pack(pady=(0, 20))
        
        # Instructions
        instructions = tk.Label(main_frame, 
                               text="Select multiple segmented datasets to merge them into a single unified dataset.\n"
                                    "Files will be properly ordered and statistics will be combined.",
                               font=("Arial", 9), fg="#666", justify=tk.LEFT)
        instructions.pack(pady=(0, 15))
        
        # Dataset selection
        select_frame = tk.LabelFrame(main_frame, text="1. Select Datasets to Merge",
                                    font=("Arial", 10, "bold"), padx=15, pady=15)
        select_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        
        btn_row = tk.Frame(select_frame)
        btn_row.pack(fill=tk.X, pady=5)
        
        tk.Button(btn_row, text="Add Dataset", command=self.add_dataset,
                 bg="#2196F3", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        tk.Button(btn_row, text="Clear All", command=self.clear_datasets,
                 bg="#FF5722", fg="white", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        
        # Dataset list with scrollbar
        list_frame = tk.Frame(select_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.dataset_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set,
                                         font=("Courier", 9), height=8)
        self.dataset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.dataset_listbox.yview)
        
        remove_btn_row = tk.Frame(select_frame)
        remove_btn_row.pack(fill=tk.X, pady=5)
        
        tk.Button(remove_btn_row, text="Remove Selected", command=self.remove_selected_dataset,
                 font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        
        self.dataset_count_label = tk.Label(select_frame, text="0 datasets selected",
                                           font=("Arial", 9, "bold"), fg="#1976D2")
        self.dataset_count_label.pack(pady=5)
        
        # Output configuration
        output_frame = tk.LabelFrame(main_frame, text="2. Output Configuration",
                                    font=("Arial", 10, "bold"), padx=15, pady=15)
        output_frame.pack(fill=tk.X, pady=(0, 15))
        
        out_row = tk.Frame(output_frame)
        out_row.pack(fill=tk.X, pady=5)
        tk.Label(out_row, text="Merged Dataset Name:", width=20, anchor=tk.W).pack(side=tk.LEFT, padx=5)
        self.output_name_var = tk.StringVar(value=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tk.Entry(out_row, textvariable=self.output_name_var, width=40).pack(side=tk.LEFT, padx=5)
        
        # Options
        tk.Label(output_frame, text="Merge Options:", font=("Arial", 9, "bold")).pack(anchor=tk.W, pady=(10, 5))
        
        self.preserve_order_var = tk.BooleanVar(value=True)
        tk.Checkbutton(output_frame, text="Preserve chronological order within each key",
                      variable=self.preserve_order_var,
                      font=("Arial", 9)).pack(anchor=tk.W, padx=20)
        
        self.create_summary_var = tk.BooleanVar(value=True)
        tk.Checkbutton(output_frame, text="Create merged summary report",
                      variable=self.create_summary_var,
                      font=("Arial", 9)).pack(anchor=tk.W, padx=20)
        
        # Processing controls
        process_frame = tk.LabelFrame(main_frame, text="3. Merge Datasets",
                                     font=("Arial", 10, "bold"), padx=15, pady=15)
        process_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(process_frame, variable=self.progress_var,
                                           maximum=100, length=400)
        self.progress_bar.pack(pady=10)
        
        self.progress_label = tk.Label(process_frame, text="Ready - Select datasets to begin",
                                       font=("Arial", 10), fg="#424242")
        self.progress_label.pack(pady=5)
        
        btn_row2 = tk.Frame(process_frame)
        btn_row2.pack(pady=10)
        
        self.merge_btn = tk.Button(btn_row2, text="Start Merge",
                                   command=self.start_merge,
                                   bg="#4CAF50", fg="white",
                                   font=("Arial", 11, "bold"),
                                   width=18, height=2,
                                   state=tk.DISABLED)
        self.merge_btn.pack(side=tk.LEFT, padx=10)
        
        # Results
        results_frame = tk.LabelFrame(main_frame, text="Results",
                                     font=("Arial", 10, "bold"), padx=15, pady=15)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        results_scroll = tk.Scrollbar(results_frame)
        results_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(results_frame, height=6, font=("Courier", 9),
                                   yscrollcommand=results_scroll.set)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        results_scroll.config(command=self.results_text.yview)
    
    def add_dataset(self):
        """Add a dataset to the merge list."""
        folder = filedialog.askdirectory(
            title="Select Segmented Dataset Folder",
            initialdir=os.path.join(os.getcwd(), "recordings", "segmented")
        )
        
        if not folder:
            return
        
        # Validate that this is a segmented dataset
        if not self.validate_dataset(folder):
            messagebox.showerror("Invalid Dataset", 
                               "Selected folder does not appear to be a valid segmented dataset.\n\n"
                               "Expected structure: folders for each key (a-z, 0-9) containing .wav files")
            return
        
        # Check for duplicates
        if folder in self.selected_datasets:
            messagebox.showwarning("Duplicate", "This dataset is already in the list.")
            return
        
        # Add to list
        self.selected_datasets.append(folder)
        self.dataset_listbox.insert(tk.END, os.path.basename(folder))
        
        # Update UI
        self.update_dataset_count()
        
    def validate_dataset(self, folder: str) -> bool:
        """Validate that a folder is a valid segmented dataset."""
        # Check if folder exists
        if not os.path.isdir(folder):
            return False
        
        # Look for at least one valid key folder with wav files
        valid_keys = set('abcdefghijklmnopqrstuvwxyz0123456789')
        found_valid = False
        
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path) and item in valid_keys:
                # Check if this folder contains .wav files
                try:
                    files = os.listdir(item_path)
                    if any(f.endswith('.wav') for f in files):
                        found_valid = True
                        break
                except:
                    pass
        
        return found_valid
    
    def remove_selected_dataset(self):
        """Remove selected dataset from the list."""
        selection = self.dataset_listbox.curselection()
        if not selection:
            return
        
        index = selection[0]
        self.dataset_listbox.delete(index)
        self.selected_datasets.pop(index)
        
        self.update_dataset_count()
    
    def clear_datasets(self):
        """Clear all selected datasets."""
        self.dataset_listbox.delete(0, tk.END)
        self.selected_datasets.clear()
        self.update_dataset_count()
    
    def update_dataset_count(self):
        """Update the dataset count label."""
        count = len(self.selected_datasets)
        self.dataset_count_label.config(text=f"{count} dataset{'s' if count != 1 else ''} selected")
        
        # Enable/disable merge button
        if count >= 2:
            self.merge_btn.config(state=tk.NORMAL)
            self.progress_label.config(text=f"Ready to merge {count} datasets")
        else:
            self.merge_btn.config(state=tk.DISABLED)
            if count == 0:
                self.progress_label.config(text="Ready - Select datasets to begin")
            else:
                self.progress_label.config(text="Select at least 2 datasets to merge")
    
    def start_merge(self):
        """Start the merge process."""
        if len(self.selected_datasets) < 2:
            messagebox.showwarning("Warning", "Please select at least 2 datasets to merge.")
            return
        
        # Capture variables before threading
        params = {
            'output_name': self.output_name_var.get(),
            'preserve_order': self.preserve_order_var.get(),
            'create_summary': self.create_summary_var.get()
        }
        
        self.merge_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        self.is_merging = True
        
        thread = threading.Thread(target=self.run_merge, args=(params,), daemon=True)
        thread.start()
    
    def run_merge(self, params: dict):
        """Run the merge process in a background thread."""
        try:
            output_name = params['output_name']
            preserve_order = params['preserve_order']
            create_summary = params['create_summary']
            
            # Create output directory
            base_output = os.path.join(os.getcwd(), "recordings", "segmented")
            output_path = os.path.join(base_output, output_name)
            
            if os.path.exists(output_path):
                self._safe_ui_update(lambda: messagebox.showerror("Error", 
                    f"Output folder '{output_name}' already exists. Please choose a different name."))
                return
            
            os.makedirs(output_path, exist_ok=True)
            
            # Update progress
            self._update_progress(5, "Analyzing datasets...")
            
            # Get all keys across all datasets
            all_keys = self.get_all_keys()
            total_keys = len(all_keys)
            
            # Merge statistics
            stats = {
                'total_files': 0,
                'key_distribution': defaultdict(int),
                'dataset_sources': {}
            }
            
            # Process each key
            for i, key in enumerate(sorted(all_keys)):
                progress = 5 + (i / total_keys) * 85
                self._update_progress(progress, f"Merging key '{key}'...")
                
                # Create output folder for this key
                key_output = os.path.join(output_path, key)
                os.makedirs(key_output, exist_ok=True)
                
                # Collect all files for this key from all datasets
                files_to_merge = []
                
                for dataset_path in self.selected_datasets:
                    key_folder = os.path.join(dataset_path, key)
                    if not os.path.isdir(key_folder):
                        continue
                    
                    # Get all .wav files
                    try:
                        wav_files = [f for f in os.listdir(key_folder) if f.endswith('.wav')]
                        
                        for wav_file in wav_files:
                            source_path = os.path.join(key_folder, wav_file)
                            files_to_merge.append({
                                'path': source_path,
                                'dataset': os.path.basename(dataset_path),
                                'original_name': wav_file
                            })
                    except Exception as e:
                        print(f"Error reading {key_folder}: {e}")
                
                # Sort files if preserve_order is enabled
                if preserve_order:
                    # Sort by dataset order, then by numeric filename
                    def sort_key(item):
                        dataset_index = self.selected_datasets.index(
                            os.path.join(os.path.dirname(os.path.dirname(item['path'])))
                        )
                        try:
                            file_num = int(os.path.splitext(item['original_name'])[0])
                        except:
                            file_num = 999999
                        return (dataset_index, file_num)
                    
                    files_to_merge.sort(key=sort_key)
                
                # Copy files with new sequential numbering
                for idx, file_info in enumerate(files_to_merge):
                    dest_path = os.path.join(key_output, f"{idx}.wav")
                    shutil.copy2(file_info['path'], dest_path)
                    
                    stats['total_files'] += 1
                    stats['key_distribution'][key] += 1
                    
                    # Track source
                    if file_info['dataset'] not in stats['dataset_sources']:
                        stats['dataset_sources'][file_info['dataset']] = 0
                    stats['dataset_sources'][file_info['dataset']] += 1
            
            # Create summary if requested
            if create_summary:
                self._update_progress(95, "Creating summary report...")
                self.create_merge_summary(output_path, stats)
            
            # Complete
            self._update_progress(100, "Merge complete!")
            
            # Show results
            result_text = f"MERGE COMPLETE\n{'='*60}\n\n"
            result_text += f"Output: {output_name}\n"
            result_text += f"Total files merged: {stats['total_files']}\n"
            result_text += f"Keys with data: {len(stats['key_distribution'])}\n\n"
            
            result_text += "KEY DISTRIBUTION:\n"
            result_text += "-" * 40 + "\n"
            for key in sorted(stats['key_distribution'].keys()):
                count = stats['key_distribution'][key]
                result_text += f"  {key:5s} : {count:4d} samples\n"
            
            result_text += "\nSOURCE DATASETS:\n"
            result_text += "-" * 40 + "\n"
            for dataset, count in sorted(stats['dataset_sources'].items()):
                result_text += f"  {dataset}: {count} files\n"
            
            self._safe_ui_update(lambda: self.results_text.insert(1.0, result_text))
            
            self._safe_ui_update(lambda: messagebox.showinfo("Success", 
                f"Successfully merged {len(self.selected_datasets)} datasets!\n\n"
                f"Output: {output_name}\n"
                f"Total files: {stats['total_files']}"))
            
        except Exception as e:
            error_msg = f"Merge failed: {str(e)}"
            self._update_progress(0, error_msg)
            self._safe_ui_update(lambda: messagebox.showerror("Error", error_msg))
            import traceback
            traceback.print_exc()
        
        finally:
            self.is_merging = False
            self._safe_ui_update(lambda: self.merge_btn.config(state=tk.NORMAL))
    
    def get_all_keys(self) -> set:
        """Get all unique keys across all selected datasets."""
        all_keys = set()
        
        for dataset_path in self.selected_datasets:
            try:
                items = os.listdir(dataset_path)
                valid_keys = set('abcdefghijklmnopqrstuvwxyz0123456789')
                
                for item in items:
                    if item in valid_keys and os.path.isdir(os.path.join(dataset_path, item)):
                        all_keys.add(item)
            except Exception as e:
                print(f"Error reading {dataset_path}: {e}")
        
        return all_keys
    
    def create_merge_summary(self, output_path: str, stats: dict):
        """Create a summary file for the merged dataset."""
        summary_path = os.path.join(output_path, "merge_summary.txt")
        
        with open(summary_path, 'w') as f:
            f.write("DATASET MERGE SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Merge Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Number of source datasets: {len(self.selected_datasets)}\n")
            f.write(f"Total files merged: {stats['total_files']}\n")
            f.write(f"Keys with data: {len(stats['key_distribution'])}\n\n")
            
            f.write("SOURCE DATASETS:\n")
            f.write("-" * 80 + "\n")
            for i, dataset_path in enumerate(self.selected_datasets, 1):
                dataset_name = os.path.basename(dataset_path)
                file_count = stats['dataset_sources'].get(dataset_name, 0)
                f.write(f"  {i}. {dataset_name} ({file_count} files)\n")
            
            f.write("\n\nKEY DISTRIBUTION:\n")
            f.write("-" * 80 + "\n")
            for key in sorted(stats['key_distribution'].keys()):
                count = stats['key_distribution'][key]
                f.write(f"  {key:5s} : {count:4d} samples\n")
            
            f.write("\n\nMERGE DETAILS:\n")
            f.write("-" * 80 + "\n")
            f.write(f"Files are numbered sequentially (0.wav, 1.wav, ...) within each key folder.\n")
            f.write(f"Original ordering from source datasets has been preserved.\n")
    
    def _update_progress(self, value: float, message: str):
        """Thread-safe progress update."""
        self._safe_ui_update(lambda: self.progress_var.set(value))
        self._safe_ui_update(lambda: self.progress_label.config(text=message))
    
    def _safe_ui_update(self, func):
        """Safely update UI from any thread."""
        try:
            self.parent.after_idle(func)
        except Exception as e:
            print(f"UI update error: {e}")
