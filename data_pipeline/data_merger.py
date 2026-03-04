"""
Data merger - smart per-key balanced sampling + distribution visualisation.

New features
------------
* Key selection  - choose which keys (a-z / 0-9) to include.
* Dataset analysis table - shows per-key WAV counts for every source dataset.
* Sampling strategy
    - Max samples slider/spinbox (0-300, 0 = unlimited).
    - Balance mode: for each key cap all datasets at the minimum non-zero count
      across them so no single dataset dominates (removes sample bias).
* Distribution visualisation - inline bar chart + statistics panel,
  updated after every Analyze or Merge.
"""
import os
import shutil
import threading
from datetime import datetime
from typing import List, Dict
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from collections import defaultdict

ALL_ALPHA = list("abcdefghijklmnopqrstuvwxyz")
ALL_NUM   = list("0123456789")
ALL_KEYS  = ALL_ALPHA + ALL_NUM


# ---------------------------------------------------------------------------
# Lightweight canvas bar-chart (no matplotlib dependency)
# ---------------------------------------------------------------------------

class BarChart(tk.Canvas):
    """Draw a horizontal bar chart directly on a tk.Canvas."""
    BAR_COLOR  = "#1976D2"
    ZERO_COLOR = "#BDBDBD"
    TEXT_COLOR = "#212121"
    LABEL_W    = 28
    BAR_H      = 16
    BAR_GAP    = 4
    PAD_X      = 10
    PAD_Y      = 10

    def draw(self, distribution: Dict[str, int]):
        self.delete("all")
        if not distribution:
            self.create_text(10, 10, text="No data", anchor="nw", fill="#999")
            return
        keys   = sorted(distribution.keys())
        counts = [distribution[k] for k in keys]
        max_c  = max(counts) if counts else 1
        w      = int(self.cget("width"))
        usable = max(10, w - self.LABEL_W - 50 - self.PAD_X * 2)
        total_h = self.PAD_Y * 2 + len(keys) * (self.BAR_H + self.BAR_GAP)
        self.config(height=max(total_h, 60))
        for i, (key, count) in enumerate(zip(keys, counts)):
            y     = self.PAD_Y + i * (self.BAR_H + self.BAR_GAP)
            bw    = int((count / max_c) * usable) if max_c else 0
            x_bar = self.PAD_X + self.LABEL_W
            self.create_text(x_bar - 4, y + self.BAR_H // 2,
                             text=key, anchor="e",
                             font=("Courier", 8, "bold"), fill=self.TEXT_COLOR)
            color = self.BAR_COLOR if count > 0 else self.ZERO_COLOR
            self.create_rectangle(x_bar, y, x_bar + max(bw, 2),
                                  y + self.BAR_H, fill=color, outline="")
            self.create_text(x_bar + bw + 4, y + self.BAR_H // 2,
                             text=str(count), anchor="w",
                             font=("Courier", 8), fill=self.TEXT_COLOR)


# ---------------------------------------------------------------------------
# Main merger tab
# ---------------------------------------------------------------------------

class DataMergerTab:
    """Merge multiple segmented datasets with key selection and balanced sampling."""

    def __init__(self, parent, config, audio_handler):
        self.parent   = parent
        self.config   = config
        self.audio    = audio_handler
        self.selected_datasets: List[str] = []
        self.is_merging = False
        # Per-dataset WAV-count cache: {dataset_path: {key: count}}
        self._analysis_cache: Dict[str, Dict[str, int]] = {}
        self.build_ui()

    # -----------------------------------------------------------------------
    # UI BUILD
    # -----------------------------------------------------------------------

    def build_ui(self):
        oc = tk.Canvas(self.parent)
        vsb = tk.Scrollbar(self.parent, orient="vertical", command=oc.yview)
        sf  = tk.Frame(oc)
        sf.bind("<Configure>",
                lambda e: oc.configure(scrollregion=oc.bbox("all")))
        oc.create_window((0, 0), window=sf, anchor="nw")
        oc.configure(yscrollcommand=vsb.set)
        vsb.pack(side="right", fill="y")
        oc.pack(side="left", fill="both", expand=True)
        oc.bind_all("<MouseWheel>",
                    lambda e: oc.yview_scroll(int(-1*(e.delta/120)), "units"))

        main = tk.Frame(sf, padx=20, pady=20)
        main.pack(fill=tk.BOTH, expand=True)

        tk.Label(main, text="Dataset Merger",
                 font=("Arial", 16, "bold"), fg="#1976D2").pack(pady=(0, 5))
        tk.Label(main,
                 text="Select datasets  ->  choose keys  ->  set limits  ->  analyze  ->  merge",
                 font=("Arial", 9), fg="#666").pack(pady=(0, 15))

        self._build_sec_datasets(main)
        self._build_sec_keys(main)
        self._build_sec_sampling(main)
        self._build_sec_analyze(main)
        self._build_sec_merge(main)
        self._build_sec_distribution(main)

    # -- Section 1: datasets -------------------------------------------------

    def _build_sec_datasets(self, p):
        fr = tk.LabelFrame(p, text="1. Select Datasets to Merge",
                           font=("Arial", 10, "bold"), padx=15, pady=15)
        fr.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        br = tk.Frame(fr)
        br.pack(fill=tk.X, pady=5)
        tk.Button(br, text="Add Dataset", command=self.add_dataset,
                  bg="#2196F3", fg="white",
                  font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)
        tk.Button(br, text="Remove Selected", command=self.remove_selected_dataset,
                  font=("Arial", 9)).pack(side=tk.LEFT, padx=5)
        tk.Button(br, text="Clear All", command=self.clear_datasets,
                  bg="#FF5722", fg="white",
                  font=("Arial", 9, "bold")).pack(side=tk.LEFT, padx=5)
        lf = tk.Frame(fr)
        lf.pack(fill=tk.BOTH, expand=True, pady=5)
        sb = tk.Scrollbar(lf)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dataset_listbox = tk.Listbox(lf, yscrollcommand=sb.set,
                                          font=("Courier", 9), height=6,
                                          selectmode=tk.EXTENDED)
        self.dataset_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.config(command=self.dataset_listbox.yview)
        self.dataset_count_label = tk.Label(fr, text="0 datasets selected",
                                            font=("Arial", 9, "bold"), fg="#1976D2")
        self.dataset_count_label.pack(pady=3)

    # -- Section 2: key selection --------------------------------------------

    def _build_sec_keys(self, p):
        fr = tk.LabelFrame(p, text="2. Select Keys to Include in Final Dataset",
                           font=("Arial", 10, "bold"), padx=15, pady=15)
        fr.pack(fill=tk.X, pady=(0, 15))
        qs = tk.Frame(fr)
        qs.pack(fill=tk.X, pady=(0, 8))
        for lbl, fn in [("All",          lambda: self._key_set_all(True)),
                        ("None",         lambda: self._key_set_all(False)),
                        ("Alpha only",   lambda: self._key_set_group("alpha")),
                        ("Numeric only", lambda: self._key_set_group("num"))]:
            tk.Button(qs, text=lbl, command=fn,
                      font=("Arial", 8), width=12).pack(side=tk.LEFT, padx=2)
        self._key_vars: Dict[str, tk.BooleanVar] = {}
        af = tk.LabelFrame(fr, text="Letters  (a-z)", font=("Arial", 8))
        af.pack(fill=tk.X, pady=3)
        for i, k in enumerate(ALL_ALPHA):
            v = tk.BooleanVar(value=True)
            self._key_vars[k] = v
            tk.Checkbutton(af, text=k, variable=v,
                           font=("Courier", 9), width=3).grid(
                               row=i//13, column=i%13, sticky="w", padx=2)
        nf = tk.LabelFrame(fr, text="Digits  (0-9)", font=("Arial", 8))
        nf.pack(fill=tk.X, pady=3)
        for i, k in enumerate(ALL_NUM):
            v = tk.BooleanVar(value=True)
            self._key_vars[k] = v
            tk.Checkbutton(nf, text=k, variable=v,
                           font=("Courier", 9), width=3).grid(
                               row=0, column=i, sticky="w", padx=2)

    # -- Section 3: sampling strategy ----------------------------------------

    def _build_sec_sampling(self, p):
        fr = tk.LabelFrame(p,
                           text="3. Sampling Strategy  (per key, per dataset)",
                           font=("Arial", 10, "bold"), padx=15, pady=15)
        fr.pack(fill=tk.X, pady=(0, 15))
        # -- Per-dataset cap row
        mr = tk.Frame(fr)
        mr.pack(fill=tk.X, pady=5)
        tk.Label(mr, text="Max samples per key per dataset:",
                 font=("Arial", 9), width=32, anchor=tk.W).pack(side=tk.LEFT)
        self.max_samples_var = tk.IntVar(value=300)
        tk.Scale(mr, from_=0, to=300, orient=tk.HORIZONTAL,
                 variable=self.max_samples_var,
                 length=200, showvalue=False).pack(side=tk.LEFT, padx=5)
        sp = tk.Spinbox(mr, from_=0, to=300,
                        textvariable=self.max_samples_var,
                        width=5, font=("Arial", 10),
                        command=self._sync_slider)
        sp.pack(side=tk.LEFT, padx=5)
        sp.bind("<FocusOut>", lambda e: self._sync_slider())
        tk.Label(mr, text="(0 = unlimited)",
                 font=("Arial", 8), fg="#888").pack(side=tk.LEFT)
        # -- Mode: Combine All vs Balance
        mode_fr = tk.LabelFrame(fr, text="Combine Mode", font=("Arial", 9, "bold"))
        mode_fr.pack(fill=tk.X, padx=5, pady=(8, 5))
        self.balance_var = tk.StringVar(value="balance")
        tk.Radiobutton(
            mode_fr,
            text="Combine All  --  concatenate every available sample from all datasets (no equalisation)",
            variable=self.balance_var, value="combine",
            font=("Arial", 9), anchor="w",
            command=self._update_strategy_label
        ).pack(anchor=tk.W, padx=10, pady=2)
        tk.Radiobutton(
            mode_fr,
            text="Balance  --  for each key cap all datasets at the minimum count\n"
                 "               (e.g. 300 vs 100 samples -> take 100 from each; removes bias)",
            variable=self.balance_var, value="balance",
            font=("Arial", 9), justify=tk.LEFT, anchor="w",
            command=self._update_strategy_label
        ).pack(anchor=tk.W, padx=10, pady=2)
        # -- Final max per key (total cap across all datasets combined)
        fm_row = tk.Frame(fr)
        fm_row.pack(fill=tk.X, pady=(8, 0))
        tk.Label(fm_row, text="Final max per key (total across all datasets):",
                 font=("Arial", 9), width=44, anchor=tk.W).pack(side=tk.LEFT)
        self.final_max_var = tk.IntVar(value=0)
        tk.Scale(fm_row, from_=0, to=2000, orient=tk.HORIZONTAL,
                 variable=self.final_max_var,
                 length=200, showvalue=False).pack(side=tk.LEFT, padx=5)
        fm_sp = tk.Spinbox(fm_row, from_=0, to=99999,
                           textvariable=self.final_max_var,
                           width=6, font=("Arial", 10),
                           command=self._sync_final_max)
        fm_sp.pack(side=tk.LEFT, padx=5)
        fm_sp.bind("<FocusOut>", lambda e: self._sync_final_max())
        tk.Label(fm_row, text="(0 = no final cap)",
                 font=("Arial", 8), fg="#888").pack(side=tk.LEFT)
        # -- Strategy summary label
        self.strategy_label = tk.Label(fr, text="",
                                       font=("Arial", 8, "italic"), fg="#555")
        self.strategy_label.pack(anchor=tk.W, padx=20, pady=(5, 0))
        self.max_samples_var.trace_add("write", lambda *_: self._update_strategy_label())
        self.balance_var.trace_add("write",     lambda *_: self._update_strategy_label())
        self.final_max_var.trace_add("write",   lambda *_: self._update_strategy_label())
        self._update_strategy_label()

    # -- Section 4: analyze --------------------------------------------------

    def _build_sec_analyze(self, p):
        fr = tk.LabelFrame(p,
                           text="4. Analyze Datasets  (per-key sample counts)",
                           font=("Arial", 10, "bold"), padx=15, pady=15)
        fr.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        br = tk.Frame(fr)
        br.pack(fill=tk.X, pady=5)
        self.analyze_btn = tk.Button(br, text="Analyze Selected Datasets",
                                     command=self.analyze_datasets,
                                     bg="#7B1FA2", fg="white",
                                     font=("Arial", 10, "bold"),
                                     state=tk.DISABLED)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        self.analyze_status = tk.Label(br,
                                       text="Add >= 1 dataset to enable analysis.",
                                       font=("Arial", 9), fg="#666")
        self.analyze_status.pack(side=tk.LEFT, padx=15)
        tf = tk.Frame(fr)
        tf.pack(fill=tk.BOTH, expand=True, pady=5)
        txsb = tk.Scrollbar(tf, orient=tk.HORIZONTAL)
        txsb.pack(side=tk.BOTTOM, fill=tk.X)
        tysb = tk.Scrollbar(tf, orient=tk.VERTICAL)
        tysb.pack(side=tk.RIGHT, fill=tk.Y)
        self.analysis_tree = ttk.Treeview(tf, show="headings", height=12,
                                          xscrollcommand=txsb.set,
                                          yscrollcommand=tysb.set)
        self.analysis_tree.pack(fill=tk.BOTH, expand=True)
        txsb.config(command=self.analysis_tree.xview)
        tysb.config(command=self.analysis_tree.yview)
        # Row colour tags
        self.analysis_tree.tag_configure("zero",    background="#FFEBEE")  # red
        self.analysis_tree.tag_configure("limited", background="#FFF8E1")  # amber
        self.analysis_tree.tag_configure("ok",      background="#E8F5E9")  # green

    # -- Section 5: output & merge -------------------------------------------

    def _build_sec_merge(self, p):
        fr = tk.LabelFrame(p, text="5. Output & Merge",
                           font=("Arial", 10, "bold"), padx=15, pady=15)
        fr.pack(fill=tk.X, pady=(0, 15))
        or_ = tk.Frame(fr)
        or_.pack(fill=tk.X, pady=5)
        tk.Label(or_, text="Merged Dataset Name:",
                 width=22, anchor=tk.W).pack(side=tk.LEFT)
        self.output_name_var = tk.StringVar(
            value=f"merged_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        tk.Entry(or_, textvariable=self.output_name_var,
                 width=40).pack(side=tk.LEFT, padx=5)
        self.create_summary_var = tk.BooleanVar(value=True)
        tk.Checkbutton(fr,
                       text="Write merge_summary.txt into output folder",
                       variable=self.create_summary_var,
                       font=("Arial", 9)).pack(anchor=tk.W, padx=20, pady=5)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(fr, variable=self.progress_var,
                                            maximum=100, length=500)
        self.progress_bar.pack(pady=8, fill=tk.X)
        self.progress_label = tk.Label(fr,
                                       text="Ready - add datasets and click Analyze first.",
                                       font=("Arial", 10), fg="#424242")
        self.progress_label.pack(pady=3)
        self.merge_btn = tk.Button(fr, text="Start Merge",
                                   command=self.start_merge,
                                   bg="#4CAF50", fg="white",
                                   font=("Arial", 12, "bold"),
                                   height=2, state=tk.DISABLED)
        self.merge_btn.pack(pady=8)
        rf = tk.LabelFrame(fr, text="Merge Log", font=("Arial", 9, "bold"))
        rf.pack(fill=tk.BOTH, expand=True, pady=5)
        rsb = tk.Scrollbar(rf)
        rsb.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text = tk.Text(rf, height=8, font=("Courier", 8),
                                    yscrollcommand=rsb.set)
        self.results_text.pack(fill=tk.BOTH, expand=True)
        rsb.config(command=self.results_text.yview)

    # -- Section 6: distribution chart ---------------------------------------

    def _build_sec_distribution(self, p):
        fr = tk.LabelFrame(p, text="6. Final Dataset Distribution",
                           font=("Arial", 10, "bold"), padx=15, pady=15)
        fr.pack(fill=tk.BOTH, expand=True, pady=(0, 15))
        tk.Label(fr, text="Updated after every Analyze or Merge.",
                 font=("Arial", 8), fg="#888").pack(anchor=tk.W)
        cc = tk.Frame(fr)
        cc.pack(fill=tk.BOTH, expand=True)
        cvsb = tk.Scrollbar(cc, orient=tk.VERTICAL)
        cvsb.pack(side=tk.RIGHT, fill=tk.Y)
        csc = tk.Canvas(cc, bg="white", height=320, yscrollcommand=cvsb.set)
        csc.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        cvsb.config(command=csc.yview)
        self._chart_inner = tk.Frame(csc, bg="white")
        csc.create_window((0, 0), window=self._chart_inner, anchor="nw")
        self._chart_inner.bind(
            "<Configure>",
            lambda e: csc.configure(scrollregion=csc.bbox("all")))
        self.bar_chart = BarChart(self._chart_inner,
                                  bg="white", width=700, height=60)
        self.bar_chart.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        sf2 = tk.LabelFrame(fr, text="Statistics", font=("Arial", 9, "bold"))
        sf2.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        ssb = tk.Scrollbar(sf2)
        ssb.pack(side=tk.RIGHT, fill=tk.Y)
        self.dist_summary = tk.Text(sf2, height=6, font=("Courier", 8),
                                    yscrollcommand=ssb.set, state=tk.DISABLED)
        self.dist_summary.pack(fill=tk.BOTH, expand=True)
        ssb.config(command=self.dist_summary.yview)

    # -----------------------------------------------------------------------
    # HELPERS / CALLBACKS
    # -----------------------------------------------------------------------

    def _key_set_all(self, value: bool):
        for v in self._key_vars.values():
            v.set(value)

    def _key_set_group(self, group: str):
        self._key_set_all(False)
        for k in (ALL_ALPHA if group == "alpha" else ALL_NUM):
            self._key_vars[k].set(True)

    def _sync_slider(self):
        try:
            val = max(0, min(300, int(self.max_samples_var.get())))
            self.max_samples_var.set(val)
        except ValueError:
            pass

    def _sync_final_max(self):
        try:
            val = max(0, int(self.final_max_var.get()))
            self.final_max_var.set(val)
        except ValueError:
            pass

    def _update_strategy_label(self):
        mx   = self.max_samples_var.get()
        mode = self.balance_var.get()
        fm   = self.final_max_var.get()
        cap  = "take all available" if mx == 0 else f"cap at {mx} per key/dataset"
        mode_txt = "then combine all" if mode == "combine" else "then equalise at per-key minimum"
        fm_txt   = f"; final clip at {fm} per key" if fm > 0 else ""
        self.strategy_label.config(text=f"Strategy: {cap}; {mode_txt}{fm_txt}.")

    def _selected_keys(self) -> List[str]:
        return [k for k in ALL_KEYS if self._key_vars[k].get()]

    def _get_counts(self, ds_path: str, keys: List[str]) -> Dict[str, int]:
        """Return {key: wav_count} for the given dataset (cached)."""
        if ds_path in self._analysis_cache:
            return self._analysis_cache[ds_path]
        result: Dict[str, int] = {}
        for k in keys:
            folder = os.path.join(ds_path, k)
            try:
                result[k] = (
                    sum(1 for f in os.listdir(folder) if f.endswith(".wav"))
                    if os.path.isdir(folder) else 0)
            except Exception:
                result[k] = 0
        self._analysis_cache[ds_path] = result
        return result

    def _compute_takes(self, raw_counts: List[int],
                       max_cap: int, balance: str,
                       final_max: int = 0) -> List[int]:
        """
        Compute how many samples to take from each dataset for one key.

        Rules
        -----
        1. Clamp every count at max_cap (0 = unlimited).
        2. balance='balance' -> equalise all non-zero entries at their minimum.
           balance='combine' -> keep clamped values (no equalisation).
        3. If final_max > 0, proportionally scale down so sum(takes) <= final_max,
           distributing any rounding remainder to the largest contributors.
        Returns a list aligned with self.selected_datasets.
        """
        avail = [min(c, max_cap) if max_cap > 0 else c for c in raw_counts]
        if balance == "balance":
            non_zero = [v for v in avail if v > 0]
            if non_zero:
                min_count = min(non_zero)
                avail = [min_count if v > 0 else 0 for v in avail]
        # Apply final-max total cap
        if final_max > 0:
            total = sum(avail)
            if total > final_max:
                avail = [int(v * final_max / total) for v in avail]
                remainder = final_max - sum(avail)
                indices = sorted(range(len(avail)), key=lambda i: -avail[i])
                for i in range(remainder):
                    avail[indices[i % len(indices)]] += 1
        return avail

    # -----------------------------------------------------------------------
    # DATASET MANAGEMENT
    # -----------------------------------------------------------------------

    def add_dataset(self):
        folder = filedialog.askdirectory(
            title="Select Segmented Dataset Folder",
            initialdir=os.path.join(os.getcwd(), "recordings", "segmented"))
        if not folder:
            return
        if not self._validate_dataset(folder):
            messagebox.showerror("Invalid Dataset",
                                 "No recognised key sub-folders with .wav files found.")
            return
        if folder in self.selected_datasets:
            messagebox.showwarning("Duplicate", "This dataset is already in the list.")
            return
        self.selected_datasets.append(folder)
        self.dataset_listbox.insert(tk.END, os.path.basename(folder))
        self._analysis_cache.pop(folder, None)
        self._update_ui_state()

    def _validate_dataset(self, folder: str) -> bool:
        if not os.path.isdir(folder):
            return False
        valid_keys = set(ALL_KEYS)
        for item in os.listdir(folder):
            item_path = os.path.join(folder, item)
            if os.path.isdir(item_path) and item in valid_keys:
                try:
                    if any(f.endswith(".wav") for f in os.listdir(item_path)):
                        return True
                except Exception:
                    pass
        return False

    def remove_selected_dataset(self):
        selection = self.dataset_listbox.curselection()
        if not selection:
            return
        for index in reversed(selection):
            self._analysis_cache.pop(self.selected_datasets[index], None)
            self.dataset_listbox.delete(index)
            self.selected_datasets.pop(index)
        self._update_ui_state()

    def clear_datasets(self):
        self.dataset_listbox.delete(0, tk.END)
        self.selected_datasets.clear()
        self._analysis_cache.clear()
        self._update_ui_state()

    def _update_ui_state(self):
        n = len(self.selected_datasets)
        self.dataset_count_label.config(
            text=f"{n} dataset{'s' if n != 1 else ''} selected")
        self.analyze_btn.config(state=tk.NORMAL if n >= 1 else tk.DISABLED)
        self.analyze_status.config(
            text="Click 'Analyze' to refresh the table."
            if n >= 1 else "Add >= 1 dataset to enable analysis.")
        self.merge_btn.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # ANALYSIS
    # -----------------------------------------------------------------------

    def analyze_datasets(self):
        if not self.selected_datasets:
            return
        self.analyze_btn.config(state=tk.DISABLED)
        self.analyze_status.config(text="Analysing...")
        threading.Thread(target=self._run_analysis, daemon=True).start()

    def _run_analysis(self):
        keys      = self._selected_keys()
        datasets  = list(self.selected_datasets)
        max_cap   = self.max_samples_var.get()
        balance   = self.balance_var.get()
        final_max = self.final_max_var.get()

        all_counts = [self._get_counts(ds, keys) for ds in datasets]

        # Compute final "will take" per key
        will_take: Dict[str, int] = {}
        per_ds_take: Dict[str, List[int]] = {}
        for k in keys:
            raw   = [all_counts[i].get(k, 0) for i in range(len(datasets))]
            takes = self._compute_takes(raw, max_cap, balance, final_max)
            per_ds_take[k]  = takes
            will_take[k]    = sum(takes)

        def _update_tree():
            tree = self.analysis_tree
            tree.delete(*tree.get_children())
            ds_names = [os.path.basename(d) for d in datasets]
            cols     = ["Key"] + ds_names + ["Total avail.", "Will take"]
            tree["columns"] = cols
            for c in cols:
                tree.heading(c, text=c)
                tree.column(c,
                            width=50 if c == "Key"
                                  else max(70, min(140, len(c) * 8 + 20)),
                            anchor="center")
            tree.column("Will take", width=80)

            for k in sorted(keys):
                rc  = [all_counts[i].get(k, 0) for i in range(len(datasets))]
                ta  = sum(rc)
                wt  = will_take[k]
                tag = ("zero" if wt == 0
                        else "limited" if wt < ta
                        else "ok")
                tree.insert("", tk.END,
                            values=[k] + rc + [ta, wt],
                            tags=(tag,))

            # Totals footer row
            tots = [sum(all_counts[i].get(k, 0) for k in keys)
                    for i in range(len(datasets))]
            ga   = sum(tots)
            gt   = sum(will_take.values())
            tree.insert("", tk.END,
                        values=["TOTAL"] + tots + [ga, gt],
                        tags=("ok",))

            self.analyze_status.config(
                text=f"Done -- {len(keys)} keys | "
                     f"{ga} available | {gt} will be taken")
            self.analyze_btn.config(state=tk.NORMAL)
            self.merge_btn.config(state=tk.NORMAL)
            self._refresh_distribution(will_take)

        self.parent.after_idle(_update_tree)

    # -----------------------------------------------------------------------
    # MERGE
    # -----------------------------------------------------------------------

    def start_merge(self):
        if not self.selected_datasets:
            messagebox.showwarning("Warning", "Please select at least 1 dataset.")
            return
        keys = self._selected_keys()
        if not keys:
            messagebox.showwarning("Warning", "No keys selected.")
            return
        name = self.output_name_var.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter an output dataset name.")
            return
        params = dict(output_name=name, keys=keys,
                      max_cap=self.max_samples_var.get(),
                      balance=self.balance_var.get(),
                      final_max=self.final_max_var.get(),
                      create_summary=self.create_summary_var.get())
        self.merge_btn.config(state=tk.DISABLED)
        self.progress_var.set(0)
        self.results_text.delete(1.0, tk.END)
        self.is_merging = True
        threading.Thread(target=self._run_merge, args=(params,), daemon=True).start()

    def _run_merge(self, params: dict):
        try:
            name      = params["output_name"]
            keys      = params["keys"]
            max_cap   = params["max_cap"]
            balance   = params["balance"]
            final_max = params["final_max"]
            create_summary = params["create_summary"]
            datasets = list(self.selected_datasets)

            out_path = os.path.join(
                os.getcwd(), "recordings", "segmented", name)
            if os.path.exists(out_path):
                self._safe_ui(lambda: messagebox.showerror(
                    "Error",
                    f"Folder '{name}' already exists. Choose a different name."))
                return

            os.makedirs(out_path, exist_ok=True)
            self._update_progress(5, "Analysing source datasets...")

            all_counts = [self._get_counts(ds, keys) for ds in datasets]
            stats = dict(
                total_files=0,
                key_distribution=defaultdict(int),
                dataset_sources=defaultdict(int),
                key_take_limits={})

            for ki, key in enumerate(sorted(keys)):
                pct = 5 + (ki / len(keys)) * 88
                self._update_progress(pct, f"Merging key '{key}'...")

                raw   = [all_counts[i].get(key, 0)
                         for i in range(len(datasets))]
                takes = self._compute_takes(raw, max_cap, balance, final_max)
                stats["key_take_limits"][key] = takes

                key_out = os.path.join(out_path, key)
                os.makedirs(key_out, exist_ok=True)

                dest_idx = 0
                for di, ds_path in enumerate(datasets):
                    take = takes[di]
                    if take == 0:
                        continue
                    key_folder = os.path.join(ds_path, key)
                    if not os.path.isdir(key_folder):
                        continue
                    try:
                        wav_files = sorted(
                            [f for f in os.listdir(key_folder)
                             if f.endswith(".wav")],
                            key=lambda f: (
                                int(os.path.splitext(f)[0])
                                if os.path.splitext(f)[0].isdigit()
                                else 999999))[:take]
                    except Exception as e:
                        print(f"Error reading {key_folder}: {e}")
                        continue

                    ds_name = os.path.basename(ds_path)
                    for wf in wav_files:
                        shutil.copy2(
                            os.path.join(key_folder, wf),
                            os.path.join(key_out, f"{dest_idx}.wav"))
                        dest_idx += 1
                        stats["total_files"]             += 1
                        stats["key_distribution"][key]   += 1
                        stats["dataset_sources"][ds_name] += 1

            if create_summary:
                self._update_progress(95, "Writing summary report...")
                self._write_summary(
                    out_path, stats, datasets, keys, max_cap, balance,
                    final_max)

            self._update_progress(100, "Merge complete!")

            dist  = dict(stats["key_distribution"])
            max_v = max(dist.values()) if dist else 1

            rt  = f"MERGE COMPLETE\n{'='*64}\n\n"
            rt += f"Output folder  : {name}\n"
            rt += f"Keys merged    : {len(dist)}\n"
            rt += f"Total files    : {stats['total_files']}\n"
            rt += f"Max cap        : {'unlimited' if max_cap == 0 else max_cap}\n"
            rt += f"Mode           : {'combine all' if balance == 'combine' else 'balanced (equalise at min)'}\n"
            rt += f"Final clip     : {'none' if final_max == 0 else str(final_max) + ' per key'}\n\n"
            rt += f"{'KEY':<6} {'SAMPLES':>8}   DISTRIBUTION BAR\n"
            rt += "-" * 56 + "\n"
            for k in sorted(dist):
                c   = dist[k]
                bar = "#" * int(c / max_v * 32)
                rt += f"  {k:<4} {c:>6}   {bar}\n"
            rt += "\nSOURCE CONTRIBUTIONS:\n" + "-" * 40 + "\n"
            for ds, cnt in sorted(stats["dataset_sources"].items()):
                rt += f"  {ds}: {cnt} files\n"

            self._safe_ui(lambda: self.results_text.insert(1.0, rt))
            self._safe_ui(lambda: self._refresh_distribution(dist))
            self._safe_ui(lambda: messagebox.showinfo(
                "Merge Complete",
                f"Merged successfully!\n\n"
                f"Output : {name}\n"
                f"Keys   : {len(dist)}\n"
                f"Files  : {stats['total_files']}"))

        except Exception as e:
            msg = f"Merge failed: {e}"
            self._update_progress(0, msg)
            self._safe_ui(lambda: messagebox.showerror("Error", msg))
            import traceback
            traceback.print_exc()
        finally:
            self.is_merging = False
            self._safe_ui(lambda: self.merge_btn.config(state=tk.NORMAL))

    # -----------------------------------------------------------------------
    # DISTRIBUTION VISUALISATION
    # -----------------------------------------------------------------------

    def _refresh_distribution(self, distribution: Dict[str, int]):
        """Refresh the bar chart and statistics panel."""
        self.bar_chart.draw(distribution)

        total  = sum(distribution.values())
        n_keys = len([v for v in distribution.values() if v > 0])
        avg    = total / n_keys if n_keys else 0
        mn     = min(distribution.values()) if distribution else 0
        mx     = max(distribution.values()) if distribution else 0

        lines = [
            f"Keys with data : {n_keys}   |   "
            f"Total samples : {total}   |   "
            f"Avg/key : {avg:.1f}   |   "
            f"Min : {mn}   |   Max : {mx}",
            "",
            f"{'Key':<6} {'Count':>6}   {'Share':>6}   Distribution",
            "-" * 60,
        ]
        for k in sorted(distribution.keys()):
            c   = distribution[k]
            pct = c / total * 100 if total else 0
            bar = "|" * int(pct / 2)   # each pipe ~ 2 %
            lines.append(f"  {k:<4} {c:>6}   {pct:5.1f}%   {bar}")

        self.dist_summary.config(state=tk.NORMAL)
        self.dist_summary.delete(1.0, tk.END)
        self.dist_summary.insert(1.0, "\n".join(lines))
        self.dist_summary.config(state=tk.DISABLED)

    # -----------------------------------------------------------------------
    # SUMMARY FILE
    # -----------------------------------------------------------------------

    def _write_summary(self, out_path: str, stats: dict,
                       datasets: List[str], keys: List[str],
                       max_cap: int, balance: str,
                       final_max: int = 0):
        dist  = dict(stats["key_distribution"])
        total = stats["total_files"]
        max_v = max(dist.values()) if dist else 1

        with open(os.path.join(out_path, "merge_summary.txt"), "w") as f:
            f.write("DATASET MERGE SUMMARY\n" + "=" * 80 + "\n\n")
            f.write(f"Merge date         : "
                    f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Source datasets    : {len(datasets)}\n")
            f.write(f"Keys selected      : {', '.join(sorted(keys))}\n")
            f.write(f"Max cap per ds/key : "
                    f"{'unlimited' if max_cap == 0 else max_cap}\n")
            f.write(f"Combine mode       : "
                    f"{'combine all' if balance == 'combine' else 'balanced (equalise at min)'}\n")
            f.write(f"Final clip per key : "
                    f"{'none' if final_max == 0 else final_max}\n")
            f.write(f"Total files        : {total}\n\n")

            f.write("SOURCE DATASETS:\n" + "-" * 80 + "\n")
            for i, ds in enumerate(datasets, 1):
                cnt = stats["dataset_sources"].get(os.path.basename(ds), 0)
                f.write(f"  {i}. {os.path.basename(ds)} -- {cnt} files\n")

            f.write("\n\nKEY DISTRIBUTION:\n" + "-" * 80 + "\n")
            f.write(f"  {'Key':<6} {'Count':>6}  {'Share':>7}  Bar\n")
            f.write("  " + "-" * 60 + "\n")
            for k in sorted(dist.keys()):
                c   = dist[k]
                pct = c / total * 100 if total else 0
                bar = "#" * int(c / max_v * 40)
                f.write(f"  {k:<6} {c:>6}  {pct:6.1f}%  {bar}\n")

            f.write("\n\nPER-KEY TAKE LIMITS (samples taken per dataset):\n")
            f.write("-" * 80 + "\n")
            ds_names = [os.path.basename(d) for d in datasets]
            hdr = f"  {'Key':<6}" + "".join(f"  {n[:14]:>14}" for n in ds_names)
            f.write(hdr + "\n  " + "-" * len(hdr) + "\n")
            for k in sorted(keys):
                takes = stats["key_take_limits"].get(k, [0] * len(datasets))
                row   = f"  {k:<6}" + "".join(f"  {t:>14}" for t in takes)
                f.write(row + "\n")

    # -----------------------------------------------------------------------
    # UTILITIES
    # -----------------------------------------------------------------------

    def _update_progress(self, value: float, message: str):
        self._safe_ui(lambda: self.progress_var.set(value))
        self._safe_ui(lambda: self.progress_label.config(text=message))

    def _safe_ui(self, func):
        """Safely schedule a UI update from any thread."""
        try:
            self.parent.after_idle(func)
        except Exception as e:
            print(f"UI update error: {e}")
