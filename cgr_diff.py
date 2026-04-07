import os
import pickle
import sys
import threading
import tkinter
import tkinter.messagebox
from functools import partial
import random

import customtkinter as ctk
from tkinter import filedialog, messagebox, simpledialog
import tkinter.filedialog as fd

import numpy as np
from Bio import Entrez
from PIL import Image
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator, FuncFormatter
from skimage.metrics import structural_similarity
from sklearn.manifold import MDS
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt, colors

try:
    import mplcursors  # type: ignore
except Exception:
    mplcursors = None

from chaos_game_representation import CGR
from distances.distance_metrics import get_dist
from sequence_generation.sequence_generation import generate_kmers, generate_dna_sequence

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
HEADER_FONT = ('Cambria', 15)
HEADER_FONT_BOLD = ('Cambria', 15, 'bold')
UNDER_REP = "Lavender"
OVER_REP = "Blue"

# main colors in the theme
COLORS = dict(
    BTN_COLOR=ctk.ThemeManager.theme["CTkButton"]["fg_color"],
    CTK_FRAME_COLORS=ctk.ThemeManager.theme["CTkFrame"]["fg_color"],
    DISABLED_BTN_COLOR="#888888",
    TEXT_NORMAL_COLOR=ctk.ThemeManager.theme["CTkButton"]["text_color"],
    TEXT_DISABLE_COLOR="#707370",  # BTN_THEME.get("text_color_disabled", TEXT_NORMAL_COLOR)
    FRAME_COLOR="#707370",
    FRAME_NORMAL_COLOR="#2B2B2B",
    FRAME_HOVER_COLOR="#444444",
    BORDER_COLOR="#333333",
    LIGHT_FRAME_COLOR="#DBDBDB",
    Green="#2E8B57",
    Blue="#3668A0",
    Lavender="#967BB6", )
KMERS = [str(i) for i in range(2, 10)]
KMERS_SYNTH = [str(i) for i in range(2, 7)]
DISTANCES = ["Normalized Euclidean", "Cosine", "Manhattan", "Descriptor", "DSSIM", "K-S", "Wasserstein"]
RESOLUTION_DICT = {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4}
PLOT_TYPES = ["Bar plot", "Line plot", "Histogram plot"]


class GUIDataStructure:
    def __init__(self):
        self.seq_name = ctk.StringVar()
        self.seq = ""
        self.start_seq = ctk.IntVar()
        self.start_txt = ctk.StringVar()
        self.end_seq = ctk.IntVar()
        self.end_txt = ctk.StringVar()


class FCGRNormalizer:
    def __init__(self, method="asinh", scale="median_nz",
                 clip_low=1.0, clip_high=99.0, c=1e4, eps=1e-12,
                 k_compensate=True, length_mode=None, ref_length=500_000):
        self.method = method
        self.scale = scale
        self.clip_low = float(clip_low)
        self.clip_high = float(clip_high)
        self.c = float(c)
        self.eps = float(eps)
        self.k_compensate = bool(k_compensate)

        # NEW:
        self.length_mode = length_mode
        self.ref_length = int(ref_length)

        self.fitted = False
        self.s = None
        self.low = None
        self.high = None

    def _choose_scale(self, f):
        if isinstance(self.scale, (int, float)):
            return float(self.scale)
        nz = f[f > 0]
        return float(np.median(nz)) if nz.size else 1.0

    def fit(self, freq_mats, ks):
        """
        freq_mats: list of 2D freq arrays in [0,1]
        ks:        optional list of k for each matrix (same length as freq_mats)
        """
        vals = []
        s_list = []
        for i, f in enumerate(freq_mats):
            f = np.asarray(f, dtype=float)
            if self.k_compensate:
                k_i = ks[i] if ks is not None else None
                f = f * (4.0 ** k_i)  # compensate for k
            if self.method == "asinh":
                s_list.append(self._choose_scale(f))
                vals.append(f.ravel())
            elif self.method == "log":
                vals.append(np.log1p(f * self.c).ravel())
            else:
                raise ValueError("Unsupported method.")

        if self.method == "asinh":
            self.s = float(np.median(s_list)) if s_list else 1.0
            pooled = np.arcsinh(np.concatenate(vals) / (self.s if self.s > 0 else 1.0))
        else:
            pooled = np.concatenate(vals) if vals else np.array([0.0])

        self.low = float(np.percentile(pooled, self.clip_low))
        self.high = float(np.percentile(pooled, self.clip_high))
        if not np.isfinite(self.low) or not np.isfinite(self.high) or self.high <= self.low:
            self.low, self.high = float(np.min(pooled)), float(np.max(pooled) + 1e-9)
        self.fitted = True

    def transform01(self, f, k=None, L=None):
        """
        Map one frequency matrix f to [0,1].
        k:  k-mer (for 4^k compensation if enabled)
        L:  sequence length (for brightness fading)
        """
        if not self.fitted:
            raise RuntimeError("FCGRNormalizer not fitted. Call .fit([...]) first.")
        f = np.asarray(f, dtype=float)

        # --- compensate for k so different k’s have similar scale ---
        if self.k_compensate and (k is not None):
            f = f * (4.0 ** k)

        # --- fade brightness based on sequence length ---
        if self.length_mode and (L is not None) and (k is not None):
            N = max(1, L - k + 1)
            if self.length_mode == "linear":
                length_scale = min(N / max(1, self.ref_length - k + 1), 1.0)
            elif self.length_mode == "sqrt":
                length_scale = min((N / max(1, self.ref_length - k + 1)) ** 0.5, 1.0)
            else:
                length_scale = 1.0
            f = f * length_scale

        # --- apply nonlinearity + global window ---
        if self.method == "asinh":
            x = np.arcsinh(f / (self.s if self.s > 0 else 1.0))
        else:
            x = np.log1p(f * self.c)

        x = np.clip(x, self.low, self.high)
        V = (x - self.low) / (self.high - self.low + self.eps)
        return V

    @staticmethod
    def fcgr_to_freq(mat):
        total = float(np.sum(mat))
        return mat / total if total > 0 else mat

    @staticmethod
    def to_uint8_from_01(V, white_is_high=True):
        if white_is_high:
            V = 1.0 - V
        V = np.clip(V, 0.0, 1.0)
        return np.round(V * 255.0).astype(np.uint8)


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        # TODO: need to clear this at some point
        self.temp_output_path = self._resource_path(".gui_temp_outputs")
        if not os.path.exists(self.temp_output_path):
            os.makedirs(self.temp_output_path)

        self.title("CGR-Diff.py")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{int(screen_width)}x{int(screen_height)}")

        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)
        self.grid_rowconfigure(1, weight=1)

        self.appearance = ctk.StringVar(value="Dark")
        # tab names (used both by navbar and tabview)
        self.nav_buttons = {}
        self.theme_button = None
        self.tab_names = ["CGR Analysis", "CGR Comparator", "Common Reference", "Multispecies Comparator"]
        self.active_tab = self.tab_names[0]

        # ------------------------- Application state variables -------------------------
        self.uploaded_seq_lists_frame = None  # frame that holds the list of uploaded files
        self.uploaded_files = []  # list of uploaded fasta files (full paths)
        self.file_names = []  # list of uploaded fasta file names (without paths)
        self.file_cards = []  # list of card widgets corresponding to uploaded files
        self.selected_file_index = None  # index of currently selected file in uploaded_files (or None)

        self.k_var = ctk.IntVar(value=9)  # k-mer selection variable
        self.dist_metric = tkinter.StringVar(value="DSSIM")  # distance metric selection variable
        self.t1_3d_filter_var = tkinter.StringVar(value="All")  # 3D bar plot filter: All / Over / Under

        # Variables for page 1 (CGR Analysis)
        self._t1_progress_status = "Press 'Run Analysis' to start."
        self.t1_status_label = None
        self._t1_progress = 0.0
        self.t1_progress_bar = None
        self.t1_synth_dialog = None
        self.t1_fcgrs_dict = None
        self._t1_selected_path = None
        self.t1_start_entry = None
        self.t1_end_entry = None
        self.t1_len_label = None
        self.t1_ds = GUIDataStructure()
        self._t1_last_seq = None
        self.t1_hist_frame = None
        self.t1_placeholder_label = None
        self.t1_hist_fig = None
        self.t1_hist_canvas = None
        self.t1_hist_save_btn = None
        self._t1_hist_cids = []
        self.t1_fcgr_frame = None
        self.t1_fcgr_fig = None
        self.t1_fcgr_canvas = None
        self.t1_fcgr_save_btn = None
        self.t1_3d_fcgr_frame = None
        self.t1_3d_fcgr_fig = None
        self.t1_3d_fcgr_canvas = None

        # Variables for page 2 (CGR Comparator)
        self.t2_ds = {'1': GUIDataStructure(), '2': GUIDataStructure()}
        self.t2_segment_size_toggle = tkinter.IntVar(value=0)  # 0: Variable, 1: Fix
        self.t2_segment_size = tkinter.StringVar(value="")  # Segment size entry variable
        self.t2_segment_entry = None  # Entry widget for segment size
        self.t2_rc = {}  # reverse complement checkbox dictionary
        self.t2_shuffle = {}  # random sequence checkbox dictionary

        self.t2_start_seq_scale = {}
        self.t2_end_seq_scale = {}
        self.t2_start_seq_entry = {}
        self.t2_end_seq_entry = {}

        self.t2_display_frame = None
        self.t2_placeholder_label = None
        self.t2_fig = None
        self.t2_canvas = None
        self.t2_save_btn = None

        # Variables for page 3 (Common Reference)
        self.t3_ds = {'1': GUIDataStructure(), '2': GUIDataStructure()}
        self.t3_segment_size = tkinter.StringVar(value="")  # 500,000 for test
        self.t3_use_rep_algo = tkinter.IntVar(value=1)  # 0: use start and end, 1: use algo
        self.t3_rep_algo_type = tkinter.StringVar(value="RepSeg")  # Representation algorithm type
        self.t3_rep_number = tkinter.StringVar(value="1")  # Number of representations to generate
        self.t3_plot_type = tkinter.StringVar(value="Bar plot")
        self.t3_seq_len_label = None
        self.t3_rep_len_label = None
        self.t3_rep_type_combobox = None
        self.t3_rep_n_entry = None
        self.t3_start_entry = None
        self.t3_end_entry = None
        self.t3_start_label = None
        self.t3_end_label = None

        self.t3_cgr_distance_history = []
        self._t3_progress_status = "Press 'Run' to start."
        self.t3_status_label = None
        self._t3_progress = 0.0
        self.t3_progress_bar = None

        self.t3_3d_display_frame = None
        self.t3_3d_placeholder_label = None
        self.t3_3d_fig = None
        self.t3_3d_canvas = None
        self._t3_mds_drawn = False
        self.t3_seg_info = None
        self.t3_ref_info = None

        self.t3_fcgr_display_frame = None
        self.t3_fcgr_placeholder_label = None
        self.t3_fcgr_fig = None
        self.t3_fcgr_canvas = None
        self.t3_fcgr_save_btn = None

        self.t3_plot_display_frame = None
        self.t3_plot_placeholder_label = None
        self.t3_plot_fig = None
        self.t3_plot_canvas = None
        self.t3_plot_save_btn = None
        self._t3_plot_cids = []  # mpl_connect ids to disconnect when redrawing

        self.t3_scale = None
        self.t3_pic_num = ctk.IntVar(value=0)

        # ------------------------- Build UI -------------------------
        self.fcgr_normalizer = FCGRNormalizer(method='asinh', scale="median_nz",
                                              clip_low=float(1.0), clip_high=float(99.0), c=float(1e4))

        self._create_top_navbar()
        self._build_main_content()

        self.t1_upload_files(hard_coded=True)

    @staticmethod
    def _resolve_ctk_color(c):
        if isinstance(c, (tuple, list)) and len(c) >= 2:
            return c[0] if ctk.get_appearance_mode() == "Light" else c[1]
        return c

    def _update_tk_canvas_theme(self):
        if not hasattr(self, "seq_list_canvas") or self.seq_list_canvas is None:
            return
        frame_bg = self._resolve_ctk_color(COLORS["CTK_FRAME_COLORS"])
        self.seq_list_canvas.configure(bg=frame_bg, highlightbackground=frame_bg)
        self.uploaded_seq_lists_frame.configure(fg_color=frame_bg, bg_color=frame_bg)

    def _restyle_uploaded_cards(self):
        if not self.file_cards:
            return

        btn_bg = self._resolve_ctk_color(ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        btn_text = self._resolve_ctk_color(ctk.ThemeManager.theme["CTkButton"]["text_color"])

        # normal label colors for unselected cards
        normal_text = "black" if ctk.get_appearance_mode() == "Light" else "white"
        disabled_text = "#707370"

        for i, card in enumerate(self.file_cards):
            is_selected = (self.selected_file_index == i)

            if is_selected:
                card.configure(fg_color=btn_bg, corner_radius=0)
            else:
                card.configure(fg_color="transparent", corner_radius=0)

            for child in card.winfo_children():
                row = child.grid_info().get("row", None)
                if is_selected:
                    child.configure(fg_color=btn_bg, text_color=btn_text)
                else:
                    child.configure(fg_color="transparent", text_color=normal_text if row == 0 else disabled_text)

    def _toggle_theme(self):
        new_mode = "Light" if self.appearance.get() == "Dark" else "Dark"
        self.appearance.set(new_mode)
        ctk.set_appearance_mode(new_mode)
        # update icon
        self.theme_button.configure(text="☀️" if new_mode == "Dark" else "🌙")
        self._update_tk_canvas_theme()
        self._restyle_uploaded_cards()

    def _create_top_navbar(self):
        nav = ctk.CTkFrame(self, corner_radius=100, border_color=COLORS["BORDER_COLOR"], border_width=1)
        nav.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        nav.grid_columnconfigure(0, weight=1)
        nav.grid_columnconfigure(1, weight=0)

        # ---- left side: tab buttons ----
        tabs_frame = ctk.CTkFrame(nav, fg_color="transparent")
        tabs_frame.grid(row=0, column=0, sticky="w", padx=(10, 0), pady=(5, 5))

        for i, name in enumerate(self.tab_names):
            btn = ctk.CTkButton(tabs_frame, text=name, command=lambda n=name: self._switch_tab(n), corner_radius=100,
                                height=32, border_width=0, font=HEADER_FONT,
                                fg_color=COLORS["BTN_COLOR"] if i == self.tab_names.index(
                                    self.active_tab) else "transparent",
                                text_color=COLORS["TEXT_NORMAL_COLOR"] if i == self.tab_names.index(
                                    self.active_tab) else COLORS["TEXT_DISABLE_COLOR"], )
            btn.grid(row=0, column=i, padx=(0, 4))
            self.nav_buttons[name] = btn

        # ---- right side: theme toggle button ----
        self.theme_button = ctk.CTkButton(nav, width=32, height=32, text="☀️", corner_radius=100,
                                          fg_color=COLORS["FRAME_HOVER_COLOR"], hover_color=COLORS["FRAME_COLOR"],
                                          command=self._toggle_theme)
        self.theme_button.grid(row=0, column=1, padx=(0, 10), pady=7, sticky="e")

    def _switch_tab(self, name: str):
        if name == self.active_tab:
            return
        self.active_tab = name
        # Update button styles
        for tab_name, btn in getattr(self, "nav_buttons", {}).items():
            if tab_name == name:
                btn.configure(fg_color=COLORS["BTN_COLOR"], text_color=COLORS["TEXT_NORMAL_COLOR"])
            else:
                btn.configure(fg_color="transparent", text_color=COLORS["TEXT_DISABLE_COLOR"])
        # rebuild main content
        self._build_main_content()

    def _build_main_content(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=1, column=0, sticky="nsew")

        if self.active_tab == "CGR Analysis":
            self._build_cgr_analysis(main)
        elif self.active_tab == "CGR Comparator":
            self._build_cgr_comparator(main)
        elif self.active_tab == "Common Reference":
            self._build_common_reference(main)
        # elif self.active_tab == "Multispecies Comparator":
        #     self._build_multispecies_comparator(main)
        else:
            main.grid_columnconfigure(0, weight=1)
            main.grid_rowconfigure(0, weight=1)

            placeholder = ctk.CTkFrame(main)
            placeholder.grid(row=0, column=0, sticky="nsew")
            placeholder.grid_columnconfigure(0, weight=1)
            placeholder.grid_rowconfigure(0, weight=1)

            label = ctk.CTkLabel(placeholder, text=f"{self.active_tab} (empty)")
            label.grid(row=0, column=0)

    def _build_cgr_analysis(self, parent):
        parent.grid_columnconfigure(0, weight=0, minsize=320)  # left panel
        parent.grid_columnconfigure(1, weight=1)  # right panel
        parent.grid_rowconfigure(0, weight=0, minsize=1)
        parent.grid_rowconfigure(1, weight=1)

        # ---------- Left panel ----------
        config_frame = ctk.CTkFrame(parent, corner_radius=8, border_color=COLORS["BORDER_COLOR"], border_width=1)
        config_frame.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_rowconfigure(1, weight=5)  # row 1 is list_frame
        config_frame.grid_propagate(False)

        # ---------- Designing the config frame (F1) ----------
        # top buttons
        top_btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        top_btn_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        top_btn_frame.grid_columnconfigure((0, 1), weight=1)

        search_btn = ctk.CTkButton(top_btn_frame, text="Search and Download", corner_radius=8, height=35,
                                   font=HEADER_FONT, text_color="white", )
        search_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        upload_btn = ctk.CTkButton(top_btn_frame, text="Upload", corner_radius=8, height=35, font=HEADER_FONT,
                                   text_color="white", command=self.t1_upload_files)
        upload_btn.grid(row=1, column=0, sticky="ew")

        generate_btn = ctk.CTkButton(top_btn_frame, text="Generate", corner_radius=8, height=35, font=HEADER_FONT,
                                     text_color="white", command=self.t1_gen_synth_seq_event)
        generate_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        # list of genomes
        list_frame = ctk.CTkFrame(config_frame, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                  fg_color="transparent")
        list_frame.grid(row=1, column=0, padx=(10, 10), pady=10, sticky="nsew")
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)

        label = ctk.CTkLabel(list_frame, text="List of available sequences:", font=HEADER_FONT, anchor="w", )
        label.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 5))

        # ---------- SCROLLABLE REGION (both directions) ----------
        scroll_container = ctk.CTkFrame(list_frame, corner_radius=0, fg_color="transparent", border_width=0, )
        scroll_container.grid(row=1, column=0, sticky="nsew", padx=(5, 5), pady=(1, 5))
        scroll_container.grid_columnconfigure(0, weight=1)
        scroll_container.grid_rowconfigure(0, weight=1)

        # plain Tk Canvas for scrolling
        # TODO: not scrollable with mouse wheel
        self.seq_list_canvas = tkinter.Canvas(scroll_container, highlightthickness=0)
        self.seq_list_canvas.grid(row=0, column=0, sticky="nsew")

        # scrollbars use this canvas
        v_scroll = ctk.CTkScrollbar(scroll_container, orientation="vertical", command=self.seq_list_canvas.yview)
        v_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 0))

        h_scroll = ctk.CTkScrollbar(scroll_container, orientation="horizontal", command=self.seq_list_canvas.xview)
        h_scroll.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.seq_list_canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.uploaded_seq_lists_frame = ctk.CTkFrame(self.seq_list_canvas)
        self.seq_list_canvas.create_window((0, 0), window=self.uploaded_seq_lists_frame, anchor="nw")

        # apply correct background immediately
        self._update_tk_canvas_theme()

        def _on_inner_configure(event):
            # update scroll region to fit inner frame (both width and height)
            self.seq_list_canvas.configure(scrollregion=self.seq_list_canvas.bbox("all"))

        self.uploaded_seq_lists_frame.bind("<Configure>", _on_inner_configure)
        self.t1_refresh_uploaded_file_list()

        # k-mer, start-end frame
        kmer_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        kmer_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="ew")
        kmer_frame.grid_columnconfigure((0, 1), weight=1)

        # k-mer for visualization
        (ctk.CTkLabel(kmer_frame, text="k-mer: ", font=HEADER_FONT_BOLD)
         .grid(row=1, column=0, sticky="w", padx=(10, 0), pady=(0, 5)))
        (ctk.CTkComboBox(kmer_frame, values=KMERS, state="readonly", variable=self.k_var)
         .grid(row=1, column=1, sticky="ew", padx=(5, 10), pady=(0, 5)))

        # Set start and end of the sequence and total length of it
        t1_start_label = ctk.CTkLabel(kmer_frame, text="Start: ", font=HEADER_FONT)
        t1_start_label.grid(row=2, column=0, sticky="w", padx=(10, 0), pady=(10, 0))
        if self.t1_start_entry is not None and self.t1_start_entry.winfo_exists():
            state = self.t1_start_entry.cget("state")
        else:
            state = "disabled"
        self.t1_start_entry = ctk.CTkEntry(kmer_frame, textvariable=self.t1_ds.start_txt)
        self.t1_start_entry.bind('<FocusOut>', partial(self._entry_change, self.t1_ds, "start"))
        self.t1_start_entry.bind('<Key-Return>', partial(self._entry_change, self.t1_ds, "start"))
        self.t1_start_entry.configure(state=state)
        self.t1_start_entry.grid(row=3, column=0, sticky="ew", padx=(10, 0), pady=(0, 0))

        t1_end_label = ctk.CTkLabel(kmer_frame, text="End: ", font=HEADER_FONT)
        t1_end_label.grid(row=2, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
        self.t1_end_entry = ctk.CTkEntry(kmer_frame, textvariable=self.t1_ds.end_txt)
        self.t1_end_entry.bind('<FocusOut>', partial(self._entry_change, self.t1_ds, "end"))
        self.t1_end_entry.bind('<Key-Return>', partial(self._entry_change, self.t1_ds, "end"))
        self.t1_end_entry.configure(state=state)
        self.t1_end_entry.grid(row=3, column=1, sticky="ew", padx=(5, 10), pady=(0, 0))

        if self.t1_len_label is not None and self.t1_len_label.winfo_exists():
            text = self.t1_len_label.cget("text")
        else:
            text = "Length=0"
        self.t1_len_label = ctk.CTkLabel(kmer_frame, text=text, font=('Cambria', 10),
                                         text_color=COLORS["TEXT_DISABLE_COLOR"], anchor="w")
        self.t1_len_label.grid(row=4, column=0, columnspan=2, sticky="ew", padx=(15, 10), pady=(0, 10))
        self.t1_len_label.grid_propagate(False)

        # bottom buttons
        bottom_btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        bottom_btn_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        bottom_btn_frame.grid_columnconfigure((0, 1), weight=1)

        remove_btn = ctk.CTkButton(bottom_btn_frame, text="Remove", corner_radius=8, height=35, font=HEADER_FONT,
                                   command=self.t1_remove_selected_file, )
        remove_btn.grid(row=1, column=0, sticky="ew")

        run_btn = ctk.CTkButton(bottom_btn_frame, text="Run Analysis", corner_radius=8, height=35, font=HEADER_FONT,
                                command=partial(self.t1_run_manager, None), )
        run_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        # ---------- Right panel ----------
        # Progress frame
        progress_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                      fg_color="transparent", height=40)
        progress_frame.grid(row=0, column=1, padx=(0, 5), pady=(5, 0), sticky="nsew")
        progress_frame.grid_columnconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(1, weight=5)
        progress_frame.grid_propagate(False)
        self.t1_progress_bar = ctk.CTkProgressBar(master=progress_frame, orientation="horizontal", )
        if getattr(self, "_t1_progress", None) is not None:
            self.t1_progress_bar.set(self._t1_progress)
        else:
            self.t1_progress_bar.set(0)
        self.t1_progress_bar.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")

        # Status label under the bar
        self._t1_progress_status = getattr(self, "_t1_progress_status", "Idle")
        self.t1_status_label = ctk.CTkLabel(master=progress_frame, text=self._t1_progress_status, anchor="w",
                                            font=ctk.CTkFont(size=11))
        self.t1_status_label.grid(row=1, column=0, padx=5, pady=(0, 6), sticky="ew")

        display_frame = ctk.CTkFrame(parent, border_width=1, corner_radius=8, border_color=COLORS["BORDER_COLOR"])
        display_frame.grid(row=1, column=1, padx=(0, 5), pady=(5, 5), sticky="nsew")
        display_frame.grid_columnconfigure(0, weight=1)
        # grid configuration for display content:
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=2)
        display_frame.grid_columnconfigure(0, weight=2)  # left
        display_frame.grid_columnconfigure(1, weight=3)  # right (larger)
        display_frame.grid_columnconfigure(2, weight=1)

        # top histogram frame (full width)
        self.t1_hist_frame = ctk.CTkFrame(display_frame, fg_color="transparent")
        self.t1_hist_frame.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=(5, 5), pady=(5, 0), )
        self.t1_hist_frame.grid_rowconfigure(0, weight=1)
        self.t1_hist_frame.grid_columnconfigure(0, weight=1)
        self.t1_hist_frame.grid_propagate(False)

        self.t1_placeholder_label = ctk.CTkLabel(master=display_frame, text="Display Area",
                                                 font=HEADER_FONT, text_color=COLORS["TEXT_DISABLE_COLOR"])
        self.t1_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")
        if getattr(self, "t1_fcgrs_dict", None) is not None:
            # Histogram
            self.t1_hist_frame.configure(corner_radius=8, border_width=1, fg_color=COLORS["LIGHT_FRAME_COLOR"],
                                         border_color=COLORS["BORDER_COLOR"])

            self._draw_panel(frame=self.t1_hist_frame, fig_attr="t1_hist_fig", canvas_attr="t1_hist_canvas",
                             save_btn_attr="t1_hist_save_btn", save_command=lambda: self._save_figure("t1_hist_fig"),
                             placeholder_attr="t1_placeholder_label", fcgrs_dict=self.t1_fcgrs_dict,
                             panel_type="kmer_hist", )

        # bottom-left frame (smaller)
        self.t1_fcgr_frame = ctk.CTkFrame(display_frame, fg_color="transparent")
        self.t1_fcgr_frame.grid(row=1, column=0, sticky="nsew", padx=(5, 0), pady=(5, 5), )
        self.t1_fcgr_frame.grid_rowconfigure(0, weight=1)
        self.t1_fcgr_frame.grid_columnconfigure(0, weight=1)
        self.t1_fcgr_frame.grid_propagate(False)
        if getattr(self, "t1_fcgr_fig", None) is not None:
            self.t1_fcgr_frame.configure(corner_radius=8, border_width=1, fg_color=COLORS["FRAME_COLOR"],
                                         border_color=COLORS["BORDER_COLOR"])
            # Create a new canvas for the existing figure, attached to the new frame
            self.t1_fcgr_canvas = FigureCanvasTkAgg(self.t1_fcgr_fig, master=self.t1_fcgr_frame)
            widget = self.t1_fcgr_canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t1_fcgr_canvas.draw()

            if getattr(self, "t1_fcgr_save_btn", None) is not None and self.t1_fcgr_save_btn.winfo_exists():
                try:
                    self.t1_fcgr_save_btn.destroy()
                except Exception:
                    pass

            self.t1_fcgr_save_btn = ctk.CTkButton(master=self.t1_fcgr_frame, text="💾", width=30, height=30,
                                                  fg_color=COLORS["BORDER_COLOR"],
                                                  hover_color=COLORS["FRAME_HOVER_COLOR"],
                                                  command=partial(self._save_figure, "t1_fcgr_fig"))
            self.t1_fcgr_save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)

        # bottom-right frame (larger)
        self.t1_3d_fcgr_frame = ctk.CTkFrame(display_frame, fg_color="transparent")
        self.t1_3d_fcgr_frame.grid(row=1, column=1, sticky="nsew", padx=(5, 5), pady=(5, 5), )
        self.t1_3d_fcgr_frame.grid_rowconfigure(0, weight=1)
        self.t1_3d_fcgr_frame.grid_columnconfigure(0, weight=1)
        self.t1_3d_fcgr_frame.grid_propagate(False)
        if getattr(self, "t1_3d_fcgr_fig", None) is not None:
            self.t1_3d_fcgr_frame.configure(corner_radius=8, border_width=1, fg_color=COLORS["FRAME_COLOR"],
                                            border_color=COLORS["BORDER_COLOR"])
            self._draw_panel(frame=self.t1_3d_fcgr_frame, fig_attr="t1_3d_fcgr_fig", canvas_attr="t1_3d_fcgr_canvas",
                             save_btn_attr="t1_3d_fcgr_save_btn",
                             save_command=lambda: self._save_figure("t1_3d_fcgr_fig"), placeholder_attr=None,
                             fcgrs_dict=self.t1_fcgrs_dict, panel_type="fcgr_3d")

        self.t1_stat_frame = ctk.CTkFrame(display_frame, fg_color="transparent")
        self.t1_stat_frame.grid(row=1, column=2, sticky="nsew", padx=(5, 5), pady=(5, 5), )
        self.t1_stat_frame.grid_rowconfigure(0, weight=0)  # title — fixed height
        self.t1_stat_frame.grid_rowconfigure(1, weight=1)  # treeview — takes all space
        self.t1_stat_frame.grid_rowconfigure(2, weight=0)  # button — fixed height
        self.t1_stat_frame.grid_columnconfigure(0, weight=1)
        self.t1_stat_frame.grid_propagate(False)
        if getattr(self, "t1_fcgrs_dict", None) is not None:
            self._update_t1_stats_table_from_fcgr(top_n=100)

    def _build_cgr_comparator(self, parent):
        parent.grid_columnconfigure(0, weight=0, minsize=320)  # left panel
        parent.grid_columnconfigure(1, weight=1)  # right panel
        parent.grid_rowconfigure(0, weight=1)
        parent.grid_rowconfigure(1, weight=4)

        # ---------- Left panel ----------
        config_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        config_frame.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=1)
        for i in range(4):
            config_frame.grid_rowconfigure(i, weight=1)
        config_frame.grid_propagate(False)
        # ---------- Right panel ----------
        slider_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        slider_frame.grid(row=0, column=1, padx=(0, 5), pady=(5, 0), sticky="nsew")
        for i in range(4):
            slider_frame.grid_rowconfigure(i, weight=1)
        slider_frame.grid_columnconfigure(2, weight=1)
        slider_frame.grid_propagate(False)
        self.t2_display_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1,
                                             border_color=COLORS["BORDER_COLOR"], fg_color=COLORS["FRAME_COLOR"])
        self.t2_display_frame.grid(row=1, column=1, padx=(0, 5), pady=(5, 5), sticky="nsew")
        self.t2_display_frame.grid_columnconfigure(0, weight=1)
        self.t2_display_frame.grid_rowconfigure(0, weight=1)
        self.t2_display_frame.grid_propagate(False)

        self.t2_placeholder_label = ctk.CTkLabel(master=self.t2_display_frame, text="Display Area", font=HEADER_FONT,
                                                 text_color="black")
        self.t2_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")

        # ---------- Designing the config frame (F2) ----------
        # Choose sequences frame
        seq_frame = ctk.CTkFrame(config_frame, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        seq_frame.grid(row=0, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        seq_frame.grid_columnconfigure(0, weight=1)
        seq_frame.grid_columnconfigure(1, weight=1)
        seq_frame.grid_rowconfigure(0, weight=1)
        seq_frame.grid_rowconfigure(1, weight=1)

        # Sequence selection
        t2_seq_combobox = {}  # combobox dictionary for sequence selection
        for i in range(2):
            (ctk.CTkLabel(seq_frame, text=f"Sequence {i + 1}: ", font=HEADER_FONT_BOLD)
             .grid(row=i, column=0, sticky="w", padx=(10, 0), pady=(10, 10 * i)))

            t2_seq_combobox[f"{i + 1}"] = ctk.CTkComboBox(seq_frame, values=self.file_names, state="readonly",
                                                          variable=self.t2_ds[str(i + 1)].seq_name,
                                                          command=partial(self.t2_sequence_selection_event, f"{i + 1}"))
            t2_seq_combobox[f"{i + 1}"].grid(row=i, column=1, sticky="ew", padx=(0, 10), pady=(10, 10 * i))

        # Radio Button (Segment size)
        # Frame for segment size settings
        segment_size_frame = ctk.CTkFrame(config_frame, fg_color="transparent", border_color=COLORS["BORDER_COLOR"],
                                          border_width=1, corner_radius=8)
        segment_size_frame.grid(row=1, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        segment_size_frame.grid_columnconfigure(0, weight=1)
        segment_size_frame.grid_columnconfigure(1, weight=1)
        for i in range(3):
            segment_size_frame.grid_rowconfigure(i, weight=1)

        (ctk.CTkLabel(segment_size_frame, text="Segment Size", font=HEADER_FONT_BOLD)
         .grid(row=0, column=0, sticky="w", padx=5))
        ctk.CTkRadioButton(segment_size_frame, text="Variable", variable=self.t2_segment_size_toggle, value=0,
                           command=self.t2_segment_size_toggle_event).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkRadioButton(segment_size_frame, text="Fix", variable=self.t2_segment_size_toggle, value=1,
                           command=self.t2_segment_size_toggle_event).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.t2_segment_entry = ctk.CTkEntry(segment_size_frame, textvariable=self.t2_segment_size)
        self.t2_segment_entry.bind('<FocusOut>', partial(self.t2_sequence_value_change, "0"))
        self.t2_segment_entry.bind('<Key-Return>', partial(self.t2_sequence_value_change, "0"))
        if self.t2_segment_size_toggle.get() == 0:
            self.t2_segment_entry.configure(state="disabled")
        self.t2_segment_entry.grid(row=2, columnspan=2, padx=(5, 5), pady=(10, 10), sticky="ew")

        # Frame for k-mer selection and distance selection
        # k-mer selection
        kmer_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        kmer_frame.grid(row=2, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        kmer_frame.grid_columnconfigure(0, weight=1)
        kmer_frame.grid_columnconfigure(1, weight=1)
        kmer_frame.grid_rowconfigure(0, weight=1)
        kmer_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(kmer_frame, text="k-mer: ", font=HEADER_FONT_BOLD).grid(row=0, column=0, sticky="w", padx=(5, 0))
        (ctk.CTkComboBox(kmer_frame, values=KMERS, state="readonly", variable=self.k_var)
         .grid(row=0, column=1, sticky="ew", padx=(0, 5)))

        # distance measure selection
        (ctk.CTkLabel(kmer_frame, text="Distance measure: ", font=HEADER_FONT_BOLD)
         .grid(row=1, column=0, padx=(5, 0), pady=(10, 0), sticky="w"))
        (ctk.CTkComboBox(kmer_frame, values=DISTANCES, state="readonly", variable=self.dist_metric)
         .grid(row=1, column=1, sticky="ew", padx=(0, 5), pady=(10, 0)))

        # reverse complement or shuffle sequence
        rv_frame = ctk.CTkFrame(config_frame, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        rv_frame.grid(row=3, column=0, padx=(10, 10), pady=10, sticky="nsew")
        rv_frame.grid_columnconfigure(0, weight=1)
        rv_frame.grid_columnconfigure(1, weight=1)
        for i in range(4):
            rv_frame.grid_rowconfigure(i, weight=1)

        for i in range(2):
            (ctk.CTkLabel(rv_frame, text=f'Sequence {i + 1}:', font=HEADER_FONT_BOLD)
             .grid(row=(i * 2), column=0, padx=(10, 0), pady=(10, 0), sticky="w"))
            self.t2_rc[str(i + 1)] = ctk.CTkCheckBox(master=rv_frame, text="Reverse Complement")
            self.t2_rc[str(i + 1)].grid(row=(i * 2), column=1, padx=(10, 0), pady=(10, 0), sticky="w")
            self.t2_shuffle[str(i + 1)] = ctk.CTkCheckBox(master=rv_frame, text="Shuffle")
            self.t2_shuffle[str(i + 1)].grid(row=(i * 2) + 1, column=1, padx=(10, 0), pady=(10, 10), sticky="w")

        # plot button
        plot_button = ctk.CTkButton(config_frame, text="Plot", corner_radius=8, height=35, font=HEADER_FONT,
                                    command=self.t2_plot)
        plot_button.grid(row=4, column=0, pady=(10, 10))

        # ---------- Design the slider frame ----------
        for i in range(2):
            (ctk.CTkLabel(slider_frame, text=f'Sequence {i + 1}:', font=HEADER_FONT_BOLD)
             .grid(row=(i * 2), column=0, padx=(10, 0), pady=(10, 0)))
            (ctk.CTkLabel(slider_frame, text='Start').grid(row=(i * 2), column=1, padx=(5, 0), pady=(10, 0)))
            (ctk.CTkLabel(slider_frame, text='End').grid(row=(i * 2) + 1, column=1, padx=(5, 0), pady=(10, 0)))

            # The start slider and entry
            seq_length = len(self.t2_ds[str(i + 1)].seq)
            to_value = seq_length if seq_length > 0 else 1
            self.t2_start_seq_scale[str(i + 1)] = ctk.CTkSlider(slider_frame, from_=0, to=to_value,
                                                                orientation="horizontal",
                                                                variable=self.t2_ds[str(i + 1)].start_seq,
                                                                command=partial(self.t2_sequence_value_change,
                                                                                str(i + 1)))
            if self.t2_ds[str(i + 1)].seq == '':
                self.t2_start_seq_scale[str(i + 1)].set(0)
                self.t2_start_seq_scale[str(i + 1)].configure(state="disabled",
                                                              button_color=COLORS["DISABLED_BTN_COLOR"])
            self.t2_start_seq_scale[str(i + 1)].grid(row=(i * 2), column=2, padx=(5, 0), pady=(10, 0), sticky="ew")

            self.t2_start_seq_entry[str(i + 1)] = ctk.CTkEntry(slider_frame,
                                                               textvariable=self.t2_ds[str(i + 1)].start_txt)
            self.t2_start_seq_entry[str(i + 1)].bind('<FocusOut>', partial(self.t2_sequence_value_change, "3"))
            self.t2_start_seq_entry[str(i + 1)].bind('<Key-Return>', partial(self.t2_sequence_value_change, "3"))
            if self.t2_ds[str(i + 1)].seq == '':
                self.t2_start_seq_entry[str(i + 1)].configure(state="disabled")
            self.t2_start_seq_entry[str(i + 1)].grid(row=(i * 2), column=3, padx=(5, 0), pady=(10, 0))
            ctk.CTkLabel(slider_frame, text='bp').grid(row=(i * 2), column=4, padx=(5, 10), pady=(10, 0))

            # The end slider and entry
            self.t2_end_seq_scale[str(i + 1)] = ctk.CTkSlider(slider_frame, from_=0, to=to_value,
                                                              orientation="horizontal",
                                                              variable=self.t2_ds[str(i + 1)].end_seq,
                                                              command=partial(self.t2_sequence_value_change,
                                                                              str(i + 1)))
            if self.t2_ds[str(i + 1)].seq == '':
                self.t2_end_seq_scale[str(i + 1)].set(0)
                self.t2_end_seq_scale[str(i + 1)].configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
            if self.t2_segment_size_toggle.get() == 1:
                self.t2_end_seq_scale[str(i + 1)].configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
            self.t2_end_seq_scale[str(i + 1)].grid(row=(i * 2) + 1, column=2, padx=(5, 0), pady=(10, 0), sticky="ew")

            self.t2_end_seq_entry[str(i + 1)] = ctk.CTkEntry(slider_frame, textvariable=self.t2_ds[str(i + 1)].end_txt)
            self.t2_end_seq_entry[str(i + 1)].bind('<FocusOut>', partial(self.t2_sequence_value_change, "3"))
            self.t2_end_seq_entry[str(i + 1)].bind('<Key-Return>', partial(self.t2_sequence_value_change, "3"))
            if self.t2_ds[str(i + 1)].seq == '':
                self.t2_end_seq_entry[str(i + 1)].configure(state="disabled")
            self.t2_end_seq_entry[str(i + 1)].grid(row=(i * 2) + 1, column=3, padx=(5, 0), pady=(10, 0))
            ctk.CTkLabel(slider_frame, text='bp').grid(row=(i * 2) + 1, column=4, padx=(5, 10), pady=(10, 0))

        # ---------- Design the display frame ----------
        if getattr(self, "t2_fig", None) is not None:
            # Create a new canvas for the existing figure, attached to the new frame
            self.t2_canvas = FigureCanvasTkAgg(self.t2_fig, master=self.t2_display_frame)
            widget = self.t2_canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t2_canvas.draw()

            if getattr(self, "t2_save_btn", None) is not None and self.t2_save_btn.winfo_exists():
                try:
                    self.t2_save_btn.destroy()
                except Exception:
                    pass

            self.t2_save_btn = ctk.CTkButton(master=self.t2_display_frame, text="💾", width=30, height=30,
                                             fg_color=COLORS["BORDER_COLOR"],
                                             hover_color=COLORS["FRAME_HOVER_COLOR"],
                                             command=partial(self._save_figure, "t2_fig"), )
            self.t2_save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)

    def _build_common_reference(self, parent):
        parent.grid_columnconfigure(0, weight=0, minsize=320)  # left panel
        parent.grid_columnconfigure(1, weight=1)  # right panel
        parent.grid_rowconfigure(0, weight=0, minsize=1)
        parent.grid_rowconfigure(1, weight=1)

        # ---------- Left panel ----------
        config_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        config_frame.grid(row=0, column=0, rowspan=4, padx=(5, 5), pady=(5, 5), sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_rowconfigure(0, weight=2)
        config_frame.grid_rowconfigure(1, weight=2)
        config_frame.grid_rowconfigure(2, weight=1)
        config_frame.grid_propagate(False)

        # ---- Designing the config frame (F3) ----
        # Choose sequences frame
        seq_frame = ctk.CTkFrame(config_frame, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        seq_frame.grid(row=0, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        seq_frame.grid_columnconfigure(0, weight=1)
        seq_frame.grid_columnconfigure(1, weight=1)
        for i in range(6):
            seq_frame.grid_rowconfigure(i, weight=1)

        # Original sequence selection combobox and reference selection combobox
        # Sequence selection
        t3_seq_combobox = {}  # combobox dictionary for sequence and reference selection
        (ctk.CTkLabel(seq_frame, text=f"Sequence: ", font=HEADER_FONT_BOLD)
         .grid(row=0, column=0, sticky="w", padx=(10, 0), pady=(10, 0)))
        t3_seq_combobox["1"] = ctk.CTkComboBox(seq_frame, values=self.file_names, state="readonly",
                                               variable=self.t3_ds['1'].seq_name,
                                               command=partial(self.t3_sequence_selection_event, "1"))
        t3_seq_combobox["1"].grid(row=0, column=1, sticky="ew", padx=(0, 10), pady=(10, 0))

        (ctk.CTkLabel(seq_frame, text=f"Reference: ", font=HEADER_FONT_BOLD)
         .grid(row=1, column=0, sticky="w", padx=(10, 0), pady=(30, 0)))
        t3_seq_combobox["2"] = ctk.CTkComboBox(seq_frame, values=self.file_names, state="readonly",
                                               variable=self.t3_ds['2'].seq_name,
                                               command=partial(self.t3_sequence_selection_event, "2"))
        t3_seq_combobox["2"].grid(row=1, column=1, columnspan=2, sticky="ew", padx=(0, 10), pady=(30, 0))

        # Checkbox to use representative
        t3_use_rep_checkbox = ctk.CTkCheckBox(master=seq_frame, text="Use representative algorithm to\n"
                                                                     "choose the reference sequence",
                                              variable=self.t3_use_rep_algo,
                                              command=self.t3_use_rep_checkbox_event)
        t3_use_rep_checkbox.grid(row=2, column=0, columnspan=2, padx=(10, 0), pady=(10, 0), sticky="w")
        # Representative type
        if self.t3_rep_type_combobox is not None and self.t3_rep_type_combobox.winfo_exists():
            state = self.t3_rep_type_combobox.cget("state")
        else:
            state = "readonly"
        self.t3_rep_type_combobox = ctk.CTkComboBox(seq_frame, values=["RepSeg", "aRepSeg"],
                                                    variable=self.t3_rep_algo_type,
                                                    command=self.t3_rep_algo_change_event,
                                                    state=state)
        self.t3_rep_type_combobox.grid(row=3, column=0, sticky="ew", padx=(10, 0), pady=(10, 0))
        # Representative number (for aRepSeg)
        if self.t3_rep_n_entry is not None and self.t3_rep_n_entry.winfo_exists():
            text_color = self.t3_rep_n_entry.cget("text_color")
            state = self.t3_rep_n_entry.cget("state")
        else:
            text_color = COLORS["TEXT_DISABLE_COLOR"]
            state = "disable"
        self.t3_rep_n_entry = ctk.CTkEntry(seq_frame, textvariable=self.t3_rep_number, state=state,
                                           text_color=text_color, )
        self.t3_rep_n_entry.grid(row=3, column=1, sticky="ew", padx=(10, 10), pady=(10, 0))

        # Set start and end for the Reference sequence
        if self.t3_start_label is not None and self.t3_start_label.winfo_exists():
            text_color = self.t3_start_label.cget("text_color")
            state = self.t3_start_label.cget("state")
        else:
            text_color = COLORS["TEXT_DISABLE_COLOR"]
            state = "disable"
        # self.t3_ds['2'].start_txt = tkinter.StringVar(value="200,000")
        # self.t3_ds['2'].end_txt = tkinter.StringVar(value="700,000")
        self.t3_start_label = ctk.CTkLabel(seq_frame, text="Start: ", font=HEADER_FONT, text_color=text_color)
        self.t3_start_label.grid(row=4, column=0, sticky="w", padx=(10, 0), pady=(10, 0))
        self.t3_start_entry = ctk.CTkEntry(seq_frame, textvariable=self.t3_ds['2'].start_txt)
        self.t3_start_entry.bind('<FocusOut>', partial(self._entry_change, self.t3_ds['2'], "start"))
        self.t3_start_entry.bind('<Key-Return>', partial(self._entry_change, self.t3_ds['2'], "start"))
        self.t3_start_entry.configure(state=state, text_color=text_color)
        self.t3_start_entry.grid(row=5, column=0, sticky="ew", padx=(10, 0), pady=(0, 0))
        if self.t3_end_label is not None and self.t3_end_label.winfo_exists():
            text_color = self.t3_end_label.cget("text_color")
            state = self.t3_end_label.cget("state")
        else:
            text_color = COLORS["TEXT_DISABLE_COLOR"]
            state = "disable"
        self.t3_end_label = ctk.CTkLabel(seq_frame, text="End: ", font=HEADER_FONT, text_color=text_color)
        self.t3_end_label.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
        self.t3_end_entry = ctk.CTkEntry(seq_frame, textvariable=self.t3_ds['2'].end_txt)
        self.t3_end_entry.bind('<FocusOut>', partial(self._entry_change, self.t3_ds['2'], "end"))
        self.t3_end_entry.bind('<Key-Return>', partial(self._entry_change, self.t3_ds['2'], "end"))
        self.t3_end_entry.configure(state=state, text_color=text_color)
        self.t3_end_entry.grid(row=5, column=1, sticky="ew", padx=(10, 10), pady=(0, 0))

        if self.t3_rep_len_label is not None and self.t3_rep_len_label.winfo_exists():
            text = self.t3_rep_len_label.cget("text")
        else:
            text = "Reference length=0"
        self.t3_rep_len_label = ctk.CTkLabel(seq_frame, text=text, font=('Cambria', 10),
                                             text_color=COLORS["TEXT_DISABLE_COLOR"], anchor="w")
        self.t3_rep_len_label.grid(row=6, column=0, columnspan=2, sticky="ew", padx=(15, 10), pady=(0, 10))
        self.t3_rep_len_label.grid_propagate(False)

        # Frame for k-mer selection and distance selection
        # k-mer selection
        kmer_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        kmer_frame.grid(row=1, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        kmer_frame.grid_columnconfigure(0, weight=1)
        kmer_frame.grid_columnconfigure(1, weight=1)
        for i in range(4):
            kmer_frame.grid_rowconfigure(i, weight=1)

        # segment size selection
        segment_size_frame = ctk.CTkFrame(kmer_frame, fg_color="transparent")
        segment_size_frame.grid_columnconfigure(0, weight=1)
        segment_size_frame.grid_columnconfigure(1, weight=2)
        segment_size_frame.grid(row=0, column=0, columnspan=2, padx=(0, 0), pady=(0, 0), sticky="nsew")
        (ctk.CTkLabel(segment_size_frame, text="Segment size: ", font=HEADER_FONT_BOLD)
         .grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(10, 0)))
        (ctk.CTkEntry(segment_size_frame, textvariable=self.t3_segment_size)
         .grid(row=0, column=1, sticky="ew", padx=(0, 5), pady=(10, 0)))

        if self.t3_seq_len_label is not None and self.t3_seq_len_label.winfo_exists():
            text = self.t3_seq_len_label.cget("text")
        else:
            text = "Sequence length=0"
        self.t3_seq_len_label = ctk.CTkLabel(segment_size_frame, text=text, font=('Cambria', 10),
                                             text_color=COLORS["TEXT_DISABLE_COLOR"], anchor="w")
        self.t3_seq_len_label.grid(row=1, column=1, sticky="ew", padx=(5, 5), pady=(0, 0))
        self.t3_seq_len_label.grid_propagate(False)

        # k-mer selection
        (ctk.CTkLabel(kmer_frame, text="k-mer: ", font=HEADER_FONT_BOLD)
         .grid(row=1, column=0, sticky="w", padx=(5, 0), pady=(0, 10)))
        (ctk.CTkComboBox(kmer_frame, values=KMERS, state="readonly", variable=self.k_var)
         .grid(row=1, column=1, sticky="ew", padx=(0, 5), pady=(0, 10)))

        # distance measure selection
        (ctk.CTkLabel(kmer_frame, text="Distance measure: ", font=HEADER_FONT_BOLD)
         .grid(row=2, column=0, sticky="w", padx=(5, 0), pady=(0, 10)))
        (ctk.CTkComboBox(kmer_frame, values=DISTANCES, state="readonly", variable=self.dist_metric)
         .grid(row=2, column=1, sticky="ew", padx=(0, 5), pady=(0, 10)))

        # plot type selection
        (ctk.CTkLabel(kmer_frame, text="Plot type: ", font=HEADER_FONT_BOLD)
         .grid(row=3, column=0, sticky="w", padx=(5, 0), pady=(0, 10)))
        (ctk.CTkComboBox(kmer_frame, values=PLOT_TYPES, state="readonly", variable=self.t3_plot_type,
                         command=self.t3_plot_change_event)
         .grid(row=3, column=1, sticky="ew", padx=(0, 5), pady=(0, 10)))

        # Run button
        plot_button = ctk.CTkButton(config_frame, text="Run", corner_radius=8, height=35, font=HEADER_FONT,
                                    command=partial(self.t3_run_manager, None))
        plot_button.grid(row=2, column=0, pady=(10, 10))

        # ---------- Right panel ----------
        progress_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                      fg_color="transparent", height=40)
        progress_frame.grid(row=0, column=1, padx=(0, 5), pady=(5, 0), sticky="nsew")
        progress_frame.grid_columnconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(0, weight=1)
        progress_frame.grid_rowconfigure(1, weight=5)
        progress_frame.grid_propagate(False)

        display_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        display_frame.grid(row=1, column=1, padx=(0, 5), pady=(5, 5), sticky="nsew")
        display_frame.grid_columnconfigure(0, weight=2)
        display_frame.grid_columnconfigure(1, weight=3)
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_rowconfigure(1, weight=1)
        display_frame.grid_rowconfigure(2, weight=0, minsize=1)
        display_frame.grid_propagate(False)

        # ---- Designing each frames ----
        # Progress frame
        self.t3_progress_bar = ctk.CTkProgressBar(master=progress_frame, orientation="horizontal", )
        if getattr(self, "_t3_progress", None) is not None:
            self.t3_progress_bar.set(self._t3_progress)
        else:
            self.t3_progress_bar.set(0)
        self.t3_progress_bar.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        # Status label under the bar
        self._t3_progress_status = getattr(self, "_t3_progress_status", "Idle")
        self.t3_status_label = ctk.CTkLabel(master=progress_frame, text=self._t3_progress_status, anchor="w",
                                            font=ctk.CTkFont(size=11))
        self.t3_status_label.grid(row=1, column=0, padx=5, pady=(0, 6), sticky="ew")

        # Display frame
        # 3D frame
        self.t3_3d_display_frame = ctk.CTkFrame(display_frame, corner_radius=8, fg_color=COLORS["FRAME_COLOR"],
                                                border_width=1, border_color=COLORS["BORDER_COLOR"])
        self.t3_3d_display_frame.grid(row=0, column=0, padx=(5, 0), pady=(5, 0), sticky="nsew")
        self.t3_3d_display_frame.grid_columnconfigure(0, weight=1)
        self.t3_3d_display_frame.grid_rowconfigure(0, weight=1)
        self.t3_3d_display_frame.grid_rowconfigure(1, weight=1, minsize=1)
        self.t3_3d_display_frame.grid_propagate(False)

        self.t3_3d_placeholder_label = ctk.CTkLabel(master=self.t3_3d_display_frame, text="Display Area",
                                                    font=HEADER_FONT, text_color="black")
        self.t3_3d_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")
        if getattr(self, "t3_3d_fig", None) is not None:
            distance_path = f"{self.temp_output_path}/t3_run/t3_distance_matrix.pkl"
            if os.path.exists(distance_path):
                with open(distance_path, "rb") as handle:
                    D = pickle.load(handle)
            self._t3_mds_drawn = False  # force reconnect
            self._draw_panel(frame=self.t3_3d_display_frame, fig_attr="t3_3d_fig", canvas_attr="t3_3d_canvas",
                             save_btn_attr=None, save_command=lambda: self._save_figure("t3_3d_fig"),
                             placeholder_attr="t3_3d_placeholder_label", fcgrs_dict=None,
                             index=int(self.t3_pic_num.get() or 0), panel_type="mds", D=D, )

        # FCGR frame
        self.t3_fcgr_display_frame = ctk.CTkFrame(display_frame, corner_radius=8, fg_color=COLORS["FRAME_COLOR"],
                                                  border_width=1, border_color=COLORS["BORDER_COLOR"])
        self.t3_fcgr_display_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 0), sticky="nsew")
        self.t3_fcgr_display_frame.grid_columnconfigure(0, weight=1)
        self.t3_fcgr_display_frame.grid_rowconfigure(0, weight=1)
        self.t3_fcgr_display_frame.grid_propagate(False)

        self.t3_fcgr_placeholder_label = ctk.CTkLabel(master=self.t3_fcgr_display_frame, text="Display Area",
                                                      font=HEADER_FONT, text_color="black")
        self.t3_fcgr_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")
        if getattr(self, "t3_fcgr_fig", None) is not None:
            # Create a new canvas for the existing figure, attached to the new frame
            self.t3_fcgr_canvas = FigureCanvasTkAgg(self.t3_fcgr_fig, master=self.t3_fcgr_display_frame)
            widget = self.t3_fcgr_canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t3_fcgr_canvas.draw()

            if getattr(self, "t3_fcgr_save_btn", None) is not None and self.t3_fcgr_save_btn.winfo_exists():
                try:
                    self.t3_fcgr_save_btn.destroy()
                except Exception:
                    pass

            self.t3_fcgr_save_btn = ctk.CTkButton(master=self.t3_fcgr_display_frame, text="💾", width=30, height=30,
                                                  fg_color=COLORS["BORDER_COLOR"],
                                                  hover_color=COLORS["FRAME_HOVER_COLOR"],
                                                  command=partial(self._save_figure, "t3_fcgr_fig"))
            self.t3_fcgr_save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)

        # Plot frame
        self.t3_plot_display_frame = ctk.CTkFrame(display_frame, corner_radius=8, border_width=1,
                                                  border_color=COLORS["BORDER_COLOR"],
                                                  fg_color=COLORS["LIGHT_FRAME_COLOR"])
        self.t3_plot_display_frame.grid(row=1, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        self.t3_plot_display_frame.grid_columnconfigure(0, weight=1)
        self.t3_plot_display_frame.grid_rowconfigure(0, weight=1)
        self.t3_plot_display_frame.grid_propagate(False)

        self.t3_plot_placeholder_label = ctk.CTkLabel(master=self.t3_plot_display_frame, text="Plot Area",
                                                      font=HEADER_FONT, text_color="black")
        self.t3_plot_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")

        if getattr(self, "t3_plot_fig", None) is not None:
            self.t3_plot_canvas = FigureCanvasTkAgg(self.t3_plot_fig, master=self.t3_plot_display_frame)
            widget = self.t3_plot_canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t3_plot_canvas.draw()

            if getattr(self, "t3_plot_save_btn", None) is not None and self.t3_plot_save_btn.winfo_exists():
                try:
                    self.t3_plot_save_btn.destroy()
                except Exception:
                    pass

            self.t3_plot_save_btn = ctk.CTkButton(master=self.t3_plot_display_frame, text="💾", width=30, height=30,
                                                  fg_color=COLORS["BORDER_COLOR"],
                                                  hover_color=COLORS["FRAME_HOVER_COLOR"],
                                                  command=partial(self._save_figure, "t3_plot_fig"))
            self.t3_plot_save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)

        # Changing the picture with slider frame
        changing_frame = ctk.CTkFrame(display_frame, fg_color="transparent", height=20)
        changing_frame.grid(row=2, column=1, sticky="nsew", padx=(5, 5), pady=(0, 5))
        changing_frame.grid_columnconfigure(0, weight=0, minsize=1)
        changing_frame.grid_columnconfigure(1, weight=10)
        changing_frame.grid_columnconfigure(2, weight=0, minsize=1)

        self.t3_scale = ctk.CTkSlider(changing_frame, from_=0, orientation=ctk.HORIZONTAL, variable=self.t3_pic_num,
                                      command=partial(self.t3_change_images, self.t3_pic_num.get()))
        if self.t3_cgr_distance_history:
            self.t3_scale.configure(to=int(len(self.t3_cgr_distance_history) - 1))
            # The scale should be at the last position
            self.t3_scale.set(self.t3_pic_num.get())
        else:
            self.t3_scale.configure(to=0)
        self.t3_scale.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        (ctk.CTkButton(changing_frame, text="⬅", width=20, command=partial(self.t3_move_previous, None))
         .grid(row=0, column=0, padx=(0, 0)))
        (ctk.CTkButton(changing_frame, text="⮕", width=20, command=partial(self.t3_move_next, None))
         .grid(row=0, column=2, padx=(0, 0)))

        # Statistics frame
        stats_frame = ctk.CTkFrame(display_frame, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                   fg_color=COLORS["FRAME_COLOR"])
        stats_frame.grid(row=1, column=0, rowspan=2, padx=(5, 0), pady=(5, 5), sticky="nsew")
        stats_label = ctk.CTkLabel(stats_frame, text="Statistical Analysis", font=HEADER_FONT,
                                   text_color="black")
        stats_label.place(relx=0.5, rely=0.01, anchor="n")

    def _build_multispecies_comparator(self, parent):
        pass

    # --------------------------------------------------
    # Helper functions for CGR analysis tab
    # --------------------------------------------------
    def t1_upload_files(self, hard_coded=False):
        if hard_coded:
            file_paths = [
                "Data/Human/chromosomes/Human-chr21.fna",
                "Data/Escherichia coli/chromosomes/E coli-genome.fna"
            ]
        else:
            file_paths = filedialog.askopenfilenames(
                title="Select FASTA files",
                filetypes=[("FASTA files", "*.fa *.fasta *.fna *.ffn *.faa *.frn"), ("All files", "*.*"), ], )
            if not file_paths:
                return  # user cancelled

        # You can avoid duplicates
        for p in file_paths:
            if p not in self.uploaded_files:
                self.uploaded_files.append(p)

        self.t1_refresh_uploaded_file_list()

    def t1_refresh_uploaded_file_list(self):
        for widget in self.uploaded_seq_lists_frame.winfo_children():
            widget.destroy()

        self.file_cards = []  # reset cards list
        self.file_names = []  # reset file names list

        if not self.uploaded_files:
            no_file_label = ctk.CTkLabel(self.uploaded_seq_lists_frame, text="No files uploaded yet.", font=HEADER_FONT,
                                         anchor="w", text_color=COLORS["TEXT_DISABLE_COLOR"])
            no_file_label.grid(row=0, column=0, padx=5, pady=5)
            self.selected_file_index = None
            return

        for i, path in enumerate(self.uploaded_files):
            fname = os.path.basename(path)
            fname = fname.split(".")[0]
            self.file_names.append(fname)

            # card for each file
            card = ctk.CTkFrame(self.uploaded_seq_lists_frame, fg_color="transparent")
            card.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            # make everything inside the card clickable
            def _make_on_click(index):
                def _on_click(event=None):
                    return self.t1_set_selected_uploaded(index, reset_range=True)

                return _on_click

            on_click = _make_on_click(i)
            card.bind("<Button-1>", on_click)

            name_label = ctk.CTkLabel(card, text=fname, anchor="w", text_color=COLORS["TEXT_NORMAL_COLOR"])
            name_label.grid(row=0, column=0, padx=(1, 1), pady=(2, 0), sticky="w")
            name_label.bind("<Button-1>", on_click)

            path_label = ctk.CTkLabel(card, text=path, anchor="w", text_color=COLORS["TEXT_DISABLE_COLOR"])
            path_label.grid(row=1, column=0, padx=(1, 1), pady=(0, 2), sticky="w")
            path_label.bind("<Button-1>", on_click)

            self.file_cards.append(card)

        # if we had a selection, re-apply highlight (in case the list was redrawn)
        if self.selected_file_index is not None:
            if 0 <= self.selected_file_index < len(self.file_cards):
                self.t1_set_selected_uploaded(self.selected_file_index)
            else:
                self.selected_file_index = None

    def t1_set_selected_uploaded(self, index, reset_range=False):
        self.selected_file_index = index

        for i, card in enumerate(self.file_cards):
            if i == index:
                # SELECTED STYLE
                card.configure(fg_color=COLORS["BTN_COLOR"], corner_radius=0)
                for child in card.winfo_children():
                    child.configure(fg_color=COLORS["BTN_COLOR"], text_color=COLORS["TEXT_NORMAL_COLOR"])

                selected_path = self.uploaded_files[index]
                self._t1_last_seq = self._read_fasta(selected_path)[1]
                seq_len = len(self._t1_last_seq)
                self.t1_ds.seq = self._t1_last_seq

                self.t1_start_entry.configure(state="normal")
                self.t1_end_entry.configure(state="normal")

                if reset_range:
                    start, end = 0, seq_len
                else:
                    start = self._parse_int(self.t1_ds.start_txt.get())
                    end = self._parse_int(self.t1_ds.end_txt.get())

                    if start is None or end is None or start < 0 or end > seq_len or start >= end:
                        start, end = 0, seq_len

                self.t1_ds.start_seq.set(start)
                self.t1_ds.end_seq.set(end)
                self.t1_ds.start_txt.set(self._format_int(start))
                self.t1_ds.end_txt.set(self._format_int(end))

                self.t1_len_label.configure(text=f"Length={seq_len:,}")

                # Set the sequence length
                self.t1_len_label.configure(text=f"Length={seq_len:,}")
            else:
                card.configure(fg_color="transparent")
                for child in card.winfo_children():
                    if child.grid_info().get("row") == 0:
                        child.configure(fg_color="transparent", text_color=COLORS["TEXT_NORMAL_COLOR"])
                    else:
                        child.configure(fg_color="transparent", text_color=COLORS["TEXT_DISABLE_COLOR"])

    def t1_remove_selected_file(self):
        if self.selected_file_index is None:
            messagebox.showinfo("No selection", "Please select a file to remove.")
            return

        removed_path = self.uploaded_files.pop(self.selected_file_index)  # remove from list
        self.selected_file_index = None  # reset selection
        self.t1_refresh_uploaded_file_list()  # refresh GUI

        # Empty the start and end and sequence length and make start and end disable
        self.t1_start_entry.configure(state="disabled")
        self.t1_end_entry.configure(state="disabled")
        self.t1_ds.start_txt.set("")
        self.t1_ds.end_txt.set("")
        self.t1_len_label.configure(text="Length=0")
        self._t1_last_seq = None

    def t1_gen_synth_seq_event(self):
        def _accept_sequence(seq, name):
            if len(seq) <= 0:
                messagebox.showerror("Error", "No sequence generated.")
                return
            else:
                path = f"{self.temp_output_path}/Synthetic"
                if not os.path.exists(path):
                    os.makedirs(path)
                fasta_path = f"{path}/{name}.fasta"
                with open(fasta_path, "w") as f:
                    f.write(f">{name}\n")
                    # Write the sequence in there in one line
                    f.write(seq + "\n")

                # Add the generated sequence to the list of uploaded files
                self.uploaded_files.append(fasta_path)
                self.t1_refresh_uploaded_file_list()
                # Set the generated sequence as selected
                self.t1_set_selected_uploaded(len(self.uploaded_files) - 1, reset_range=True)

        # create once, reuse forever
        if (not hasattr(self, "t1_synth_dialog") or self.t1_synth_dialog is None or
                not self.t1_synth_dialog.winfo_exists()):
            self.t1_synth_dialog = GenerateSyntheticSequence(self, on_save=_accept_sequence)
        else:
            self.t1_synth_dialog.on_save = _accept_sequence  # update callback if needed
            self.t1_synth_dialog.show()  # bring back & focus

        self.wait_window(self.t1_synth_dialog)

    def t1_run_manager(self, event):
        if self.selected_file_index is None:
            messagebox.showinfo("No selection", "Please select a file to analyze.")
            return

        path = self.uploaded_files[self.selected_file_index]
        if not os.path.isfile(path):
            messagebox.showinfo("Error", "The selected file does not exist.")
            return

        if not self._t1_last_seq:
            messagebox.showinfo("Error", "The selected FASTA file contains no sequence data.")
            return

        start = self._parse_int(self.t1_ds.start_txt.get())
        end = self._parse_int(self.t1_ds.end_txt.get())
        seq = self._t1_last_seq[start:end]
        k = self.k_var.get()

        global foo_thread_1
        foo_thread_1 = threading.Thread(target=self.t1_run, args=(seq, start, end, k))
        foo_thread_1.daemon = True
        foo_thread_1.start()
        self.after(20, self.t1_check_thread)

    def t1_run(self, seq, start, end, k):
        self._t1_progress_status = "Counting 3-mers..."
        self._t1_progress = 0.0

        # Count kmers (map internal 0..1 to 0..0.55)
        counts = self._count_kmers(seq, 3, progress_cb=lambda p: setattr(self, "_t1_progress", 0.00 + 0.55 * p))
        # Labels (map internal 0..1 to 0.55..0.65)
        self._t1_progress_status = "Building 3-mer labels..."
        labels = self._labels_kmers(3, progress_cb=lambda p: setattr(self, "_t1_progress", 0.55 + 0.10 * p))

        # FCGR
        self._t1_progress_status = "Computing FCGR... Hang tight, this may take a while for long sequences... :)"
        self._t1_progress = 0.65
        fcgr = CGR(seq, k).get_fcgr_fast(progress_cb=lambda p: setattr(self, "_t1_progress", 0.65 + 0.35 * p),
                                         step=200_000)
        self._t1_progress_status = "Done."
        self._t1_progress = 1.0

        self.t1_fcgrs_dict = {"fcgr": fcgr, "b": start, "e": end, "seq_len": len(self.t1_ds.seq),
                              "seq": seq, "k": k, "counts": counts, "labels": labels}

    def t1_check_thread(self):
        # update UI from main thread
        if self.t1_progress_bar is not None:
            self.t1_progress_bar.set(getattr(self, "_t1_progress", 0.0) or 0.0)

        if self.t1_status_label is not None:
            self.t1_status_label.configure(text=getattr(self, "_t1_progress_status", ""))

        if foo_thread_1.is_alive():
            self.after(20, self.t1_check_thread)
        else:
            self.t1_progress_bar.set(1.0)
            if self.t1_status_label is not None:
                self.t1_status_label.configure(text=getattr(self, "_t1_progress_status", "Done"))

            # 3-mer frequency analysis histogram
            self.t1_hist_frame.configure(corner_radius=8, border_width=1, fg_color=COLORS["LIGHT_FRAME_COLOR"],
                                         border_color=COLORS["BORDER_COLOR"])
            self._draw_panel(frame=self.t1_hist_frame, fig_attr="t1_hist_fig", canvas_attr="t1_hist_canvas",
                             save_btn_attr="t1_hist_save_btn", save_command=lambda: self._save_figure("t1_hist_fig"),
                             placeholder_attr="t1_placeholder_label", fcgrs_dict=self.t1_fcgrs_dict,
                             panel_type="kmer_hist", )
            # FCGR plot
            self.t1_fcgr_frame.configure(corner_radius=8, border_width=1, fg_color=COLORS["FRAME_COLOR"],
                                         border_color=COLORS["BORDER_COLOR"])
            self._draw_panel(frame=self.t1_fcgr_frame, fig_attr="t1_fcgr_fig", canvas_attr="t1_fcgr_canvas",
                             save_btn_attr="t1_fcgr_save_btn", save_command=lambda: self._save_figure("t1_fcgr_fig"),
                             placeholder_attr=None, fcgrs_dict=self.t1_fcgrs_dict,
                             panel_type="fcgr", )

            # 3D FCGR plot
            self.t1_3d_fcgr_frame.configure(corner_radius=8, border_width=1, fg_color=COLORS["FRAME_COLOR"],
                                            border_color=COLORS["BORDER_COLOR"])
            self._draw_panel(frame=self.t1_3d_fcgr_frame, fig_attr="t1_3d_fcgr_fig", canvas_attr="t1_3d_fcgr_canvas",
                             save_btn_attr="t1_3d_fcgr_save_btn",
                             save_command=lambda: self._save_figure("t1_3d_fcgr_fig"), placeholder_attr=None,
                             fcgrs_dict=self.t1_fcgrs_dict, panel_type="fcgr_3d")

            self._update_t1_stats_table_from_fcgr(top_n=100)

    def _t1_disconnect_hist_events(self):
        if getattr(self, "t1_hist_canvas", None) is not None:
            for cid in getattr(self, "_t1_hist_cids", []):
                try:
                    self.t1_hist_canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._t1_hist_cids = []

    def _plot_kmer_histogram(self, fig, bg, seq_len, k, counts, labels, canvas):
        # avoid duplicate hover handlers
        self._t1_disconnect_hist_events()

        # Data
        total = int(counts.sum())
        subtitle = f"Length: {seq_len:,}  |  Valid {k}-mers: {total:,}  |  [{OVER_REP}: ≥ average frequency, {UNDER_REP}: < average frequency]"

        # Clear and axes
        fig.clf()
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)

        # Plot
        x = np.arange(len(counts))  # 64 for k=3

        # avg + colors + avg line
        avg = float(np.mean(counts)) if len(counts) else 0.0
        colors = np.where(counts >= avg, COLORS[OVER_REP], COLORS[UNDER_REP])  # green / blue
        bars = ax.bar(x, counts, width=0.85, color=colors)

        ax.axhline(avg, linestyle="--", linewidth=1.0, alpha=0.8, color="#2B2B2B")
        ax.text(0.99, avg, f"avg: {int(round(avg)):,}", ha="right", va="bottom",
                fontsize=8, transform=ax.get_yaxis_transform())

        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.15, top=0.95)
        fig.text(0.07, 1, subtitle, ha="left", va="top", fontsize=8)

        # X-axis ticks (clean spacing, not glued to y-axis)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="center", fontsize=7)
        ax.tick_params(axis="x", pad=2)
        ax.margins(x=0.01)
        ax.set_xlim(-0.8, len(counts) - 0.2)

        # Y-axis: plain numbers, correct formatting, smaller font
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.get_offset_text().set_visible(False)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}"))

        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

        # HOVER
        if len(bars) > 0:
            base_colors = [b.get_facecolor() for b in bars]
            hover_color = "#CC8899"
            hovered = {"idx": None}

            tooltip = ax.annotate("", xy=(0, 0), xytext=(12, 12), textcoords="offset points", ha="left",
                                  va="bottom", fontsize=8, color="white",
                                  bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.15",
                                            fc=(0.25, 0.25, 0.25, 0.90), ec=(1, 1, 1, 0.20), lw=0.8))
            tooltip.set_visible(False)
            tooltip.xyann = (12, 12)  # default offset

            def on_motion(event):
                if event.inaxes != ax:
                    # leaving axes: restore and hide tooltip
                    if hovered["idx"] is not None:
                        bars[hovered["idx"]].set_facecolor(base_colors[hovered["idx"]])
                        hovered["idx"] = None
                        tooltip.set_visible(False)
                        canvas.draw_idle()
                    return

                found_idx = None
                for i, b in enumerate(bars):
                    contains, _ = b.contains(event)
                    if contains:
                        found_idx = i
                        break

                if found_idx == hovered["idx"]:
                    return

                # restore previous hover color
                if hovered["idx"] is not None:
                    bars[hovered["idx"]].set_facecolor(base_colors[hovered["idx"]])

                if found_idx is not None:
                    # apply hover color
                    bars[found_idx].set_facecolor(hover_color)
                    # tooltip content + anchor near bar top
                    tooltip.set_text(f"{labels[found_idx]}\n{int(counts[found_idx]):,}")
                    tooltip.xy = (found_idx, counts[found_idx])
                    tooltip.set_visible(True)

                    # Keep tooltip inside axes bounds (robust, 2-pass)
                    def _place_tooltip_inside_axes():
                        tooltip.xyann = (12, 12)  # start default
                        for _ in range(2):
                            canvas.draw()  # force accurate bbox
                            renderer = fig.canvas.get_renderer()

                            bbox = tooltip.get_window_extent(renderer=renderer)
                            ax_bbox = ax.get_window_extent(renderer=renderer)

                            dx, dy = tooltip.xyann

                            if bbox.x1 > ax_bbox.x1:
                                dx = -bbox.width - 12
                            if bbox.x0 < ax_bbox.x0:
                                dx = 12

                            if bbox.y1 > ax_bbox.y1:
                                dy = -bbox.height - 12
                            if bbox.y0 < ax_bbox.y0:
                                dy = 12
                            tooltip.xyann = (dx, dy)

                    _place_tooltip_inside_axes()
                else:
                    tooltip.set_visible(False)

                hovered["idx"] = found_idx
                canvas.draw_idle()

            cid = canvas.mpl_connect("motion_notify_event", on_motion)
            self._t1_hist_cids.append(cid)

        canvas.draw_idle()

    @staticmethod
    def _count_kmers(seq, k, progress_cb=None):
        def prog(x):
            if progress_cb:
                progress_cb(float(x))

        prog(0.00)

        seq = (seq or "").upper()
        n = len(seq)
        if k <= 0:
            prog(1.00)
            return np.array([], dtype=np.int64)

        m = 4 ** k
        if n < k:
            prog(1.00)
            return np.zeros(m, dtype=np.int64)

        # ASCII -> 0..3 for A,C,G,T; invalid -> -1
        table = np.full(256, -1, dtype=np.int8)
        table[ord("A")] = 0
        table[ord("C")] = 1
        table[ord("G")] = 2
        table[ord("T")] = 3

        prog(0.10)

        s = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        x = table[s].astype(np.int16)
        if x.size < k:
            prog(1.00)
            return np.zeros(m, dtype=np.int64)

        prog(0.20)

        valid = (x >= 0).astype(np.int8)
        window_valid = np.convolve(valid, np.ones(k, dtype=np.int8), mode="valid") == k
        if not np.any(window_valid):
            prog(1.00)
            return np.zeros(m, dtype=np.int64)

        prog(0.35)

        pow4 = (4 ** np.arange(k - 1, -1, -1, dtype=np.int64))
        shape = (x.size - k + 1, k)
        strides = (x.strides[0], x.strides[0])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        prog(0.45)

        # Heavy part
        W = windows.astype(np.int64)
        total = W.shape[0]
        chunk = max(50_000, total // 20)  # ~20 updates, min chunk size

        codes_parts = []
        for i in range(0, total, chunk):
            j = min(i + chunk, total)
            codes_parts.append((W[i:j] * pow4).sum(axis=1))
            # progress from 0.45 -> 0.85 during chunking
            prog(0.45 + 0.40 * (j / total))

        codes = np.concatenate(codes_parts, axis=0)
        codes = codes[window_valid]

        prog(0.90)

        counts = np.bincount(codes, minlength=m).astype(np.int64)

        prog(1.00)
        return counts

    @staticmethod
    def _labels_kmers(k, progress_cb=None):
        def prog(x):
            if progress_cb:
                progress_cb(float(x))

        if k <= 0:
            prog(1.0)
            return []

        bases = np.array(list("ACGT"))
        m = 4 ** k
        labels = [""] * m

        # update ~50 times
        step = max(1, m // 50)

        for code in range(m):
            c = code
            chars = []
            for _ in range(k):
                chars.append(bases[c % 4])
                c //= 4
            labels[code] = "".join(reversed(chars))

            if (code % step) == 0:
                prog(code / m)

        prog(1.0)
        return labels

    @staticmethod
    def _xy_to_kmer(x: int, y: int, k: int) -> str:
        bits_to_base = {(0, 0): "C", (1, 0): "G", (0, 1): "A", (1, 1): "T"}
        out = []
        for i in range(k - 1, -1, -1):
            xb = (x >> i) & 1
            yb = (y >> i) & 1
            out.append(bits_to_base[(xb, yb)])
        return "".join(out)

    def _plot_fcgr_3d(self, fcgrs, bg=None, fig=None, canvas=None, include_zeros=True, filter_mode=None):
        import time
        from matplotlib.colors import LinearSegmentedColormap, Normalize
        from mpl_toolkits.mplot3d import proj3d

        # ---------- figure ----------
        Zfull = np.asarray(fcgrs["fcgr"], dtype=float)
        h, w = Zfull.shape
        Xfull, Yfull = np.meshgrid(np.arange(w), np.arange(h))

        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(bg)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.4, 0.4, 0.4, 0.3)
            axis._axinfo["grid"]["linewidth"] = 0.5
            axis._axinfo["grid"]["linestyle"] = "-"
        ax.tick_params(axis="x", pad=0, labelsize=8)
        ax.tick_params(axis="y", pad=0, labelsize=8)
        ax.tick_params(axis="z", pad=2, labelsize=8)

        over_under = LinearSegmentedColormap.from_list("over_under", [COLORS[UNDER_REP], COLORS[OVER_REP]])

        avg = float(Zfull.mean())

        if filter_mode in ("Over", "Under"):
            # --- 3D bar plot for filtered subsets ---
            xf = Xfull.ravel().astype(float)
            yf = Yfull.ravel().astype(float)
            zf = Zfull.ravel()
            nz_mask = zf > 0
            xf, yf, zf = xf[nz_mask], yf[nz_mask], zf[nz_mask]

            if filter_mode == "Over":
                sel = zf >= avg
                title = f"Over-represented  (≥ avg {avg:.1f})"
            else:
                sel = zf < avg
                title = f"Under-represented  (< avg {avg:.1f})"
            xf, yf, zf = xf[sel], yf[sel], zf[sel]

            # cap to top-N bars to stay responsive
            MAX_BARS = 500
            if zf.size > MAX_BARS:
                top_idx = np.argpartition(zf, -MAX_BARS)[-MAX_BARS:]
                xf, yf, zf = xf[top_idx], yf[top_idx], zf[top_idx]
                title = f"{title}"

            ax.text2D(0.5, 0.97, title, transform=ax.transAxes, ha="center", va="top", fontsize=8, color="black")

            if zf.size > 0:
                norm = Normalize(vmin=zf.min(), vmax=zf.max())
                bar_colors = over_under(norm(zf))
            else:
                bar_colors = COLORS[UNDER_REP]

            ax.bar3d(xf, yf, np.zeros_like(zf), 0.8, 0.8, zf,
                     color=bar_colors, shade=True, zsort="average")

            xs = xf.astype(int)
            ys = yf.astype(int)
            zs = zf
            zs_proj = zf   # bar3d is drawn at raw counts, projection matches
        else:
            # --- Surface plot for "All" (fast, full resolution) ---
            # Plot in asinh space so the surface isn't dominated by a few spikes,
            # but relabel the z-axis ticks with the corresponding real counts.
            Zdisp = np.arcsinh(Zfull)
            ax.plot_surface(Xfull, Yfull, Zdisp, cmap=over_under,
                            linewidth=0, antialiased=False, shade=True)

            # Replace asinh tick values with real counts (sinh is the inverse)
            from matplotlib.ticker import FuncFormatter
            ax.zaxis.set_major_formatter(
                FuncFormatter(lambda v, _: f"{int(round(np.sinh(v)))}")
            )

            xs = Xfull.ravel().astype(int)
            ys = Yfull.ravel().astype(int)
            zs = Zfull.ravel()        # raw counts — shown in tooltip
            zs_proj = Zdisp.ravel()   # asinh values — match what was rendered

        ax.set_zlabel("count", fontsize=10)
        ax.invert_yaxis()

        fig.subplots_adjust(left=0.05, right=0.82, bottom=0.08, top=0.98)

        # ---------- hover wiring needs a real canvas ----------
        if not hasattr(self, "_t1_fcgr3d_cids"):
            self._t1_fcgr3d_cids = []
        for cid in self._t1_fcgr3d_cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        self._t1_fcgr3d_cids = []

        # Tooltip anchored to axes (stable + readable)
        tooltip = ax.text2D(
            0.02, 0.98, "", transform=ax.transAxes, ha="left", va="top", fontsize=8, color="white",
            bbox=dict(boxstyle="round,pad=0.35", fc=(0.25, 0.25, 0.25, 0.90), ec=(1, 1, 1, 0.20), lw=0.8))
        tooltip.set_visible(False)

        # cache projected screen coords; recompute only when view changes / redraw
        state = {"xy_pix": None, "view": None, "last_idx": None, "last_t": 0.0}

        def _reproject():
            # project 3D points into 2D and then into pixel coords
            # zs_proj matches the actual rendered z-space (asinh for surface, raw for bars)
            x2, y2, _ = proj3d.proj_transform(xs, ys, zs_proj, ax.get_proj())
            disp = ax.transData.transform(np.column_stack([x2, y2]))  # pixel coords
            state["xy_pix"] = disp
            state["view"] = (ax.elev, ax.azim)
            state["last_idx"] = None

        def on_draw(event):
            # any draw can change projection (rotation/zoom)
            state["xy_pix"] = None

        def on_motion(event):
            if event.inaxes != ax:
                if tooltip.get_visible():
                    tooltip.set_visible(False)
                    canvas.draw_idle()
                return

            # throttle a bit to avoid heavy work on super-fast mouse motion
            t = time.time()
            if t - state["last_t"] < 0.03:  # ~30ms
                return
            state["last_t"] = t

            if state["xy_pix"] is None or state["view"] != (ax.elev, ax.azim):
                _reproject()

            xy = state["xy_pix"]
            if xy is None or len(xy) == 0:
                return

            mx, my = event.x, event.y
            d2 = (xy[:, 0] - mx) ** 2 + (xy[:, 1] - my) ** 2
            idx = int(np.argmin(d2))
            dist2 = float(d2[idx])

            # only show tooltip if cursor is close enough to a point
            if dist2 > 14 ** 2:
                if tooltip.get_visible():
                    tooltip.set_visible(False)
                    canvas.draw_idle()
                return

            if state["last_idx"] == idx and tooltip.get_visible():
                return

            x = int(xs[idx])
            y = int(ys[idx])
            z = float(zs[idx])
            km = self._xy_to_kmer(x, y, self.k_var.get())
            tooltip.set_text(f"{km}\n{int(round(z)):,}")

            tooltip.set_visible(True)
            state["last_idx"] = idx
            canvas.draw_idle()

        cid1 = canvas.mpl_connect("motion_notify_event", on_motion)
        cid2 = canvas.mpl_connect("draw_event", on_draw)
        self._t1_fcgr3d_cids.extend([cid1, cid2])

        return fig

    def _get_top_kmers_from_fcgr(self, fcgr, k, top_n=100, include_zeros=False):
        Z = np.asarray(fcgr, dtype=float)

        # flatten
        zs = Z.ravel()
        if not include_zeros:
            nz = zs != 0
            if not np.any(nz):
                return []
            idx_all = np.flatnonzero(nz)
            vals = zs[idx_all]
        else:
            idx_all = np.arange(zs.size)
            vals = zs

        # top-N without fully sorting everything
        n = min(top_n, vals.size)
        if n <= 0:
            return []

        top_local = np.argpartition(vals, -n)[-n:]
        top_local = top_local[np.argsort(vals[top_local])[::-1]]

        # map flat index -> (x,y)
        w = Z.shape[1]
        flat_idx = idx_all[top_local]
        ys = (flat_idx // w).astype(int)
        xs = (flat_idx % w).astype(int)

        out = []
        for x, y, v in zip(xs, ys, vals[top_local]):
            out.append((self._xy_to_kmer(int(x), int(y), int(k)), int(round(float(v)))))
        return out

    def _update_t1_stats_table_from_fcgr(self, top_n=100):
        from tkinter import ttk

        # clear old widgets
        for w in self.t1_stat_frame.winfo_children():
            w.destroy()

        fcgr = self.t1_fcgrs_dict.get("fcgr")
        k = int(self.t1_fcgrs_dict.get("k"))
        rows = self._get_top_kmers_from_fcgr(fcgr, k, top_n=top_n, include_zeros=False)

        # ------------------ STYLE ------------------
        style = ttk.Style()
        style.configure("Custom.Treeview", font=("Segoe UI", 12))
        style.configure("Custom.Treeview.Heading", font=("Segoe UI", 13, "bold"))

        # ------------------ TITLE ------------------
        (ctk.CTkLabel(self.t1_stat_frame, text=f"Top {top_n} {k}-mer Frequencies", font=("Segoe UI", 18, "bold"))
         .grid(row=0, column=0, columnspan=2, pady=(5, 2)))

        # ------------------ TREE ------------------
        columns = ("kmer", "value")
        tree = ttk.Treeview(self.t1_stat_frame, columns=columns, show="headings", height=18, selectmode="none",
                            style="Custom.Treeview")

        tree.heading("kmer", text=f"{k}-mer")
        tree.heading("value", text="Count")
        tree.column("kmer", anchor="center", width=110)
        tree.column("value", anchor="center", width=90)

        tree.tag_configure("oddrow", background="#2b2b2b")
        tree.tag_configure("evenrow", background="#242424")
        for i, (km, v) in enumerate(rows):
            tag = "evenrow" if i % 2 == 0 else "oddrow"
            tree.insert("", "end", values=(km, f"{v:,}"), tags=(tag,))

        sb = ttk.Scrollbar(self.t1_stat_frame, orient="vertical", command=tree.yview)
        tree.configure(yscrollcommand=sb.set)

        tree.grid(row=1, column=0, sticky="nsew")
        sb.grid(row=1, column=1, sticky="ns")

        # ------------------ DOWNLOAD BUTTON ------------------
        def _download_full_table():
            from tkinter import filedialog
            import csv
            path = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                initialfile=f"{k}mer_counts.csv",
                title="Save k-mer table",
            )
            if not path:
                return
            all_rows = self._get_top_kmers_from_fcgr(
                fcgr, k, top_n=4 ** k, include_zeros=True
            )
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([f"{k}-mer", "Count"])
                writer.writerows(all_rows)

        ctk.CTkButton(
            self.t1_stat_frame, text="⬇ Download Full Table",
            height=30, fg_color=COLORS["BORDER_COLOR"],
            hover_color=COLORS["FRAME_HOVER_COLOR"],
            command=_download_full_table,
        ).grid(row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(4, 6))

    def t2_sequence_selection_event(self, sender, value):
        # Set its sequence and sequence name
        sequence_path = self.uploaded_files[self.file_names.index(value)]
        self.t2_ds[sender].seq = self._read_fasta(sequence_path)[1]
        # Set start and end in scales
        try:
            self.t2_ds[sender].end_seq.set(len(self.t2_ds[sender].seq))
            if len(self.t2_ds[sender].seq) > 0:
                self.t2_start_seq_scale[sender].configure(state="normal", button_color=COLORS["BTN_COLOR"])
                self.t2_end_seq_scale[sender].configure(state="normal", button_color=COLORS["BTN_COLOR"])
            else:
                self.t2_start_seq_scale[sender].configure(state="disable", button_color=COLORS["DISABLED_BTN_COLOR"])
                self.t2_end_seq_scale[sender].configure(state="disable", button_color=COLORS["DISABLED_BTN_COLOR"])
            self.t2_start_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.t2_start_seq_scale[sender].set(0)
            self.t2_end_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.t2_end_seq_scale[sender].set(len(self.t2_ds[sender].seq))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load sequence {sender}.")

        # Set start and end in entries
        self.t2_sync_text_vars(self.t2_ds, sender)
        # Clear segment_size and disable segment size entry
        self.t2_segment_size_toggle.set(0)
        self.t2_segment_size.set("")
        self.t2_segment_entry.configure(state="disable")
        # Make end scales and entries normal
        for key, value in self.t2_end_seq_scale.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal", button_color=COLORS["BTN_COLOR"])
        for key, value in self.t2_end_seq_entry.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal")
        for key, value in self.t2_start_seq_entry.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal")

    def t2_segment_size_toggle_event(self):
        if self.t2_segment_size_toggle.get() == 0:
            # ---- turn OFF fixed window mode (variable end) ----
            self.t2_segment_size.set("")
            self.t2_segment_entry.configure(state="disable")

            # end scales
            for key, value in self.t2_end_seq_scale.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal", button_color=COLORS["BTN_COLOR"])
            for key, value in self.t2_end_seq_entry.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal")

        else:
            # ---- turn ON fixed window mode ----
            self.t2_segment_entry.configure(state="normal")
            # default = 500,000 but not larger than the smallest non-empty seq
            default_size = 500000
            non_empty_lengths = [len(ds.seq) for ds in self.t2_ds.values() if ds.seq != '']
            if non_empty_lengths:
                default_size = min(default_size, min(non_empty_lengths))

            self.t2_segment_size.set(self._format_int(default_size))

            # end scales become disabled in fixed window mode
            for key, value in self.t2_end_seq_scale.items():
                value.configure(state="disable", button_color=COLORS["DISABLED_BTN_COLOR"])
            for key, value in self.t2_end_seq_entry.items():
                value.configure(state="disable")

            seg_size = self._parse_int(self.t2_segment_size.get())
            if seg_size is None or seg_size <= 0:
                return  # shouldn't happen, but be safe

            for key, ds in self.t2_ds.items():
                if len(ds.seq) > 0:
                    self.t2_apply_segment_size_constraint(key, seg_size)

        # After toggling, sync text fields to show the corrected values
        for key, ds in self.t2_ds.items():
            self.t2_sync_text_vars(self.t2_ds, key)

    def t2_sequence_value_change(self, sender, value):
        if sender == "0":  # Segment size changed
            seg_size = self._parse_int(self.t2_segment_size.get())

            # Empty or non-digit input is invalid
            if seg_size is None or seg_size <= 0:
                return messagebox.showerror("Error", "Segment size must be a positive integer.")

            for key, value in self.t2_ds.items():
                if self.t2_ds[key].seq == '':
                    continue
                # If segment size is out of range, send an error and set to 500,000
                if seg_size < 1 or seg_size > len(self.t2_ds[key].seq):
                    # reset to default if out of range
                    default_size = min(500000, len(self.t2_ds[key].seq))
                    self.t2_segment_size.set(self._format_int(default_size))
                    messagebox.showerror("Error", "Segment size is out of range.")
                    seg_size = default_size

                # enforce start + seg_size <= len(seq) and keep seg_size fixed
                self.t2_apply_segment_size_constraint(key, seg_size)

            # normalize display of the segment size
            self.t2_segment_size.set(self._format_int(seg_size))

        elif sender in ["1", "2"]:  # Scale changed
            if self.t2_ds[sender].seq == '':
                return
            seq_len = len(self.t2_ds[sender].seq)
            slider_val = float(value)

            if self.t2_segment_size_toggle.get() == 1:
                # -------- FIXED WINDOW MODE --------
                seg_size = self._parse_int(self.t2_segment_size.get())
                if seg_size is None or seg_size <= 0:
                    return messagebox.showerror("Error", "Segment size must be a positive integer.")

                # Apply fixed-size constraint (this also ensures start < end)
                self.t2_apply_segment_size_constraint(sender, seg_size)
                # Extra safety, but usually redundant now:
                self.t2_ensure_start_before_end(self.t2_ds, sender, show_message=False)

            else:
                # -------- VARIABLE WINDOW MODE --------
                # We keep the nice auto-fix but without any hidden window-size limits.
                start = self.t2_ds[sender].start_seq.get()
                end = self.t2_ds[sender].end_seq.get()

                # Clamp raw values into [0, seq_len]
                start = max(0, min(start, seq_len))
                end = max(0, min(end, seq_len))

                # Decide which slider moved: the one whose current value is closest to slider_val
                # (CTkSlider passes its own value as 'value' into the callback)
                if abs(slider_val - start) <= abs(slider_val - end):
                    # User moved START slider
                    start = slider_val

                    # Ensure start is in range
                    if start < 0:
                        start = 0
                    if start > seq_len:
                        start = seq_len

                    # Maintain start < end
                    if start >= end:
                        if start >= seq_len - 1:
                            # Push to the very end: [seq_len-1, seq_len]
                            start = max(seq_len - 1, 0)
                            end = seq_len
                        else:
                            end = start + 1
                else:
                    # User moved END slider
                    end = slider_val

                    if end < 0:
                        end = 0
                    if end > seq_len:
                        end = seq_len

                    # Maintain start < end
                    if end <= start:
                        if end <= 1:
                            start = 0
                            end = max(1, end)
                        else:
                            start = end - 1

                # Write back
                self.t2_ds[sender].start_seq.set(int(start))
                self.t2_ds[sender].end_seq.set(int(end))

        elif sender in ["3"]:  # Entry changed
            # pull values from start_txt / end_txt into start_seq / end_seq
            for key, ds in self.t2_ds.items():
                self.t2_reverse_sync_text_vars(self.t2_ds, key)
            if self.t2_segment_size_toggle.get() == 1:
                seg_size = self._parse_int(self.t2_segment_size.get())
                if seg_size is None or seg_size <= 0:
                    return messagebox.showerror("Error", "Segment size must be a positive integer.")
                for key, ds in self.t2_ds.items():
                    if ds.seq == '':
                        continue
                    # Entries typed so respect fixed segment size + bounds
                    self.t2_apply_segment_size_constraint(key, seg_size)

        # Sync text fields (with commas) at the end
        for key, ds in self.t2_ds.items():
            self.t2_sync_text_vars(self.t2_ds, key)

    def t2_plot(self):
        if self.t2_ds["1"].seq == "" or self.t2_ds["2"].seq == "":
            return messagebox.showerror("Error", "Please upload or choose the sequences first.")
        if self.k_var.get() == 0:
            return messagebox.showerror("Error", "Please choose the k-mer value.")
        if self.dist_metric.get() == "":
            return messagebox.showerror("Error", "Please choose the distance measure.")
        fcgrs_dict = {}
        for key in self.t2_ds.keys():
            fcgrs_dict[key] = {}
            seq = self.t2_ds[key].seq[self.t2_ds[key].start_seq.get():self.t2_ds[key].end_seq.get()]
            if self.t2_rc[key].get():
                seq = self._reverse_complement(seq)
            if self.t2_shuffle[key].get():
                seq = list(seq)
                random.shuffle(seq)
                seq = ''.join(seq)
            fcgrs_dict[key]["fcgr"] = CGR(seq, self.k_var.get()).get_fcgr()

            fcgrs_dict[key]["seq_len"] = len(self.t2_ds[key].seq)
            fcgrs_dict[key]["b"] = self.t2_ds[key].start_seq.get()
            fcgrs_dict[key]["e"] = self.t2_ds[key].end_seq.get()

        diff = fcgrs_dict["2"]["fcgr"] - fcgrs_dict["1"]["fcgr"]
        fcgrs_dict["diff"] = diff
        distance_value = get_dist(fcgrs_dict["1"]["fcgr"], fcgrs_dict["2"]["fcgr"], dist_m=self.dist_metric.get())
        fcgrs_dict["distance"] = distance_value

        self._draw_panel(frame=self.t2_display_frame, fig_attr="t2_fig", canvas_attr="t2_canvas",
                         save_btn_attr="t2_save_btn", save_command=lambda: self._save_figure("t2_fig"),
                         placeholder_attr="t2_placeholder_label", fcgrs_dict=fcgrs_dict, )

    def t2_sync_text_vars(self, ds, sender):
        ds[sender].start_txt.set(self._format_int(ds[sender].start_seq.get()))
        ds[sender].end_txt.set(self._format_int(ds[sender].end_seq.get()))

    def t2_reverse_sync_text_vars(self, ds, sender):
        start_raw = ds[sender].start_txt.get().strip()
        end_raw = ds[sender].end_txt.get().strip()
        seq_len = len(ds[sender].seq)

        # Parse integers (supports comma inputs)
        start_val = self._parse_int(start_raw) if start_raw != "" else 0
        end_val = self._parse_int(end_raw) if end_raw != "" else len(ds[sender].seq)

        # Prevent error if the entry is not digits (if its "" it is okay)
        if start_val is None:
            ds[sender].start_txt.set("0")
            return messagebox.showerror("Error", "Start value must be a positive integer.")

        if end_val is None:
            ds[sender].end_txt.set(str(len(ds[sender].seq)))
            return messagebox.showerror("Error", "End value must be a positive integer.")

        # Validate range
        if start_val < 0 or start_val > seq_len:
            ds[sender].start_txt.set("0")
            return messagebox.showerror("Error", "Start value is out of range.")

        if end_val < 0 or end_val > seq_len:
            ds[sender].end_txt.set(str(seq_len))
            return messagebox.showerror("Error", "End value is out of range.")

        # Update sequence values
        ds[sender].start_seq.set(start_val)
        ds[sender].end_seq.set(end_val)

        self.t2_ensure_start_before_end(ds, sender, show_message=True)

    def t2_ensure_start_before_end(self, ds, sender, show_message=True):
        seq_len = len(ds[sender].seq)
        start_val = ds[sender].start_seq.get()
        end_val = ds[sender].end_seq.get()

        if start_val >= end_val:
            # Simple policy:
            # - try to move end to start+1 if possible
            # - otherwise move start to end-1
            if start_val < seq_len:
                end_val = start_val + 1
                ds[sender].end_seq.set(end_val)
                ds[sender].end_txt.set(self._format_int(end_val))
            else:
                start_val = max(0, end_val - 1)
                ds[sender].start_seq.set(start_val)
                ds[sender].start_txt.set(self._format_int(start_val))

            if show_message:
                messagebox.showerror("Error", "Start value must be smaller than end value.")

    def t2_apply_segment_size_constraint(self, key, seg_size):
        ds = self.t2_ds[key]
        seq_len = len(ds.seq)

        if seq_len == 0 or seg_size is None or seg_size <= 0:
            return

        # segment size cannot be larger than the sequence
        if seg_size > seq_len:
            seg_size = seq_len

        max_start = seq_len - seg_size
        start = ds.start_seq.get()

        if start < 0:
            start = 0
        if start > max_start:
            start = max_start

        ds.start_seq.set(start)
        ds.end_seq.set(start + seg_size)

    # --------------------------------------------------
    # Helper functions for Common Reference tab
    # --------------------------------------------------
    def t3_sequence_selection_event(self, sender, value):
        sequence_path = self.uploaded_files[self.file_names.index(value)]
        sequence = self._read_fasta(sequence_path)[1]
        seq_len = len(sequence)
        if sender == "1":
            self.t3_ds['1'].seq = sequence
            # Set the Reference sequence as the sequence
            self.t3_ds['2'].seq = sequence
            self.t3_ds['2'].seq_name.set(value)
            # Set the sequence length
            self.t3_seq_len_label.configure(text=f"Sequence length={seq_len:,}")
            self.t3_rep_len_label.configure(text=f"Reference length={seq_len:,}")
            self.t3_use_rep_algo.set(1)
            self.t3_use_rep_checkbox_event()
        elif sender == "2":
            self.t3_ds['2'].seq = sequence
            self.t3_ds['2'].seq_name.set(value)
            self.t3_rep_len_label.configure(text=f"Reference length={seq_len:,}")
            self.t3_use_rep_algo.set(0)
            self.t3_use_rep_checkbox_event()

    def t3_use_rep_checkbox_event(self):
        if self.t3_use_rep_algo.get() == 1:
            # Enable the algo type combobox
            self.t3_rep_type_combobox.configure(state="normal")
            if self.t3_rep_algo_type.get() == "aRepSeg":
                self.t3_rep_n_entry.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])
            # Disable start and end
            self.t3_start_label.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
            self.t3_start_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
            self.t3_end_label.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
            self.t3_end_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
        else:
            # Disable the algo type combobox
            self.t3_rep_type_combobox.configure(state="disabled")
            self.t3_rep_n_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
            # Enable start and end
            self.t3_start_label.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])
            self.t3_start_entry.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])
            self.t3_end_label.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])
            self.t3_end_entry.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])

    def t3_rep_algo_change_event(self, value):
        if value == "aRepSeg":
            self.t3_rep_number.set("30")
            self.t3_rep_n_entry.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])
        elif value == "RepSeg":
            self.t3_rep_number.set("1")
            self.t3_rep_n_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])

    def t3_run_manager(self, event):
        if self.t3_ds["1"].seq == "" or self.t3_ds["2"].seq == "":
            return messagebox.showerror("Error", "Please upload or choose the sequences first.")

        # if using the representative check the validity of combobox and the entry value
        if self.t3_use_rep_algo.get() == 1:
            if self.t3_rep_algo_type.get() == "":
                return messagebox.showerror("Error", "Please choose the representative algorithm.")
            if self.t3_rep_algo_type.get() == "aRepSeg":
                n_val = self._parse_int(self.t3_rep_number.get())
                if n_val is None or n_val <= 0:
                    return messagebox.showerror("Error", "Please enter a valid positive integer for n.")

        # if not using representative check the validity of start and end entries
        if self.t3_use_rep_algo.get() == 0:
            start_str = self.t3_ds['2'].start_txt.get().strip()
            end_str = self.t3_ds['2'].end_txt.get().strip()
            if start_str == "" or end_str == "":
                return messagebox.showerror("Error", "Start and end values must be positive integers, "
                                                     "within reference sequence length.")
            # Allow numbers with commas
            start = self._parse_int(start_str)
            end = self._parse_int(end_str)
            if start is None or end is None:
                return messagebox.showerror("Error", "Start and end values must be positive integers, "
                                                     "within sequence length.")
            seq_len = len(self.t3_ds['2'].seq)
            if start < 0 or start > seq_len or end < 0 or end > seq_len or start >= end:
                return messagebox.showerror("Error", "Start and end values must be positive integers, "
                                                     "within sequence length, and start < end.")

        # check the validity of the entry for segment size (should be a positive digit smaller than the sequence length)
        seg_str = self.t3_segment_size.get().strip()
        if seg_str == "":
            return messagebox.showerror("Error", "Segment size must be a positive integer.")
        seg_size = self._parse_int(seg_str)
        if seg_size is None:
            return messagebox.showerror("Error", "Segment size must be a positive integer.")
        if seg_size <= 0 or seg_size > len(self.t3_ds["2"].seq):
            return messagebox.showerror("Error", "Segment size is out of range.")

        # check the validity of other options
        if self.k_var.get() == 0:
            return messagebox.showerror("Error", "Please choose the k-mer value.")
        if self.dist_metric.get() == "":
            return messagebox.showerror("Error", "Please choose the distance measure.")
        if self.t3_plot_type.get() == "":
            return messagebox.showerror("Error", "Please choose the plot type.")
        global foo_thread_2
        foo_thread_2 = threading.Thread(target=self.t3_run)
        foo_thread_2.daemon = True
        foo_thread_2.start()
        self.after(20, self.t3_check_thread)

    def t3_run(self):
        self.t3_cgr_distance_history = []
        seg_size = self._parse_int(self.t3_segment_size.get().strip())

        path = f"{self.temp_output_path}/t3_run"
        if not os.path.exists(path):
            os.makedirs(path)
        fcgrs_dict = {}

        if self.t3_use_rep_algo.get() == 0:
            t3_step_length = np.floor(len(self.t3_ds["1"].seq) / seg_size)
            t3_step_length = int(t3_step_length)
            self._t3_progress_status = "Computing FCGR for the Reference sequence... Hang tight, this may take a while for a long sequence..."
            self._t3_progress = 0.0

            ref_b = self._parse_int(self.t3_ds['2'].start_txt.get().strip())
            ref_e = self._parse_int(self.t3_ds['2'].end_txt.get().strip())
            ref_cgr = CGR(self.t3_ds['2'].seq[ref_b:ref_e], self.k_var.get())

            self._t3_progress = 1.0 / (t3_step_length + 2)
            im1 = ref_cgr.get_fcgr()
            self._t3_progress = 2.0 / (t3_step_length + 2)

            fcgrs_dict["ref"] = {"fcgr": im1, "b": ref_b, "e": ref_e, "seq_len": len(self.t3_ds["2"].seq)}

            distance_matrix = np.zeros((t3_step_length, t3_step_length))

            # the sliding sequence
            for i in range(t3_step_length):
                self._t3_progress_status = f"Processing segment {i + 1} of {t3_step_length}..."
                self._t3_progress = (i + 3) / (t3_step_length + 2)
                b2 = i * seg_size
                e2 = (i + 1) * seg_size
                cgr2 = CGR(self.t3_ds["1"].seq[b2:e2], self.k_var.get())
                im2 = cgr2.get_fcgr()

                diff = im2 - im1
                dist = get_dist(im1, im2, dist_m=self.dist_metric.get())
                self.t3_cgr_distance_history.append(dist)

                fcgrs_dict[i] = {"fcgr": im2, "b": b2, "e": e2, "seq_len": len(self.t3_ds["1"].seq),
                                 "diff": diff, "distance": dist}

                # Calculate pairwise distance between this new image and all previous images
                for j in range(i + 1):
                    distance_matrix[i, j] = get_dist(fcgrs_dict[i]["fcgr"], fcgrs_dict[j]["fcgr"],
                                                     dist_m=self.dist_metric.get())
                    distance_matrix[j, i] = distance_matrix[i, j]

            ref_d = np.asarray(self.t3_cgr_distance_history, dtype=np.float32)  # (t3_step_length,)
            D = np.zeros((t3_step_length + 1, t3_step_length + 1), dtype=np.float32)
            # put reference-to-segment distances into first row/col
            D[0, 1:] = ref_d
            D[1:, 0] = ref_d
            # put segment-to-segment distances into remaining cells
            D[1:, 1:] = distance_matrix.astype(np.float32)

            with open(f"{path}/t3_run.pkl", 'wb') as f:
                pickle.dump(fcgrs_dict, f)
            with open(f"{path}/t3_distance_matrix.pkl", 'wb') as f:
                pickle.dump(D, f)

        elif self.t3_use_rep_algo.get() == 1:
            # TODO: need to fill the run for representative algorithm
            pass
        else:
            messagebox.showerror("Error", "Unknown representative algorithm option.")

    def t3_check_thread(self):
        if self.t3_progress_bar is not None:
            self.t3_progress_bar.set(getattr(self, "_t3_progress", 0.0) or 0.0)

        if self.t3_status_label is not None:
            self.t3_status_label.configure(text=getattr(self, "_t3_progress_status", ""))

        if foo_thread_2.is_alive():
            self.after(20, self.t3_check_thread)
        else:
            self.t3_progress_bar.set(1.0)
            if self.t3_status_label is not None:
                self.t3_status_label.configure(text=getattr(self, "_t3_progress_status", "Done"))
            self.t3_scale.configure(to=int(len(self.t3_cgr_distance_history) - 1))  # Update the scale range
            self.t3_pic_num.set(0)

            # Display the 3d plot, image, and the chart
            with open(f"{self.temp_output_path}/t3_run/t3_distance_matrix.pkl", 'rb') as handle:
                D = pickle.load(handle)
            with open(f"{self.temp_output_path}/t3_run/t3_run.pkl", "rb") as f:
                fcgrs_dict = pickle.load(f)

            self.t3_seg_info = {k: {"b": v.get("b"), "e": v.get("e")}
                                for k, v in fcgrs_dict.items() if isinstance(k, int) and isinstance(v, dict)}

            ref = fcgrs_dict.get("ref")
            self.t3_ref_info = {"b": ref.get("b"), "e": ref.get("e")} if isinstance(ref, dict) else None
            del fcgrs_dict

            # MDS (3d)
            self._t3_mds_drawn = False
            self._draw_panel(frame=self.t3_3d_display_frame, fig_attr="t3_3d_fig",
                             canvas_attr="t3_3d_canvas", save_btn_attr=None,
                             save_command=lambda: self._save_figure("t3_3d_fig"),
                             placeholder_attr="t3_3d_placeholder_label", fcgrs_dict=None, index=0, panel_type="mds",
                             D=D)

            # Display image and chart
            self.after_idle(lambda: self.t3_change_images(0, None))

    def t3_change_images(self, index, value):
        index = round(value) if value is not None else index
        # MDS change color
        # self._t3_mds_set_selected(index)
        self._plot_mds(fig=self.t3_3d_fig, bg=None, D=None, index=index, canvas=self.t3_3d_canvas)
        # FCGR image
        self._draw_panel(frame=self.t3_fcgr_display_frame, fig_attr="t3_fcgr_fig", canvas_attr="t3_fcgr_canvas",
                         save_btn_attr="t3_fcgr_save_btn", save_command=lambda: self._save_figure("t3_fcgr_fig"),
                         placeholder_attr="t3_fcgr_placeholder_label", fcgrs_dict=None, index=index)
        # Chart
        self._draw_panel(frame=self.t3_plot_display_frame, fig_attr="t3_plot_fig",
                         canvas_attr="t3_plot_canvas", save_btn_attr="t3_plot_save_btn",
                         save_command=lambda: self._save_figure("t3_plot_fig"),
                         placeholder_attr="t3_plot_placeholder_label", fcgrs_dict=None, index=index, panel_type="chart")

    def t3_move_previous(self, value):
        pic_num = self.t3_pic_num
        if pic_num.get() > 0:
            pic_num.set(pic_num.get() - 1)
            self.t3_change_images(pic_num.get(), None)

    def t3_move_next(self, value):
        pic_num = self.t3_pic_num
        dist_history_len = len(self.t3_cgr_distance_history)
        if pic_num.get() < dist_history_len - 1:
            pic_num.set(pic_num.get() + 1)
            self.t3_change_images(pic_num.get(), None)

    def t3_plot_change_event(self, value):
        # if the plot type is changed, check if there is data to plot
        if not self.t3_cgr_distance_history or len(self.t3_cgr_distance_history) == 0:
            return
        # re-plot the chart with the new plot type
        self._draw_panel(frame=self.t3_plot_display_frame, fig_attr="t3_plot_fig",
                         canvas_attr="t3_plot_canvas", save_btn_attr="t3_plot_save_btn",
                         save_command=lambda: self._save_figure("t3_plot_fig"),
                         placeholder_attr="t3_plot_placeholder_label", fcgrs_dict=None,
                         index=self.t3_pic_num.get(), panel_type="chart")

    def _t3_disconnect_plot_events(self):
        """Disconnect old mpl_connect callbacks for the Tab-3 distance plot."""
        if getattr(self, "_t3_plot_cids", None) and getattr(self, "t3_plot_canvas", None) is not None:
            for cid in self._t3_plot_cids:
                try:
                    self.t3_plot_canvas.mpl_disconnect(cid)
                except Exception:
                    pass
        self._t3_plot_cids = []

    def _plot_charts(self, fig, bg, dists, index, canvas):
        # --- Clamp index safely (important on first draw) ---
        try:
            index = int(index)
        except Exception:
            index = 0
        index = max(0, min(index, len(dists) - 1))

        # --- Reset old callbacks (prevents duplicate click handlers) ---
        self._t3_disconnect_plot_events()

        # --- Clear and build axes ---
        fig.clf()
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)
        ax.set_facecolor(bg)

        ax.set_xlabel("Segment index")
        ax.set_ylabel(f"{self.dist_metric.get()} distance")

        plot_type = self.t3_plot_type.get().strip()

        # Shared colors
        selected_color = "red"
        hover_color = "#CC8899"

        # ============================================================
        # LINE PLOT
        # ============================================================
        if plot_type == "Line plot":
            xs = list(range(len(dists)))
            ys = dists

            default_color = "tab:blue"

            # --- Draw the LINE (visual only, not interactive) ---
            ax.plot(xs, ys, linestyle='-', linewidth=1.5, color=default_color, zorder=1)

            # --- Draw MARKERS as scatter (for interaction) ---
            colors = [default_color] * len(xs)
            colors[index] = selected_color

            sc = ax.scatter(xs, ys, c=colors, s=30, picker=True, zorder=2)

            ax.set_xlabel("Segment index")
            ax.set_ylabel(f"{self.dist_metric.get()} distance")

            hovered_idx = {"idx": None}

            def base_color(i):
                return selected_color if i == index else default_color

            # ---------- CLICK ----------
            def on_pick(event):
                if not hasattr(event, "ind") or event.ind is None or len(event.ind) == 0:
                    return
                new_idx = int(event.ind[0])
                self.t3_pic_num.set(new_idx)
                self.t3_change_images(new_idx, None)

            cid_pick = canvas.mpl_connect("pick_event", on_pick)
            self._t3_plot_cids.append(cid_pick)

            # ---------- HOVER ----------
            def on_motion(event):
                if event.inaxes != ax:
                    return

                contains, info = sc.contains(event)
                if not contains:
                    if hovered_idx["idx"] is not None:
                        colors[hovered_idx["idx"]] = base_color(hovered_idx["idx"])
                        sc.set_color(colors)
                        hovered_idx["idx"] = None
                        canvas.draw_idle()
                    return

                new_hover = int(info["ind"][0])
                if new_hover == hovered_idx["idx"]:
                    return

                # restore previous hover
                if hovered_idx["idx"] is not None:
                    colors[hovered_idx["idx"]] = base_color(hovered_idx["idx"])

                # apply hover color
                colors[new_hover] = hover_color
                sc.set_color(colors)
                hovered_idx["idx"] = new_hover
                canvas.draw_idle()

            cid_motion = canvas.mpl_connect("motion_notify_event", on_motion)
            self._t3_plot_cids.append(cid_motion)

            canvas.draw_idle()
            return

        # ============================================================
        # HISTOGRAM PLOT
        # ============================================================
        if plot_type == "Histogram plot":
            ax.set_xlabel(f"{self.dist_metric.get()} distance")
            ax.set_ylabel("Count")

            n, bins, patches = ax.hist(dists, bins=min(30, max(5, int(len(dists) ** 0.5))))

            # selected vertical marker
            ax.axvline(dists[index], linestyle="--", linewidth=2, color=selected_color)

            canvas.draw_idle()
            return

        # ============================================================
        # BAR PLOT
        # ============================================================
        if plot_type == "Bar plot":
            xs = range(len(dists))
            bars = ax.bar(xs, dists, picker=True)

            ax.set_xlabel("Segment index")
            ax.set_ylabel(f"{self.dist_metric.get()} distance")

            # --- Colors ---
            default_color = bars[0].get_facecolor() if bars else None

            # --- Apply initial colors ---
            for i, b in enumerate(bars):
                if i == index:
                    b.set_facecolor(selected_color)
                else:
                    b.set_facecolor(default_color)

            # --- Map bar artist -> index ---
            bar_to_idx = {bar: i for i, bar in enumerate(bars)}

            # Track currently hovered bar
            hovered_bar = {"bar": None}

            # ---------- CLICK ----------
            def on_pick(event):
                bar = getattr(event, "artist", None)
                new_idx = bar_to_idx.get(bar, None)
                if new_idx is None:
                    return
                self.t3_pic_num.set(new_idx)
                self.t3_change_images(new_idx, None)

            cid_pick = canvas.mpl_connect("pick_event", on_pick)
            self._t3_plot_cids.append(cid_pick)

            # ---------- HOVER ----------
            def on_motion(event):
                if event.inaxes != ax:
                    return

                found_bar = None
                for bar in bars:
                    contains, _ = bar.contains(event)
                    if contains:
                        found_bar = bar
                        break

                # No change → do nothing
                if found_bar is hovered_bar["bar"]:
                    return

                # Restore previous hovered bar color
                if hovered_bar["bar"] is not None:
                    i = bar_to_idx[hovered_bar["bar"]]
                    if i == index:
                        hovered_bar["bar"].set_facecolor(selected_color)
                    else:
                        hovered_bar["bar"].set_facecolor(default_color)

                # Apply hover color
                if found_bar is not None:
                    found_bar.set_facecolor(hover_color)

                hovered_bar["bar"] = found_bar
                canvas.draw_idle()

            cid_motion = canvas.mpl_connect("motion_notify_event", on_motion)
            self._t3_plot_cids.append(cid_motion)

            canvas.draw_idle()

    def _t3_mds_set_selected(self, index):
        fig = getattr(self, "t3_3d_fig", None)
        canvas = getattr(self, "t3_3d_canvas", None)
        if fig is None or canvas is None or not fig.axes:
            return

        ax = fig.axes[0]

        sc_seg = None
        for col in ax.collections:
            if col.get_gid() == "t3_mds_segments":
                sc_seg = col
                break
        if sc_seg is None or not hasattr(sc_seg, "_t3_colors"):
            return

        n = sc_seg._t3_colors.shape[0]
        try:
            index = int(index)
        except Exception:
            index = 0
        index = max(0, min(index, n - 1))

        old = sc_seg._t3_selected_idx
        if old == index:
            return

        # restore old selected color (unless it's hovered)
        if sc_seg._t3_hovered_idx == old:
            sc_seg._t3_colors[old] = sc_seg._t3_hover
        else:
            sc_seg._t3_colors[old] = sc_seg._t3_default

        # set new selected color (unless it is hovered -> keep selected stronger)
        sc_seg._t3_selected_idx = index
        sc_seg._t3_colors[index] = sc_seg._t3_selected

        sc_seg.set_facecolors(sc_seg._t3_colors)
        canvas.draw_idle()

    def _plot_mds(self, fig, bg, D, index, canvas):
        def _t3_update_seg_legend(seg_index: int):
            leg = getattr(canvas, "_t3_mds_legend", None)
            if leg is None or not hasattr(leg, "_t3_seg_texts"):
                return

            sb = self.t3_seg_info[seg_index].get("b")
            se = self.t3_seg_info[seg_index].get("e")
            slen = se - sb

            def _fmt(v):
                return f"{v:,}" if isinstance(v, (int, np.integer)) else "—"

            t_begin, t_end, t_len = leg._t3_seg_texts
            t_begin.set_text(f"Start:    {_fmt(sb)}")
            t_end.set_text(f"End:      {_fmt(se)}")
            t_len.set_text(f"Length: {_fmt(slen)}")

            canvas.draw_idle()

        # If already drawn, just update selection color and return
        if getattr(self, "_t3_mds_drawn", False):
            self._t3_mds_set_selected(index)
            _t3_update_seg_legend(index)
            canvas.draw_idle()
            return

        D = np.asarray(D, dtype=float)
        n_total = D.shape[0]  # (ref + segments)
        n_seg = n_total - 1
        if n_seg <= 0:
            return
        index = max(0, min(int(index), n_seg - 1))

        # ---- Compute 3D MDS embedding once ----
        mds = MDS(n_components=3, dissimilarity="precomputed", random_state=24)
        X = mds.fit_transform(D)  # shape (n_total, 3)

        X_ref = X[0]
        X_seg = X[1:]  # shape (n_seg, 3) -> seg indices 0..n_seg-1

        # ---- Draw ----
        fig.clf()
        fig.patch.set_facecolor(bg)

        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(bg)
        ax.set_position([0.05, 0.06, 1.0, 0.94])  # left bottom width height

        # Subtle grid
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis._axinfo["grid"]["color"] = (0.4, 0.4, 0.4, 0.3)
            axis._axinfo["grid"]["linewidth"] = 0.5
            axis._axinfo["grid"]["linestyle"] = "-"

        # Labels
        label_tick_color = "black"
        ax.set_xlabel("X", labelpad=-5, size=5, color=label_tick_color)
        ax.set_ylabel("Y", labelpad=-5, size=5, color=label_tick_color)
        ax.set_zlabel("Z", labelpad=-5, size=5, color=label_tick_color)

        # Tick params
        ax.tick_params(axis="x", pad=0, labelsize=5)
        ax.tick_params(axis="y", pad=0, labelsize=5)
        ax.tick_params(axis="z", pad=0, labelsize=5)

        for tick in ax.get_xticklabels():
            tick.set_color(label_tick_color)
        for tick in ax.get_yticklabels():
            tick.set_color(label_tick_color)
        for tick in ax.get_zticklabels():
            tick.set_color(label_tick_color)

        # ---- Colors ----
        default_color = to_rgba("tab:blue")
        selected_color = to_rgba("red")
        hover_color = to_rgba("#CC8899")
        ref_color = to_rgba("purple")
        edge_color = to_rgba("#1E2A56")

        # ---- Segment points: pickable ----
        xs, ys, zs = X_seg[:, 0], X_seg[:, 1], X_seg[:, 2]

        colors = np.tile(default_color, (n_seg, 1))
        colors[index] = selected_color

        sc_seg = ax.scatter(xs, ys, zs, s=10, picker=True, depthshade=False, edgecolor=edge_color, linewidths=0.2)
        sc_seg.set_gid("t3_mds_segments")
        sc_seg.set_facecolors(colors)

        # Store ONLY tiny state on the artist itself (not big arrays on self)
        sc_seg._t3_colors = colors
        sc_seg._t3_default = default_color
        sc_seg._t3_selected = selected_color
        sc_seg._t3_hover = hover_color
        sc_seg._t3_selected_idx = index
        sc_seg._t3_hovered_idx = None

        # ---- Reference point: separate artist ----
        sc_ref = ax.scatter([X_ref[0]], [X_ref[1]], [X_ref[2]],
                            c=[ref_color], s=20, picker=False, zorder=5, marker='*')
        sc_ref.set_gid("t3_mds_ref")

        # ---- Prevent duplicate callbacks (store cids on canvas, not on self) ----
        if not hasattr(canvas, "_t3_mds_cids"):
            canvas._t3_mds_cids = []
        for cid in canvas._t3_mds_cids:
            try:
                canvas.mpl_disconnect(cid)
            except Exception:
                pass
        canvas._t3_mds_cids = []

        # ---- Pick ----
        def on_pick(event):
            if event.artist is not sc_seg:
                return
            if not hasattr(event, "ind") or event.ind is None or len(event.ind) == 0:
                return

            new_idx = int(event.ind[0])  # 0..n_seg-1
            self.t3_pic_num.set(new_idx)
            self.t3_change_images(new_idx, None)
            _t3_update_seg_legend(new_idx)

        canvas._t3_mds_cids.append(canvas.mpl_connect("pick_event", on_pick))

        # ---- Hover ----
        def _apply_colors():
            sc_seg.set_facecolors(sc_seg._t3_colors)
            canvas.draw_idle()

        def on_motion(event):
            if event.inaxes != ax:
                # clear hover when leaving axes
                if sc_seg._t3_hovered_idx is not None:
                    i_old = sc_seg._t3_hovered_idx
                    sc_seg._t3_hovered_idx = None
                    # restore old hovered point color
                    if i_old == sc_seg._t3_selected_idx:
                        sc_seg._t3_colors[i_old] = sc_seg._t3_selected
                    else:
                        sc_seg._t3_colors[i_old] = sc_seg._t3_default
                    _apply_colors()
                return

            contains, info = sc_seg.contains(event)
            if not contains:
                if sc_seg._t3_hovered_idx is not None:
                    i_old = sc_seg._t3_hovered_idx
                    sc_seg._t3_hovered_idx = None
                    sc_seg._t3_colors[i_old] = (
                        sc_seg._t3_selected if i_old == sc_seg._t3_selected_idx else sc_seg._t3_default
                    )
                    _apply_colors()
                return

            i_new = int(info["ind"][0])
            if i_new == sc_seg._t3_hovered_idx:
                return

            # restore previous hovered
            if sc_seg._t3_hovered_idx is not None:
                i_old = sc_seg._t3_hovered_idx
                sc_seg._t3_colors[i_old] = (
                    sc_seg._t3_selected if i_old == sc_seg._t3_selected_idx else sc_seg._t3_default
                )

            # apply new hover (but don't override selected)
            sc_seg._t3_hovered_idx = i_new
            if i_new != sc_seg._t3_selected_idx:
                sc_seg._t3_colors[i_new] = sc_seg._t3_hover

            _apply_colors()

        canvas._t3_mds_cids.append(canvas.mpl_connect("motion_notify_event", on_motion))

        self._t3_mds_drawn = True

        # Legend with segment and reference info
        # Compute info
        rb = self.t3_ref_info.get("b")
        re = self.t3_ref_info.get("e")
        rlen = re - rb

        # Segment info (current selected index)
        sb = self.t3_seg_info[index].get("b")
        se = self.t3_seg_info[index].get("e")
        slen = se - sb

        def _fmt(v):
            return f"{v:,}" if isinstance(v, (int, np.integer)) else "—"

        legend_elements = [
            Line2D([0], [0], marker='*', color='none',
                   markerfacecolor=ref_color, markeredgecolor=ref_color,
                   markersize=5, label="Reference"),
            Line2D([], [], linestyle='none', label=f"Start:    {_fmt(rb)}"),
            Line2D([], [], linestyle='none', label=f"End:      {_fmt(re)}"),
            Line2D([], [], linestyle='none', label=f"Length: {_fmt(rlen)}"),

            Line2D([0], [0], marker='o', color='none',
                   markerfacecolor=selected_color, markeredgecolor=edge_color, markeredgewidth=0.4,
                   markersize=4, label="Segment"),
            Line2D([], [], linestyle='none', label=f"Start:    {_fmt(sb)}"),
            Line2D([], [], linestyle='none', label=f"End:      {_fmt(se)}"),
            Line2D([], [], linestyle='none', label=f"Length: {_fmt(slen)}"),
        ]

        leg = fig.legend(handles=legend_elements, fontsize=5, loc="upper left", bbox_to_anchor=(0.0, 1.00),
                         frameon=True, labelspacing=0.25, borderpad=0.35, handlelength=0.8, handletextpad=0.4, )
        # make legend background semi-transparent
        frame = leg.get_frame()
        frame.set_alpha(0.5)  # 0.0 = fully transparent, 1.0 = opaque
        frame.set_facecolor("white")

        # Text order matches the labels above.
        texts = leg.get_texts()
        # indices: 0="Reference", 1..3=ref lines, 4="Segment", 5..7=seg lines
        leg._t3_ref_texts = texts[1:4]
        leg._t3_seg_texts = texts[5:8]
        canvas._t3_mds_legend = leg

        canvas.draw_idle()

    # --------------------------------------------------
    # General helper functions
    # --------------------------------------------------
    @staticmethod
    def _resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        if getattr(sys, 'frozen', False):  # If running as a bundled .app
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    @staticmethod
    def _read_fasta(file_path):
        sequence = ""
        record_count = 0
        with open(file_path) as file:
            for line in file:
                line = line.strip()
                if line.startswith(">"):
                    record_count += 1
                    if record_count > 1:
                        raise ValueError("FASTA contains multiple records; expected exactly one.")
                    file_name = (line[1:].split()[0] or "unknown")
                else:
                    sequence += line
        return file_name, sequence

    @staticmethod
    def _reverse_complement(sequence):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        bases = [complement[base] for base in sequence]
        bases = reversed(bases)
        return ''.join(bases)

    @staticmethod
    def _scaling(chromosome_length):
        scale = 1_000_000
        while (chromosome_length / scale) < 2:
            scale //= 1000
        if scale == 1_000_000:
            scaling = "Mbp"
        elif scale == 1_000:
            scaling = "Kbp"
        else:
            scaling = "bp"
        return scale, scaling

    @staticmethod
    def _parse_int(text):
        if text is None:
            return None
        cleaned = text.replace(",", "").strip()
        return int(cleaned) if cleaned.isdigit() else None

    @staticmethod
    def _format_int(value):
        return f"{value:,}"

    def _entry_change(self, ds, which, event=None):
        start_raw = ds.start_txt.get().strip()
        end_raw = ds.end_txt.get().strip()

        # No sequence selected
        if ds.seq == '':
            if which == "start":
                ds.start_txt.set("")
            else:
                ds.end_txt.set("")
            return messagebox.showerror("Error", "No sequence selected.")

        # If the other field is empty, don't validate yet
        if start_raw == "" or end_raw == "":
            return

        # Parse comma-friendly integer values
        start = self._parse_int(start_raw)
        end = self._parse_int(end_raw)

        # Parse-int check failed
        if start is None or end is None:
            if which == "start":
                ds.start_txt.set("")
            else:
                ds.end_txt.set("")
            return messagebox.showerror("Error", "Start and end values must be positive integers, "
                                                 "within sequence length.")

        seq_len = len(ds.seq)
        # Range validation
        if start < 0 or start > seq_len or end < 0 or end > seq_len or start >= end:
            if which == "start":
                ds.start_txt.set("")
            else:
                ds.end_txt.set("")
            return messagebox.showerror("Error", "Start and end values must be positive integers, "
                                                 "within sequence length, and start < end.")

        # All good so update stored values
        ds.start_seq.set(start)
        ds.end_seq.set(end)

        # Normalize formatting with commas so display is consistent
        ds.start_txt.set(self._format_int(start))
        ds.end_txt.set(self._format_int(end))

    def _save_figure(self, fig_attr):
        fig = getattr(self, fig_attr, None)
        if fig is None:
            return messagebox.showerror("Error", "No figure to save. Please plot first.")

        file_path = fd.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"),
                                                    ("SVG Image", "*.svg"), ("All Files", "*.*")],
                                         title="Save figure")
        if not file_path:
            return  # user cancelled

        try:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception:
            messagebox.showerror("Error", "Could not save figure.")

    def _plot_fcgrs(self, fcgrs, bg=None, fig=None, index=0):
        if fig is None:
            fig = plt.Figure()
        extent = (0, 1, 0, 1)
        if bg is not None:
            fig.patch.set_facecolor(bg)

        if fig == self.t1_fcgr_fig:
            ax = fig.add_subplot(111)

            scale, scaling = self._scaling(fcgrs["seq_len"])
            b = fcgrs["b"]
            e = fcgrs["e"]
            length = fcgrs["e"] - fcgrs["b"]

            # plot the data
            # img = CGR.array2img(fcgrs["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            # img = Image.fromarray(img)
            # ax.imshow(img, cmap='gray', extent=extent)  # Greys_r
            f = CGR.normalize(fcgrs["fcgr"])
            # f = FCGRNormalizer._fcgr_to_freq(np.asarray(fcgrs["fcgr"], dtype=float))
            self.fcgr_normalizer.fit([f], ks=[self.k_var.get()])
            V = self.fcgr_normalizer.transform01(f, k=self.k_var.get(), L=length)
            img_uint8 = FCGRNormalizer.to_uint8_from_01(V, white_is_high=True)
            img = Image.fromarray(img_uint8)
            ax.imshow(img, cmap="gray", origin="upper")
            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax.set_title(f'{round(b / scale, 2)} - {round(e / scale, 2)} {scaling}')
            corner_labels = [("A", (0.00, -0.01), (-0.05, -0.05), "right", "top"),  # bottom-left
                             ("C", (0.00, 0.99), (-0.05, +0.05), "right", "bottom"),  # top-left
                             ("T", (1.00, -0.01), (+0.05, -0.05), "left", "top"),  # bottom-right
                             ("G", (1.00, 0.99), (+0.05, +0.05), "left", "bottom")]  # top-right
            for text, xy, offset, ha, va in corner_labels:
                ax.annotate(text, xy=xy, xycoords="axes fraction", xytext=offset, textcoords="offset points",
                            ha=ha, va=va, fontsize=10, color="black", clip_on=False)

        if fig == self.t2_fig:
            ax1, ax2, ax3 = fig.subplots(1, 3)

            scale_1, scaling_1 = self._scaling(fcgrs["1"]["seq_len"])
            b1 = fcgrs["1"]["b"]
            e1 = fcgrs["1"]["e"]
            length1 = e1 - b1
            scale_2, scaling_2 = self._scaling(fcgrs["2"]["seq_len"])
            b2 = fcgrs["2"]["b"]
            e2 = fcgrs["2"]["e"]
            length2 = e2 - b2

            # plot the data on the subplots
            # # img1 = CGR.array2img(fcgrs["1"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            # # img1 = Image.fromarray(img1)
            # # ax1.imshow(img1, cmap='gray', extent=extent)  # Reds_r
            f1 = CGR.normalize(fcgrs["1"]["fcgr"])
            # f = FCGRNormalizer._fcgr_to_freq(np.asarray(fcgrs["1"]["fcgr"], dtype=float))
            self.fcgr_normalizer.fit([f1], ks=[self.k_var.get()])
            V = self.fcgr_normalizer.transform01(f1, k=self.k_var.get(), L=length1)
            img_uint8 = FCGRNormalizer.to_uint8_from_01(V, white_is_high=True)
            img1 = Image.fromarray(img_uint8)
            ax1.imshow(img1, cmap="gray", origin="upper")
            ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax1.set_title(f'Sequence 1\n{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')

            import matplotlib.colors as mcolors
            norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
            im2 = ax2.imshow(fcgrs['diff'], cmap='seismic', norm=norm, extent=extent)
            ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax2.set_title(f'Difference\ndistance = {round(fcgrs["distance"], 4)}')

            # # img2 = CGR.array2img(fcgrs["2"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            # # img2 = Image.fromarray(img2)
            # # ax3.imshow(img2, cmap='gray', extent=extent)  # Blues_r
            f2 = CGR.normalize(fcgrs["2"]["fcgr"])
            # f = FCGRNormalizer._fcgr_to_freq(np.asarray(fcgrs["2"]["fcgr"], dtype=float))
            self.fcgr_normalizer.fit([f2], ks=[self.k_var.get()])
            V = self.fcgr_normalizer.transform01(f2, k=self.k_var.get(), L=length2)
            img_uint8 = FCGRNormalizer.to_uint8_from_01(V, white_is_high=True)
            img2 = Image.fromarray(img_uint8)
            ax3.imshow(img2, cmap="gray", origin="upper")
            ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax3.set_title(f'Sequence 2\n{round(b2 / scale_2, 2)} - {round(e2 / scale_2, 2)} {scaling_2}')

            # --- add color panel ---
            fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
            cbar_ax2 = fig.add_axes([0.36, 0.1, 0.3, 0.02])  # Adjust position as needed
            cbar = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
            cbar.set_label(f'Red: Greater k-mer value in Sequence 1 , Blue: Greater k-mer value in Sequence 2',
                           fontsize=10)
            cbar.ax.xaxis.set_label_position('top')  # Position label at top of colorbar
            cbar.ax.xaxis.labelpad = 5
            cbar.ax.tick_params(labelsize=8)

        elif fig == self.t3_fcgr_fig:
            ax1, ax3 = fig.subplots(1, 2)

            scale_1, scaling_1 = self._scaling(fcgrs["ref"]["seq_len"])
            b1 = fcgrs["ref"]["b"]
            e1 = fcgrs["ref"]["e"]
            length1 = e1 - b1
            scale_2, scaling_2 = self._scaling(fcgrs[index]["seq_len"])
            b2 = fcgrs[index]["b"]
            e2 = fcgrs[index]["e"]
            length2 = e2 - b2

            # plot the data on the subplots
            # img1 = CGR.array2img(fcgrs["ref"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            # img1 = Image.fromarray(img1)
            # ax1.imshow(img1, cmap='gray', extent=extent)  # Reds_r
            f1 = CGR.normalize(fcgrs["ref"]["fcgr"])
            self.fcgr_normalizer.fit([f1], ks=[self.k_var.get()])
            V = self.fcgr_normalizer.transform01(f1, k=self.k_var.get(), L=length1)
            img_uint8 = FCGRNormalizer.to_uint8_from_01(V, white_is_high=True)
            img1 = Image.fromarray(img_uint8)
            ax1.imshow(img1, cmap="gray", origin="upper")
            ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax1.set_title(f'Reference\n{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')

            # img2 = CGR.array2img(fcgrs[index]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            # img2 = Image.fromarray(img2)
            # ax3.imshow(img2, cmap='gray', extent=extent)  # Blues_r
            f2 = CGR.normalize(fcgrs[index]["fcgr"])
            self.fcgr_normalizer.fit([f2], ks=[self.k_var.get()])
            V = self.fcgr_normalizer.transform01(f2, k=self.k_var.get(), L=length2)
            img_uint8 = FCGRNormalizer.to_uint8_from_01(V, white_is_high=True)
            img2 = Image.fromarray(img_uint8)
            ax3.imshow(img2, cmap="gray", origin="upper")
            ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax3.set_title(f'Segment\n{round(b2 / scale_2, 2)} - {round(e2 / scale_2, 2)} {scaling_2}')

            # --- add distance text below both panels ---
            fig.subplots_adjust(bottom=0.12)  # make room for the text
            fig.text(0.5, 0.06, f"Distance = {round(fcgrs[index]['distance'], 4)}",
                     ha="center", va="center", fontsize=14)

        return fig

    def _draw_panel(self, frame, fig_attr, canvas_attr, save_btn_attr, save_command, placeholder_attr, fcgrs_dict,
                    index=None, panel_type="fcgr", D=None):
        # --- 1) Figure setup ---
        bg = frame.cget("fg_color")

        fig = getattr(self, fig_attr, None)
        if fig is None:
            frame.update_idletasks()

            if fig_attr == "t3_fcgr_fig" or fig_attr == "t3_plot_fig":
                dpi = 80
            elif fig_attr == "t3_3d_fig":
                dpi = 150
            else:
                dpi = 120

            fig = plt.Figure(dpi=dpi)
            fig.patch.set_facecolor(bg)
            setattr(self, fig_attr, fig)

        # --- 2) Canvas setup ---
        canvas = getattr(self, canvas_attr, None)
        needs_new_canvas = (canvas is None
                            or not canvas.get_tk_widget().winfo_exists()
                            or canvas.get_tk_widget().master is not frame)
        if needs_new_canvas:
            canvas = FigureCanvasTkAgg(fig, master=frame)
            widget = canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            setattr(self, canvas_attr, canvas)

            # --- Save button setup ---
            if panel_type in ("fcgr", "chart", "kmer_hist"):
                save_btn = getattr(self, save_btn_attr, None)
                if save_btn is None or not save_btn.winfo_exists() or save_btn.master is not frame:
                    save_btn = ctk.CTkButton(master=frame, text="💾", width=30, height=30,
                                             fg_color=COLORS["BORDER_COLOR"], hover_color=COLORS["FRAME_HOVER_COLOR"],
                                             command=save_command, )
                    save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)
                    setattr(self, save_btn_attr, save_btn)
            if panel_type in ("mds", "fcgr_3d"):
                toolbar_attr = f"{canvas_attr}_toolbar"
                toolbar = getattr(self, toolbar_attr, None)

                # If canvas changed, drop old toolbar
                if toolbar is not None and getattr(toolbar, "canvas", None) is not canvas:
                    try:
                        toolbar.destroy()
                    except Exception:
                        pass
                    toolbar = None

                if toolbar is None or not getattr(toolbar, "winfo_exists", lambda: False)():
                    toolbar = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
                    toolbar.update()
                    setattr(self, toolbar_attr, toolbar)

                nav_state_attr = f"{canvas_attr}_nav_state"
                if not hasattr(self, nav_state_attr):
                    setattr(self, nav_state_attr, {"pan_on": False})
                nav_state = getattr(self, nav_state_attr)
                nav_state["pan_on"] = False
                nav_state["base_set"] = False  # re-capture base view after each redraw

                BTN_ON = COLORS["BTN_COLOR"]
                BTN_OFF = COLORS["BORDER_COLOR"]
                BTN_HOVER = COLORS["FRAME_HOVER_COLOR"]

                # Helper to place buttons in a row (bottom-left)
                def _mk_btn(name_attr, text, cmd, x):
                    btn = getattr(self, name_attr, None)
                    if btn is None or not btn.winfo_exists() or btn.master is not frame:
                        btn = ctk.CTkButton(master=frame, text=text, width=30, height=30,
                                            fg_color=BTN_OFF, hover_color=BTN_HOVER, command=cmd)
                        setattr(self, name_attr, btn)
                    btn.place(relx=0.01, rely=0.99, anchor="sw", x=x)
                    return btn

                def _set_btn_active(btn, active: bool):
                    btn.configure(fg_color=BTN_ON if active else BTN_OFF, hover_color=BTN_ON if active else BTN_HOVER)

                def _get_ax():
                    return fig.axes[0] if fig.axes else None

                def _ensure_base_view():
                    if nav_state.get("base_set", False):
                        return
                    ax0 = _get_ax()
                    if ax0 is None:
                        return

                    nav_state["base_xlim"] = ax0.get_xlim()
                    nav_state["base_ylim"] = ax0.get_ylim()
                    if hasattr(ax0, "get_zlim"):
                        nav_state["base_zlim"] = ax0.get_zlim()

                    # 3D camera (so reset also restores rotation)
                    if hasattr(ax0, "elev") and hasattr(ax0, "azim"):
                        nav_state["base_elev"] = ax0.elev
                        nav_state["base_azim"] = ax0.azim
                    if hasattr(ax0, "roll"):
                        nav_state["base_roll"] = ax0.roll

                    nav_state["base_set"] = True

                nav_state["_ensure_base_view"] = _ensure_base_view

                def _zoom_out(step=1.25):
                    _ensure_base_view()

                    ax = _get_ax()
                    if ax is None:
                        return

                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()

                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    hx = (x1 - x0) / 2.0 * step
                    hy = (y1 - y0) / 2.0 * step

                    ax.set_xlim(cx - hx, cx + hx)
                    ax.set_ylim(cy - hy, cy + hy)

                    if hasattr(ax, "get_zlim"):
                        z0, z1 = ax.get_zlim()
                        cz = (z0 + z1) / 2.0
                        hz = (z1 - z0) / 2.0 * step
                        ax.set_zlim(cz - hz, cz + hz)

                    canvas.draw_idle()

                def _zoom_in(step=1.25):
                    _ensure_base_view()

                    ax = _get_ax()
                    if ax is None:
                        return

                    x0, x1 = ax.get_xlim()
                    y0, y1 = ax.get_ylim()

                    cx = (x0 + x1) / 2.0
                    cy = (y0 + y1) / 2.0
                    hx = (x1 - x0) / 2.0 / step
                    hy = (y1 - y0) / 2.0 / step

                    ax.set_xlim(cx - hx, cx + hx)
                    ax.set_ylim(cy - hy, cy + hy)

                    if hasattr(ax, "get_zlim"):
                        z0, z1 = ax.get_zlim()
                        cz = (z0 + z1) / 2.0
                        hz = (z1 - z0) / 2.0 / step
                        ax.set_zlim(cz - hz, cz + hz)

                    canvas.draw_idle()

                # Create buttons
                reset_btn = _mk_btn(f"{canvas_attr}_reset_btn", "🏠", None, x=0)
                save_btn = _mk_btn(f"{canvas_attr}_save_btn", "💾", toolbar.save_figure, x=33)
                zoomin_btn = _mk_btn(f"{canvas_attr}_zoomin_btn", "➕", _zoom_in, x=66)
                zoomout_btn = _mk_btn(f"{canvas_attr}_zoomout_btn", "➖", _zoom_out, x=99)
                pan_btn = _mk_btn(f"{canvas_attr}_pan_btn", "✋", None, x=132)

                def _toggle_pan():
                    _ensure_base_view()
                    toolbar.pan()
                    nav_state["pan_on"] = not nav_state["pan_on"]
                    _set_btn_active(pan_btn, nav_state["pan_on"])

                def _reset():
                    _ensure_base_view()

                    if nav_state["pan_on"]:
                        toolbar.pan()
                        nav_state["pan_on"] = False
                        _set_btn_active(pan_btn, False)

                    ax = _get_ax()
                    if ax is None:
                        return

                    if "base_xlim" in nav_state:
                        ax.set_xlim(nav_state["base_xlim"])
                    if "base_ylim" in nav_state:
                        ax.set_ylim(nav_state["base_ylim"])
                    if hasattr(ax, "set_zlim") and "base_zlim" in nav_state:
                        ax.set_zlim(nav_state["base_zlim"])

                    # restore 3D camera
                    if hasattr(ax, "view_init") and "base_elev" in nav_state and "base_azim" in nav_state:
                        try:
                            if "base_roll" in nav_state:
                                ax.view_init(elev=nav_state["base_elev"], azim=nav_state["base_azim"],
                                             roll=nav_state["base_roll"])
                            else:
                                ax.view_init(elev=nav_state["base_elev"], azim=nav_state["base_azim"])
                        except TypeError:
                            ax.view_init(elev=nav_state["base_elev"], azim=nav_state["base_azim"])
                    canvas.draw_idle()

                # Now assign commands (after functions exist)
                pan_btn.configure(command=_toggle_pan)
                reset_btn.configure(command=_reset)

                _set_btn_active(pan_btn, nav_state["pan_on"])
                # frame.after(0, _ensure_base_view)

                # Filter segmented button — only for the 3D FCGR bar panel
                if panel_type == "fcgr_3d":
                    filter_btn_attr = f"{canvas_attr}_filter_btn"
                    filter_btn = getattr(self, filter_btn_attr, None)
                    if filter_btn is None or not filter_btn.winfo_exists() or filter_btn.master is not frame:
                        def _on_filter_change(_val):
                            self._draw_panel(
                                frame=self.t1_3d_fcgr_frame,
                                fig_attr="t1_3d_fcgr_fig",
                                canvas_attr="t1_3d_fcgr_canvas",
                                save_btn_attr="t1_3d_fcgr_save_btn",
                                save_command=lambda: self._save_figure("t1_3d_fcgr_fig"),
                                placeholder_attr=None,
                                fcgrs_dict=self.t1_fcgrs_dict,
                                panel_type="fcgr_3d",
                            )
                        filter_btn = ctk.CTkSegmentedButton(
                            master=frame, values=["All", "Over", "Under"],
                            variable=self.t1_3d_filter_var,
                            command=_on_filter_change,
                            width=165, height=26,
                            fg_color=COLORS["BORDER_COLOR"],
                            selected_color=COLORS["BTN_COLOR"],
                            selected_hover_color=COLORS["BTN_COLOR"],
                            unselected_color=COLORS["BORDER_COLOR"],
                            unselected_hover_color=COLORS["FRAME_HOVER_COLOR"],
                        )
                        filter_btn.place(relx=0.99, rely=0.01, anchor="ne")
                        setattr(self, filter_btn_attr, filter_btn)

        # --- 3) Clear figure and re-plot ---
        fig.clear()
        if panel_type == "fcgr":
            # If we are in third tab and no fcgrs_dict provided, load from pickle
            if fig_attr == "t3_fcgr_fig" and not fcgrs_dict:
                with open(f"{self.temp_output_path}/t3_run/t3_run.pkl", 'rb') as handle:
                    fcgrs_dict = pickle.load(handle)
            self._plot_fcgrs(fcgrs_dict, bg=bg, fig=fig, index=index)
        if panel_type == "fcgr_3d":
            self._plot_fcgr_3d(fcgrs_dict, bg=bg, fig=fig, canvas=canvas,
                               filter_mode=self.t1_3d_filter_var.get())
        elif panel_type == "chart":
            dists = list(self.t3_cgr_distance_history)
            self._plot_charts(fig=fig, bg=bg, dists=dists, index=index, canvas=canvas)
        elif panel_type == "mds":
            self._plot_mds(fig=fig, bg=bg, D=D, index=index, canvas=canvas)
        elif panel_type == "kmer_hist":
            seq_len = fcgrs_dict["e"] - fcgrs_dict["b"]
            # k = fcgrs_dict["k"]
            counts = fcgrs_dict["counts"]
            labels = fcgrs_dict["labels"]
            self._plot_kmer_histogram(fig=fig, bg=bg, seq_len=seq_len, k=3, counts=counts, labels=labels, canvas=canvas)

        # --- 4) Hide placeholder if present ---
        if placeholder_attr:
            placeholder = getattr(self, placeholder_attr, None)
            if placeholder is not None and placeholder.winfo_exists():
                try:
                    placeholder.place_forget()
                except Exception:
                    pass

        # --- 5) Redraw canvas ---
        canvas.draw()

        # Eagerly capture the base view after every redraw so reset always
        # returns to the initial position of the current plot.
        nav_state_attr = f"{canvas_attr}_nav_state"
        if hasattr(self, nav_state_attr):
            _ns = getattr(self, nav_state_attr)
            _ebv = _ns.get("_ensure_base_view")
            if callable(_ebv):
                frame.after_idle(_ebv)


class GenerateSyntheticSequence(ctk.CTkToplevel):
    def __init__(self, parent, on_save):
        super().__init__(parent)
        self.parent = parent
        self.on_save = on_save
        self.t1_generated_sequence = ""
        self.t2_generated_sequence = ""
        self.t3_generated_sequence = ""

        self.title("Generate Synthetic Sequence")
        # make it modal & on top of parent
        self.transient(self.parent)
        self.grab_set()
        self.focus_set()
        # size/center relative to parent (not the screen)
        self.parent.update_idletasks()
        w = int(self.parent.winfo_width() * 0.5)
        h = int(self.parent.winfo_height() * 0.6)
        x = self.parent.winfo_rootx() + (self.parent.winfo_width() - w) // 2
        y = self.parent.winfo_rooty() + (self.parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        # We have different tabviews
        tabview = ctk.CTkTabview(self, width=w - 20, height=h - 20)
        tabview.pack(padx=10, pady=10, fill="both", expand=True)
        tab_names = ["Entropy", "2-mers", "k-mers"]
        for name in tab_names:
            tabview.add(name)

        # ------------------------- state variables -------------------------
        # Entropy tab variables
        self.t1_frame = None
        self.t1_fig = None
        self.t1_canvas = None
        self.t1_save_btn = None
        self.t1_placeholder_label = None
        self.t1_k_var = ctk.IntVar(value=6)
        self.t1_r_var = ctk.DoubleVar(value=1.0)
        self.t1_r_value_label = None
        self.t1_last_valid_seq_len = None
        self.t1_seq_len = ctk.StringVar(value="500,000")
        self.t1_seq_len_real = None

        # 2mer tab variables
        self.t2_frame = None
        self.t2_fig = None
        self.t2_canvas = None
        self.t2_save_btn = None
        self.t2_placeholder_label = None
        self.k_var_dict = {}
        self.k_value_label_dict = {}
        self.t2_kmers = generate_kmers(2)
        self.t2_seq_len = ctk.StringVar(value="500,000")
        self.t2_seq_len_real = None

        # kmer tab variables
        self.t3_frame = None
        self.t3_fig = None
        self.t3_canvas = None
        self.t3_save_btn = None
        self.t3_placeholder_label = None
        self.t3_k_var = ctk.IntVar(value=3)
        self.t3_kmers = generate_kmers(self.t3_k_var.get())
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.t3_kmers)}
        self.logits = np.zeros(len(self.t3_kmers), dtype=float)
        self.t3_current_kmer = None
        self.t3_kmer_entry = None
        self.t3_slider_var = ctk.DoubleVar(value=0.0)
        self.t3_kmer_slider = None
        self.t3_kmer_label = None
        self.t3_summary_box = None
        self.t3_seq_len = ctk.StringVar(value="500,000")
        self.t3_seq_len_real = None

        # ------------------------- Build each tab once (state persists when switching tabs) -------------------------
        self._build_entropy_tab(tabview.tab(tab_names[0]))
        self._build_2mer_tab(tabview.tab(tab_names[1]))
        self._build_kmer_tab(tabview.tab(tab_names[2]))

    # ------------------------------------------------------------------
    # Tab 1: Entropy
    # ------------------------------------------------------------------
    def _build_entropy_tab(self, tab):
        tab.grid_rowconfigure(0, weight=50)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_propagate(False)

        config_frame = ctk.CTkFrame(tab, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        config_frame.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_columnconfigure(1, weight=1)
        config_frame.grid_columnconfigure(2, weight=1)
        config_frame.grid_propagate(False)

        # k-mer size
        ctk.CTkLabel(config_frame, text="k-mer:").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        (ctk.CTkComboBox(config_frame, values=KMERS_SYNTH, state="readonly", variable=self.t1_k_var, width=80)
         .grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(10, 0)))

        # sequence length
        ctk.CTkLabel(config_frame, text="Sequence length:").grid(row=1, column=0, sticky="w", padx=10, pady=(10, 0))
        segment_entry = ctk.CTkEntry(config_frame, textvariable=self.t1_seq_len)
        segment_entry.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ew")
        segment_entry.bind('<FocusOut>', self._sequence_length_change)
        segment_entry.bind('<Key-Return>', self._sequence_length_change)
        (ctk.CTkLabel(config_frame, text="Length after generate:", text_color=COLORS["TEXT_DISABLE_COLOR"],
                      font=('Cambria', 10)).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 0)))
        if self.t1_seq_len_real is not None and self.t1_seq_len_real.winfo_exists():
            text = self.t1_seq_len_real.cget("text")
        else:
            text = ""
        self.t1_seq_len_real = ctk.CTkLabel(config_frame, text=text, font=('Cambria', 10),
                                            text_color=COLORS["TEXT_DISABLE_COLOR"], anchor="w")
        self.t1_seq_len_real.grid(row=2, column=1, sticky="ew", padx=(5, 5), pady=(0, 0))
        self.t1_seq_len_real.grid_propagate(False)

        # entropy scaling factor
        (ctk.CTkLabel(config_frame, text="Entropy scaling factor:")
         .grid(row=3, column=0, sticky="w", padx=10, pady=(10, 0)))
        (ctk.CTkSlider(config_frame, from_=0.25, to=1.0, variable=self.t1_r_var, width=150)
         .grid(row=3, column=1, padx=(0, 5), pady=(10, 0), sticky="ew"))
        self.t1_r_value_label = ctk.CTkLabel(config_frame, text=f"{self.t1_r_var.get():.2f}", width=60)
        self.t1_r_value_label.grid(row=3, column=2, padx=(0, 10), pady=(10, 0), sticky="w")
        self.t1_r_var.trace_add("write", self.t1_update_r_label)

        # Generate button
        (ctk.CTkButton(config_frame, text="Generate", command=lambda: self.generate_sequence("t1"))
         .grid(row=4, column=0, columnspan=3, padx=10, pady=(10, 10)))

        # ------------------------- Right panel -------------------------
        self.t1_frame = ctk.CTkFrame(tab, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                     fg_color="white")
        self.t1_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.t1_frame.grid_columnconfigure(0, weight=1)
        self.t1_frame.grid_rowconfigure(0, weight=1)
        self.t1_frame.grid_propagate(False)

        self.t1_placeholder_label = ctk.CTkLabel(master=self.t1_frame, text="Plot Area", text_color="black")
        self.t1_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")

        if getattr(self, "t1_fig", None) is not None:
            self.t1_canvas = FigureCanvasTkAgg(self.t1_fig, master=self.t1_frame)
            widget = self.t1_canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t1_canvas.draw()

            if getattr(self, "t1_save_btn", None) is not None and self.t1_save_btn.winfo_exists():
                try:
                    self.t1_save_btn.destroy()
                except Exception:
                    pass
            self.t1_save_btn = ctk.CTkButton(master=self.t1_frame, text="💾", width=30, height=30,
                                             fg_color=COLORS["BORDER_COLOR"], hover_color=COLORS["FRAME_HOVER_COLOR"],
                                             command=partial(self._save_figure, "t1_fig"))
            self.t1_save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)

        # Save
        (ctk.CTkButton(tab, text="Save Sequence", command=lambda: self.save_sequence("t1"))
         .grid(row=1, column=1, padx=5, pady=5))

    def t1_update_r_label(self, *args):
        self.t1_r_value_label.configure(text=f"{self.t1_r_var.get():.2f}")

    # ------------------------------------------------------------------
    # Tab 2: 2-mers
    # ------------------------------------------------------------------
    def _build_2mer_tab(self, tab):
        tab.grid_rowconfigure(0, weight=50)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_propagate(False)

        config_frame = ctk.CTkFrame(tab, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        config_frame.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=0)
        config_frame.grid_columnconfigure(1, weight=1)
        config_frame.grid_columnconfigure(2, weight=0)
        config_frame.grid_propagate(False)

        # k-mer sliders
        last_row = 0
        for i, kmer in enumerate(self.t2_kmers):
            pad = 0 if i > 0 else 8
            config_frame.grid_rowconfigure(i, weight=1)

            ctk.CTkLabel(config_frame, text=f"{kmer}: ").grid(row=i, column=0, padx=(10, 0), pady=(pad, 0), sticky="w")

            var = ctk.DoubleVar(value=0.0)
            self.k_var_dict[kmer] = var

            (ctk.CTkSlider(config_frame, from_=-3, to=3, variable=var, width=150, height=14).
             grid(row=i, column=1, padx=(10, 0), pady=(pad, 0), sticky="ew"))

            self.k_value_label_dict[kmer] = ctk.CTkLabel(config_frame, text="0.0000", width=60)
            self.k_value_label_dict[kmer].grid(row=i, column=2, padx=(10, 10), pady=(pad, 0), sticky="w")
            # Bind the slider to update the label
            try:
                var.trace_add("write", self.update_all_k_labels)
            except AttributeError:
                var.trace("w", self.update_all_k_labels)
            last_row = i + 1
        self.update_all_k_labels()

        # Sequence length
        seq_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        seq_frame.grid(row=last_row, column=0, columnspan=3, padx=(10, 10), pady=(5, 0), sticky="w")
        ctk.CTkLabel(seq_frame, text="Sequence length:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        segment_entry = ctk.CTkEntry(seq_frame, textvariable=self.t2_seq_len, width=150)
        segment_entry.grid(row=0, column=1, sticky="w")
        segment_entry.bind('<FocusOut>', self._sequence_length_change)
        segment_entry.bind('<Key-Return>', self._sequence_length_change)
        (ctk.CTkLabel(seq_frame, text="Length after generate:", text_color=COLORS["TEXT_DISABLE_COLOR"],
                      font=('Cambria', 10)).grid(row=1, column=0, sticky="w", padx=(0, 5), pady=(0, 0)))
        if self.t2_seq_len_real is not None and self.t2_seq_len_real.winfo_exists():
            text = self.t2_seq_len_real.cget("text")
        else:
            text = ""
        self.t2_seq_len_real = ctk.CTkLabel(seq_frame, text=text, font=('Cambria', 10),
                                            text_color=COLORS["TEXT_DISABLE_COLOR"], anchor="w")
        self.t2_seq_len_real.grid(row=1, column=1, sticky="ew", padx=(5, 5), pady=(0, 0))
        self.t2_seq_len_real.grid_propagate(False)

        # buttons (reset + generate)
        btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        btn_frame.grid(row=last_row + 1, column=0, columnspan=3, padx=10, pady=(10, 10), sticky="ew")
        ctk.CTkButton(btn_frame, text="Reset", command=self.t2_reset_logits).pack(side="left")
        ctk.CTkButton(btn_frame, text="Generate", command=lambda: self.generate_sequence("t2")).pack(side="right")

        # ------------------------- Right panel -------------------------
        self.t2_frame = ctk.CTkFrame(tab, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                     fg_color="white")
        self.t2_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.t2_frame.grid_columnconfigure(0, weight=1)
        self.t2_frame.grid_rowconfigure(0, weight=1)
        self.t2_frame.grid_propagate(False)

        self.t2_placeholder_label = ctk.CTkLabel(master=self.t2_frame, text="Plot Area", text_color="black")
        self.t2_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")

        if getattr(self, "t2_fig", None) is not None:
            self.t2_canvas = FigureCanvasTkAgg(self.t2_fig, master=self.t2_frame)
            self.t2_canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t2_canvas.draw()

            if getattr(self, "t2_save_btn", None) is not None and self.t2_save_btn.winfo_exists():
                try:
                    self.t2_save_btn.destroy()
                except Exception:
                    pass
            self.t2_save_btn = ctk.CTkButton(master=self.t2_frame, text="💾", width=30, height=30,
                                             fg_color=COLORS["BORDER_COLOR"], hover_color=COLORS["FRAME_HOVER_COLOR"],
                                             command=partial(self._save_figure, "t2_fig"))
            self.t2_save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)

        # Save
        (ctk.CTkButton(tab, text="Save Sequence", command=lambda: self.save_sequence("t2"))
         .grid(row=1, column=1, padx=5, pady=5))

    def update_all_k_labels(self, *args):
        # collect logits from all sliders
        logits = np.array([self.k_var_dict[k].get() for k in self.t2_kmers], dtype=float)
        # stable softmax
        logits -= logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        # update every label
        for k, p in zip(self.t2_kmers, probs):
            self.k_value_label_dict[k].configure(text=f"{p:.4f}")

    def t2_reset_logits(self):
        for k in self.t2_kmers:
            self.k_var_dict[k].set(0.0)

    # ------------------------------------------------------------------
    # Tab 3: k-mers (logits)
    # ------------------------------------------------------------------
    def _build_kmer_tab(self, tab):
        tab.grid_rowconfigure(0, weight=50)
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=2)
        tab.grid_columnconfigure(1, weight=1)
        tab.grid_propagate(False)

        config_frame = ctk.CTkFrame(tab, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
        config_frame.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_columnconfigure(1, weight=1)
        for i in range(7):
            config_frame.grid_rowconfigure(i, weight=1)
        config_frame.grid_propagate(False)

        # k combobox
        ctk.CTkLabel(config_frame, text="k-mer length:").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        (ctk.CTkComboBox(config_frame, values=KMERS_SYNTH, state="readonly", variable=self.t3_k_var, width=80,
                         command=self.t3_set_kmers_event).grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(10, 0)))

        # sequence length
        ctk.CTkLabel(config_frame, text="Sequence length:").grid(row=1, column=0, sticky="w", padx=10, pady=(10, 0))
        segment_entry = ctk.CTkEntry(config_frame, textvariable=self.t3_seq_len)
        segment_entry.grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ew")
        segment_entry.bind('<FocusOut>', self._sequence_length_change)
        segment_entry.bind('<Key-Return>', self._sequence_length_change)
        (ctk.CTkLabel(config_frame, text="Length after generate:", text_color=COLORS["TEXT_DISABLE_COLOR"],
                      font=('Cambria', 10)).grid(row=2, column=0, sticky="w", padx=10, pady=(0, 0)))
        if self.t3_seq_len_real is not None and self.t3_seq_len_real.winfo_exists():
            text = self.t3_seq_len_real.cget("text")
        else:
            text = ""
        self.t3_seq_len_real = ctk.CTkLabel(config_frame, text=text, font=('Cambria', 10),
                                            text_color=COLORS["TEXT_DISABLE_COLOR"], anchor="w")
        self.t3_seq_len_real.grid(row=2, column=1, sticky="ew", padx=(5, 5), pady=(0, 0))
        self.t3_seq_len_real.grid_propagate(False)

        # k-mer entry + slider + save checkmark
        entry_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        entry_frame.grid(row=3, column=0, columnspan=2, padx=10, pady=(5, 5), sticky="ew")
        entry_frame.grid_columnconfigure(0, weight=0)
        entry_frame.grid_columnconfigure(1, weight=1)
        entry_frame.grid_columnconfigure(2, weight=0)
        entry_frame.grid_columnconfigure(3, weight=0)

        ctk.CTkLabel(entry_frame, text="k-mer").grid(row=0, column=0, sticky="w")

        self.t3_kmer_entry = ctk.CTkEntry(entry_frame, placeholder_text="e.g., ACAC", width=80)
        self.t3_kmer_entry.grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.t3_kmer_entry.bind("<Return>", lambda _e: self.t3_load_kmer_into_slider())

        self.t3_kmer_slider = ctk.CTkSlider(entry_frame, from_=-3.0, to=3.0, variable=self.t3_slider_var)
        self.t3_kmer_slider.grid(row=1, column=1, sticky="ew", pady=(0, 5))
        self.t3_kmer_slider.configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])

        self.t3_kmer_label = ctk.CTkLabel(entry_frame, text=f"{self.t3_slider_var.get():.4f}")
        self.t3_kmer_label.grid(row=1, column=2, sticky="w", pady=(0, 5))
        self.t3_slider_var.trace_add("write", self.t3_update_kmer_label)

        (ctk.CTkButton(entry_frame, text="✓", width=10, command=self.t3_refresh_summary)
         .grid(row=1, column=3, sticky="w", padx=(10, 0), pady=(0, 5)))

        # summary textbox
        ctk.CTkLabel(config_frame, text="Summary").grid(row=4, column=0, padx=10, pady=(5, 0), sticky="w")
        self.t3_summary_box = ctk.CTkTextbox(config_frame, height=160)
        self.t3_summary_box.grid(row=5, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

        self.t3_refresh_summary()

        # buttons (reset + generate)
        btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        btn_frame.grid(row=6, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        ctk.CTkButton(btn_frame, text="Reset", command=self.t3_reset_logits).pack(side="left")
        ctk.CTkButton(btn_frame, text="Generate", command=lambda: self.generate_sequence("t3")).pack(side="right")

        # ------------------------- Right panel -------------------------
        self.t3_frame = ctk.CTkFrame(tab, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                     fg_color="white")
        self.t3_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        self.t3_frame.grid_columnconfigure(0, weight=1)
        self.t3_frame.grid_rowconfigure(0, weight=1)
        self.t3_frame.grid_propagate(False)

        self.t3_placeholder_label = ctk.CTkLabel(master=self.t3_frame, text="Plot Area", text_color="black")
        self.t3_placeholder_label.place(relx=0.5, rely=0.01, anchor="n")

        if getattr(self, "t3_fig", None) is not None:
            self.t3_canvas = FigureCanvasTkAgg(self.t3_fig, master=self.t3_frame)
            self.t3_canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t3_canvas.draw()

            if getattr(self, "t3_save_btn", None) is not None and self.t3_save_btn.winfo_exists():
                try:
                    self.t3_save_btn.destroy()
                except Exception:
                    pass
            self.t3_save_btn = ctk.CTkButton(master=self.t3_frame, text="💾", width=30, height=30,
                                             fg_color=COLORS["BORDER_COLOR"], hover_color=COLORS["FRAME_HOVER_COLOR"],
                                             command=partial(self._save_figure, "t3_fig"))
            self.t3_save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)

        # Save
        (ctk.CTkButton(tab, text="Save Sequence", command=lambda: self.save_sequence("t3"))
         .grid(row=1, column=1, padx=5, pady=5))

    def t3_set_kmers_event(self, *args):
        self.t3_kmers = generate_kmers(self.t3_k_var.get())
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.t3_kmers)}
        self.logits = np.zeros(len(self.t3_kmers), dtype=float)
        self.t3_current_kmer = None
        # Reset the entry + slider, and refresh the summary box
        self.t3_kmer_entry.delete(0, tkinter.END)
        self.t3_slider_var.set(0.0)
        self.t3_kmer_slider.configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
        self.t3_refresh_summary()

    def t3_load_kmer_into_slider(self, *args):
        kmer = self.t3_kmer_entry.get().strip().upper()
        if len(kmer) != self.t3_k_var.get() or any(c not in 'ACGT' for c in kmer):
            messagebox.showerror("Invalid k-mer",
                                 f"Please enter a length-{self.t3_k_var.get()} k-mer using only A,C,G,T.")
            self.t3_current_kmer = None
            self.t3_kmer_entry.delete(0, tkinter.END)
            self.t3_slider_var.set(0.0)
            self.t3_kmer_slider.configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
            return
        self.t3_current_kmer = kmer
        idx = self.kmer_to_idx[self.t3_current_kmer]
        self.t3_slider_var.set(float(self.logits[idx]))
        self.t3_kmer_slider.configure(state="normal", button_color=COLORS["BTN_COLOR"])

    def t3_softmax(self):
        x = self.logits - self.logits.max()
        e = np.exp(x)
        return e / e.sum()

    def t3_update_kmer_label(self, *args):
        if self.t3_current_kmer is None:
            self.t3_kmer_label.configure(text=f"{self.t3_slider_var.get():.4f}")
            return
        idx = self.kmer_to_idx[self.t3_current_kmer]
        self.logits[idx] = float(self.t3_slider_var.get())
        self.t3_kmer_label.configure(text=f"{self.t3_softmax()[idx]:.4f}")

    def t3_reset_logits(self, *args):
        self.logits[:] = 0.0
        if self.t3_current_kmer:
            idx = self.kmer_to_idx[self.t3_current_kmer]
            self.t3_slider_var.set(float(self.logits[idx]))
        self.t3_refresh_summary()

    def t3_refresh_summary(self):
        if self.t3_kmer_entry.get().strip().upper() == "":
            self.t3_current_kmer = None
            self.t3_slider_var.set(0.0)
            self.t3_kmer_slider.configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
        p = self.t3_softmax()
        # summarize: if all equal
        if np.allclose(self.logits, self.logits[0]):
            txt = (f"All {len(self.t3_kmers)} k-mers have equal weight.\n"
                   f"Each probability = {1.0 / len(self.t3_kmers):.4f}\n")
        else:
            # list only k-mers whose logits differ from the median by > 1e-9 (assigned)
            med = np.median(self.logits)
            assigned = [(km, p[self.kmer_to_idx[km]]) for km in self.t3_kmers
                        if abs(self.logits[self.kmer_to_idx[km]] - med) > 1e-9]
            assigned_sorted = sorted(assigned, key=lambda x: -x[1])
            others = 1.0 - sum(prob for _, prob in assigned_sorted)
            n_others = len(self.t3_kmers) - len(assigned_sorted)
            avg_other = (others / n_others) if n_others > 0 else 0.0

            lines = [f"Assigned k-mers ({len(assigned_sorted)}):"]
            lines += [f"  {km}: {prob:.8f}" for km, prob in assigned_sorted]
            if n_others > 0:
                lines += [f"Others ({n_others}): avg ≈ {avg_other:.8f} (sum ≈ {others:.8f})"]
            txt = "\n".join(lines)

        self.t3_summary_box.configure(state="normal")
        self.t3_summary_box.delete("1.0", tkinter.END)
        self.t3_summary_box.insert("1.0", txt)
        self.t3_summary_box.configure(state="disabled")

    # ------------------------------------------------------------------
    # General functions for all tabs
    # ------------------------------------------------------------------
    def _sequence_length_change(self, tab, *args):
        if tab == "t1":
            sequence = self.t1_seq_len
        elif tab == "t2":
            sequence = self.t2_seq_len
        elif tab == "t3":
            sequence = self.t3_seq_len
        else:
            return
        val = sequence.get().replace(",", "").strip()

        if not val.isdigit() or int(val) <= 0:
            messagebox.showerror("Error", "Sequence length must be a positive integer.")

            # revert to last valid value (or a default)
            last = getattr(self, "t1_last_valid_seq_len", None)
            if last is None:
                last = 1000
            sequence.set(f"{last:,}")
            return

        self.t1_last_valid_seq_len = int(val)
        sequence.set(f"{int(val):,}")

    def generate_sequence(self, frame_num):
        if frame_num == "t1":
            k = self.t1_k_var.get()
            seq_len = int(self.t1_seq_len.get().replace(",", "").strip())
            r = self.t1_r_var.get()
            self.t1_generated_sequence, _, _ = generate_dna_sequence(k, seq_len, target_entropy=r * (2 * k))
            sequence = self.t1_generated_sequence
            # Fix sequence length in the input box
            self.t1_seq_len_real.configure(text=f"{len(self.t1_generated_sequence):,}")
            kmer_counts_dict = {'diff_message': "", 'adjust_message': ""}
        elif frame_num == "t2":
            k = 2
            seq_len = int(self.t2_seq_len.get().replace(",", "").strip())
            slider_values = [self.k_var_dict[kmer].get() for kmer in self.t2_kmers]
            self.t2_generated_sequence, kmer_counts_dict, _ = generate_dna_sequence(k, seq_len, p_input=slider_values)
            sequence = self.t2_generated_sequence

            # Update sequence length and k-mer counts in the labels and the scales
            self.t2_seq_len_real.configure(text=f"{len(sequence):,}")

            probs = []
            for kmer in self.t2_kmers:
                c = kmer_counts_dict["counts"].get(kmer, 0)
                probs.append(c / len(sequence))
            probs = np.array(probs, dtype=float)

            eps = 1e-12
            logits = np.log(np.clip(probs, eps, 1.0) * len(self.t2_kmers))
            logits = np.clip(logits, -3.0, 3.0)

            for kmer, logit in zip(self.t2_kmers, logits):
                self.k_var_dict[kmer].set(float(logit))

            self.update_all_k_labels()
        elif frame_num == "t3":
            k = self.t3_k_var.get()
            seq_len = int(self.t3_seq_len.get().replace(",", "").strip())
            p_input = self.t3_softmax()
            self.t3_generated_sequence, kmer_counts_dict, _ = generate_dna_sequence(k, seq_len, p_input=p_input)
            sequence = self.t3_generated_sequence

            # Update sequence length
            self.t3_seq_len_real.configure(text=f"{len(sequence):,}")

            probs = np.array([kmer_counts_dict["counts"].get(km, 0) / len(sequence) for km in self.t3_kmers],
                             dtype=float)

            eps = 1e-12
            # logits that reproduce probs under softmax (up to clipping)
            logits = np.log(np.clip(probs, eps, 1.0) * len(self.t3_kmers))
            logits = np.clip(logits, -3.0, 3.0)
            self.logits[:] = logits

            # update the currently loaded kmer slider, if any
            if self.t3_current_kmer is not None:
                idx = self.kmer_to_idx[self.t3_current_kmer]
                self.t3_slider_var.set(float(self.logits[idx]))
                self.t3_kmer_slider.configure(state="normal", button_color=COLORS["BTN_COLOR"])
            else:
                self.t3_slider_var.set(0.0)
                self.t3_kmer_slider.configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])

            # refresh summary (and prob label updates via trace)
            self.t3_refresh_summary()
        else:
            return
        # Show warning message in kmer_counts_dict
        if kmer_counts_dict['adjust_message'] != "":
            messagebox.showwarning("Generation Warning", kmer_counts_dict['adjust_message'])
        elif kmer_counts_dict['diff_message'] != "":
            messagebox.showwarning("Generation Warning", kmer_counts_dict['diff_message'])
        fcgr = CGR(sequence, k).get_fcgr()

        # Prepare plot area
        frame = getattr(self, f"{frame_num}_frame")
        fig_name = f"{frame_num}_fig"
        canvas_name = f"{frame_num}_canvas"
        save_btn_name = f"{frame_num}_save_btn"
        placeholder_name = f"{frame_num}_placeholder_label"

        # --- 1) Figure setup ---
        fig = getattr(self, fig_name, None)
        if fig is None:
            fig = plt.Figure(dpi=120)
            setattr(self, fig_name, fig)

        # --- 2) Canvas setup ---
        canvas = getattr(self, canvas_name, None)
        if canvas is None or not canvas.get_tk_widget().winfo_exists() or canvas.get_tk_widget().master is not frame:
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            setattr(self, canvas_name, canvas)
            # --- Save button ---
            save_btn = getattr(self, save_btn_name, None)
            if save_btn is None or not save_btn.winfo_exists() or save_btn.master is not frame:
                save_btn = ctk.CTkButton(master=frame, text="💾", width=30, height=30, fg_color=COLORS["BORDER_COLOR"],
                                         hover_color=COLORS["FRAME_HOVER_COLOR"],
                                         command=partial(self._save_figure, fig_name), )
            save_btn.place(relx=0.01, rely=0.99, anchor="sw", x=0)
            setattr(self, save_btn_name, save_btn)

        # --- 3) Clear figure and re-plot ---
        fig.clear()
        # Draw FCGR
        ax = fig.add_subplot(111)
        f = CGR.normalize(fcgr)
        self.parent.fcgr_normalizer.fit([f], ks=[self.t3_k_var.get()])
        V = self.parent.fcgr_normalizer.transform01(f, k=self.t3_k_var.get(), L=len(sequence))
        img_uint8 = FCGRNormalizer.to_uint8_from_01(V, white_is_high=True)
        img = Image.fromarray(img_uint8)
        ax.imshow(img, cmap="gray", origin="upper")
        ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        corner_labels = [("A", (0.00, -0.01), (-0.05, -0.05), "right", "top"),
                         ("C", (0.00, 0.99), (-0.05, +0.05), "right", "bottom"),
                         ("T", (1.00, -0.01), (+0.05, -0.05), "left", "top"),
                         ("G", (1.00, 0.99), (+0.05, +0.05), "left", "bottom")]
        for text, xy, offset, ha, va in corner_labels:
            ax.annotate(text, xy=xy, xycoords="axes fraction", xytext=offset, textcoords="offset points",
                        ha=ha, va=va, fontsize=10, color="black", clip_on=False)

        # --- 4) Hide placeholder ---
        placeholder = getattr(self, placeholder_name, None)
        if placeholder is not None and placeholder.winfo_exists():
            try:
                placeholder.place_forget()
            except Exception:
                pass

        # --- 5) Redraw canvas ---
        canvas.draw()

    def _save_figure(self, fig_attr):
        fig = getattr(self, fig_attr, None)
        if fig is None:
            return messagebox.showerror("Error", "No figure to save. Please plot first.")

        file_path = fd.asksaveasfilename(defaultextension=".png", title="Save figure",
                                         filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"),
                                                    ("SVG Image", "*.svg"), ("All Files", "*.*")])
        if not file_path:
            return  # user cancelled

        try:
            fig.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception:
            messagebox.showerror("Error", "Could not save figure.")

    def show(self):
        self.deiconify()
        self.lift()
        self.focus_force()
        self.grab_set()

    def _on_user_close(self):
        # hide instead of destroy -> keeps all vars + widget state
        self.grab_release()
        self.withdraw()

    def save_sequence(self, tab):
        if tab == "t1":
            seq = getattr(self, "t1_generated_sequence", "")
            k = self.t1_k_var.get()
        elif tab == "t2":
            seq = getattr(self, "t2_generated_sequence", "")
            k = 2
        elif tab == "t3":
            seq = getattr(self, "t3_generated_sequence", "")
            k = self.t3_k_var.get()
        else:
            return
        if not seq:
            messagebox.showerror("Error", "No sequence generated.")
            return

        default_name = f"Synthetic_k{k}_{len(seq)}bp"
        dialog = CTkAskString(self, initial_value=default_name)

        self.wait_window(dialog)
        name = dialog.result

        if name is None:  # user pressed Cancel
            return

        name = name.strip()
        if not name:
            messagebox.showwarning("Warning", "Name cannot be empty.")
            return

        if callable(self.on_save):
            self.on_save(seq, name)

        self.grab_release()
        self.withdraw()


class CTkAskString(ctk.CTkToplevel):
    def __init__(self, parent, initial_value=""):
        super().__init__(parent)
        self.parent = parent
        self.result = None
        self.title("Sequence Name")

        # make it modal & on top of parent
        self.transient(self.parent)
        self.grab_set()
        self.focus_set()
        # size/center relative to parent (not the screen)
        self.parent.update_idletasks()
        w = int(self.parent.winfo_width() * 0.4)
        h = int(self.parent.winfo_height() * 0.3)
        x = self.parent.winfo_rootx() + (self.parent.winfo_width() - w) // 2
        y = self.parent.winfo_rooty() + (self.parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")
        self.resizable(False, False)

        ctk.CTkLabel(self, text="Enter a name for the synthetic sequence:").pack(pady=(20, 10))

        self.entry = ctk.CTkEntry(self, width=250)
        self.entry.pack()
        self.entry.insert(0, initial_value)
        self.entry.focus()

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(btn_frame, text="OK", width=100, command=self._on_ok).pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="Cancel", width=100, command=self.destroy).pack(side="left", padx=10)

        self.bind("<Return>", lambda e: self._on_ok())

    def _on_ok(self):
        self.result = self.entry.get().strip()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()

    # pyinstaller --onefile --windowed --name "MyApp" --add-data "assets:assets" GUI.py
    # open dist/MyApp.app or ./dist/MyApp.app/Contents/MacOS/MyApp

    # pyinstaller --onefile --windowed --name "MyApp" --icon=icon.icns GUI.py

    # def load_matplotlib():
    #     import matplotlib.pyplot as plt
    #     return plt

    # xattr - cr MyApp.app
    # chmod + x MyApp.app/Contents/MacOS/MyApp
