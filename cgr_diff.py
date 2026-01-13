import os
import pickle
import sys
import threading
import tkinter
import tkinter.messagebox
from collections import Counter
from functools import partial
import random

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter.filedialog as fd

import numpy as np
from Bio import Entrez
from PIL import Image
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator, FuncFormatter
from sklearn.manifold import MDS
from matplotlib.backends._backend_tk import NavigationToolbar2Tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib import pyplot as plt, colors

# Optional: adds hover tooltips for Matplotlib artists
try:
    import mplcursors  # type: ignore
except Exception:
    mplcursors = None

from chaos_game_representation import CGR
from distances.distance_metrics import get_dist
from sequence_generation.sampling import generate_kmers
from sequence_generation.sequence_generation import generate_dna_sequence

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"
HEADER_FONT = ('Cambria', 15)
HEADER_FONT_BOLD = ('Cambria', 15, 'bold')

# main colors in the theme
COLORS = dict(
    BTN_COLOR=ctk.ThemeManager.theme["CTkButton"]["fg_color"],
    DISABLED_BTN_COLOR="#888888",
    TEXT_NORMAL_COLOR=ctk.ThemeManager.theme["CTkButton"]["text_color"],
    TEXT_DISABLE_COLOR="#707370",  # BTN_THEME.get("text_color_disabled", TEXT_NORMAL_COLOR)
    FRAME_COLOR="#707370",
    FRAME_NORMAL_COLOR="#2B2B2B",
    FRAME_HOVER_COLOR="#444444",
    BORDER_COLOR="#333333",
    LIGHT_FRAME_COLOR="#DBDBDB", )
KMERS = [str(i) for i in range(1, 10)]
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

    def invalidate(self):
        self.seq_name.set("")
        self.seq = ""
        self.start_seq.set(0)
        self.end_seq.set(0)


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
        self.display_content_frame = None  # frame in the right panel to hold analysis sub-frames

        self.k_var = ctk.IntVar(value=6)  # k-mer selection variable
        self.dist_metric = tkinter.StringVar(value="DSSIM")  # distance metric selection variable

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
        self.t3_segment_size = tkinter.StringVar(value="500,000")  # 500,000 for test
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
        self._t3_progress = 0.0
        self.t3_progress_bar = None

        self.t3_3d_display_frame = None
        self.t3_3d_placeholder_label = None
        self.t3_3d_fig = None
        self.t3_3d_canvas = None
        self._t3_mds_drawn = False
        self.t3_mds_toolbar = None

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
        self._create_top_navbar()
        self._build_main_content()

        self._upload_files(hard_coded=True)

    def _toggle_theme(self):
        new_mode = "Light" if self.appearance.get() == "Dark" else "Dark"
        self.appearance.set(new_mode)
        ctk.set_appearance_mode(new_mode)
        # update icon
        if new_mode == "Dark":
            self.theme_button.configure(text="☀️")
        else:
            self.theme_button.configure(text="🌙")

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
        parent.grid_rowconfigure(0, weight=1)

        # ---------- Left panel ----------
        config_frame = ctk.CTkFrame(parent, corner_radius=8, border_color=COLORS["BORDER_COLOR"], border_width=1)
        config_frame.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        config_frame.grid_columnconfigure(0, weight=1)
        config_frame.grid_rowconfigure(1, weight=1)  # row 1 is list_frame
        config_frame.grid_propagate(False)
        # ---------- Right panel (scrollable, only vertical) ----------
        # TODO: not scrollable with mouse wheel
        display_frame = ctk.CTkScrollableFrame(parent, border_width=1, corner_radius=8,
                                               border_color=COLORS["BORDER_COLOR"], label_text=" Display Area ")
        display_frame.grid(row=0, column=1, padx=(0, 5), pady=(5, 5), sticky="nsew")
        display_frame.grid_columnconfigure(0, weight=1)
        self.display_content_frame = display_frame
        # self._enable_display_scroll_wheel()

        # ---------- Designing the config frame (F1) ----------
        # top buttons
        top_btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        top_btn_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        top_btn_frame.grid_columnconfigure((0, 1), weight=1)

        search_btn = ctk.CTkButton(top_btn_frame, text="Search and Download", corner_radius=8, height=35,
                                   font=HEADER_FONT, text_color="white", )
        search_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        upload_btn = ctk.CTkButton(top_btn_frame, text="Upload", corner_radius=8, height=35, font=HEADER_FONT,
                                   text_color="white", command=self._upload_files)
        upload_btn.grid(row=1, column=0, sticky="ew")

        generate_btn = ctk.CTkButton(top_btn_frame, text="Generate", corner_radius=8, height=35, font=HEADER_FONT,
                                     text_color="white", )
        generate_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        # list of genomes
        list_frame = ctk.CTkFrame(config_frame, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"])
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
        # TODO: change the color background of the canvas when toggle theme
        canvas = tkinter.Canvas(scroll_container, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        # vertical scrollbar
        v_scroll = ctk.CTkScrollbar(scroll_container, orientation="vertical", command=canvas.yview, )
        v_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        # horizontal scrollbar
        h_scroll = ctk.CTkScrollbar(scroll_container, orientation="horizontal", command=canvas.xview, )
        h_scroll.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # inner frame that actually holds the file entries
        self.uploaded_seq_lists_frame = ctk.CTkFrame(canvas, fg_color="transparent", )
        # TODO: change the color background of the canvas
        canvas.create_window((0, 0), window=self.uploaded_seq_lists_frame, anchor="nw")

        def _on_inner_configure(event):
            # update scroll region to fit inner frame (both width and height)
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.uploaded_seq_lists_frame.bind("<Configure>", _on_inner_configure)
        self._refresh_uploaded_file_list()

        # bottom buttons
        bottom_btn_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        bottom_btn_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        bottom_btn_frame.grid_columnconfigure((0, 1), weight=1)

        remove_btn = ctk.CTkButton(bottom_btn_frame, text="Remove", corner_radius=8, height=35, font=HEADER_FONT,
                                   command=self._remove_selected_file, )
        remove_btn.grid(row=1, column=0, sticky="ew")

        run_btn = ctk.CTkButton(bottom_btn_frame, text="Run Analysis", corner_radius=8, height=35, font=HEADER_FONT,
                                command=self._run_analysis_selected_file, )
        run_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

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
            self.t2_save_btn.place(relx=0.01, rely=0.99, anchor="sw")

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
        self.t3_ds['2'].start_txt = tkinter.StringVar(value="200,000")
        self.t3_ds['2'].end_txt = tkinter.StringVar(value="700,000")
        self.t3_start_label = ctk.CTkLabel(seq_frame, text="Start: ", font=HEADER_FONT, text_color=text_color)
        self.t3_start_label.grid(row=4, column=0, sticky="w", padx=(10, 0), pady=(10, 0))
        self.t3_start_entry = ctk.CTkEntry(seq_frame, textvariable=self.t3_ds['2'].start_txt)
        self.t3_start_entry.bind('<FocusOut>', partial(self.t3_entry_change, "start"))
        self.t3_start_entry.bind('<Key-Return>', partial(self.t3_entry_change, "start"))
        self.t3_start_entry.configure(state=state)
        self.t3_start_entry.grid(row=5, column=0, sticky="ew", padx=(10, 0), pady=(0, 10))
        if self.t3_end_label is not None and self.t3_end_label.winfo_exists():
            text_color = self.t3_end_label.cget("text_color")
            state = self.t3_end_label.cget("state")
        else:
            text_color = COLORS["TEXT_DISABLE_COLOR"]
            state = "disable"
        self.t3_end_label = ctk.CTkLabel(seq_frame, text="End: ", font=HEADER_FONT, text_color=text_color)
        self.t3_end_label.grid(row=4, column=1, sticky="w", padx=(10, 0), pady=(10, 0))
        self.t3_end_entry = ctk.CTkEntry(seq_frame, textvariable=self.t3_ds['2'].end_txt)
        self.t3_end_entry.bind('<FocusOut>', partial(self.t3_entry_change, "end"))
        self.t3_end_entry.bind('<Key-Return>', partial(self.t3_entry_change, "end"))
        self.t3_end_entry.configure(state=state)
        self.t3_end_entry.grid(row=5, column=1, sticky="ew", padx=(10, 10), pady=(0, 10))

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
        (ctk.CTkComboBox(kmer_frame, values=PLOT_TYPES, state="readonly", variable=self.t3_plot_type)
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
        # TODO: add a label to show progress percentage or status

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
            # Create a new canvas for the existing figure, attached to the new frame
            self.t3_3d_canvas = FigureCanvasTkAgg(self.t3_3d_fig, master=self.t3_3d_display_frame)
            widget = self.t3_3d_canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
            self.t3_3d_canvas.draw()

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
            self.t3_fcgr_save_btn.place(relx=0.01, rely=0.99, anchor="sw")

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
            self.t3_plot_save_btn.place(relx=0.01, rely=0.99, anchor="sw")

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
    def _upload_files(self, hard_coded=False):
        if hard_coded:
            file_paths = [
                "Data/Human/chromosomes/chr21.fna",
                "Data/Escherichia coli/chromosomes/chrA.fna"
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

        self._refresh_uploaded_file_list()

    def _refresh_uploaded_file_list(self):
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
            self.file_names.append(fname)

            # card for each file
            card = ctk.CTkFrame(self.uploaded_seq_lists_frame, fg_color="transparent")
            card.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            # make everything inside the card clickable
            def _make_on_click(index):
                def _on_click(event=None):
                    return self._set_selected_uploaded(index)

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
                self._set_selected_uploaded(self.selected_file_index)
            else:
                self.selected_file_index = None

    def _set_selected_uploaded(self, index: int):
        self.selected_file_index = index

        for i, card in enumerate(self.file_cards):
            if i == index:
                # SELECTED STYLE
                card.configure(fg_color=COLORS["BTN_COLOR"], corner_radius=0)
                for child in card.winfo_children():
                    if child.grid_info().get("row") == 0:
                        child.configure(fg_color=COLORS["BTN_COLOR"], text_color=COLORS["TEXT_NORMAL_COLOR"])
                    else:
                        child.configure(fg_color=COLORS["BTN_COLOR"], text_color=COLORS["TEXT_NORMAL_COLOR"])
            else:
                card.configure(fg_color="transparent")
                for child in card.winfo_children():
                    if child.grid_info().get("row") == 0:
                        child.configure(fg_color="transparent", text_color=COLORS["TEXT_NORMAL_COLOR"])
                    else:
                        child.configure(fg_color="transparent", text_color=COLORS["TEXT_DISABLE_COLOR"])

    def _remove_selected_file(self):
        if self.selected_file_index is None:
            messagebox.showinfo("No selection", "Please select a file to remove.")
            return

        removed_path = self.uploaded_files.pop(self.selected_file_index)  # remove from list
        self.selected_file_index = None  # reset selection
        self._refresh_uploaded_file_list()  # refresh GUI

    def _run_analysis_selected_file(self):
        if self.selected_file_index is None:
            messagebox.showinfo("No selection", "Please select a file to analyze.")
            return

        path = self.uploaded_files[self.selected_file_index]
        if not os.path.isfile(path):
            messagebox.showinfo("Error", "The selected file does not exist.")
            return

        name, seq = self._read_fasta(path)
        if not seq:
            messagebox.showinfo("Error", "The selected FASTA file contains no sequence data.")
            return

        # ----- build / rebuild layout in the scrollable right panel -----
        # clear old content
        for child in self.display_content_frame.winfo_children():
            child.destroy()

        # grid configuration for display content:
        self.display_content_frame.grid_rowconfigure(0, weight=1)
        self.display_content_frame.grid_rowconfigure(1, weight=1)
        self.display_content_frame.grid_rowconfigure(2, weight=1)
        self.display_content_frame.grid_columnconfigure(0, weight=1)  # left
        self.display_content_frame.grid_columnconfigure(1, weight=2)  # right (larger)

        # top histogram frame (full width)
        self.hist_frame = ctk.CTkFrame(self.display_content_frame, corner_radius=8, border_width=1, height=300,
                                       fg_color=COLORS["LIGHT_FRAME_COLOR"], border_color=COLORS["BORDER_COLOR"])
        self.hist_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 5), )
        self.hist_frame.grid_rowconfigure(0, weight=1)
        self.hist_frame.grid_columnconfigure(0, weight=1)
        self.hist_frame.grid_propagate(False)

        # second frame (full width, below histogram)
        self.middle_frame = ctk.CTkFrame(self.display_content_frame, corner_radius=8, border_width=1, height=300)
        self.middle_frame.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=5, )
        self.middle_frame.grid_propagate(False)

        # bottom-left frame (smaller)
        self.bottom_left_frame = ctk.CTkFrame(self.display_content_frame, corner_radius=8, border_width=1, height=400)
        self.bottom_left_frame.grid(row=2, column=0, sticky="nsew", padx=(10, 5), pady=(5, 10), )
        self.bottom_left_frame.grid_propagate(False)

        # bottom-right frame (larger)
        self.bottom_right_frame = ctk.CTkFrame(self.display_content_frame, corner_radius=8, border_width=1, height=400)
        self.bottom_right_frame.grid(row=2, column=1, sticky="nsew", padx=(5, 10), pady=(5, 10), )
        self.bottom_right_frame.grid_propagate(False)

        # ---- the analysis goes here ----
        # TODO: implement the analysis and plotting functions :
        #  1) k-mer analysis
        #  2) over representative and under representative analysis
        #  3) FCGR plot
        #  4) 3D FCGR plot
        # 3-mer frequency analysis histogram
        k = 3
        self._plot_kmer_histogram(self.hist_frame, seq, k)

    def _plot_kmer_histogram(self, plot_frame, seq, k):
        # Clear frame
        for child in plot_frame.winfo_children():
            child.destroy()

        # Data
        counts = self._count_kmers(seq, k)
        labels = self._labels_kmers(k)
        total = int(counts.sum())

        subtitle = f"length: {len(seq):,}  |  valid 3-mers: {total:,}"

        # Ensure frame has a real size
        plot_frame.update_idletasks()
        w = max(300, plot_frame.winfo_width())
        h = max(200, plot_frame.winfo_height())

        dpi = 120
        fig = plt.Figure(figsize=(w / dpi, h / dpi), dpi=dpi)
        bg = plot_frame.cget("fg_color")
        fig.patch.set_facecolor(bg)
        ax = fig.add_subplot(111)

        # Plot
        x = np.arange(64)
        bars = ax.bar(x, counts, width=0.85)

        fig.subplots_adjust(left=0.07, right=0.995, bottom=0.15, top=0.95)
        fig.text(0.07, 1, subtitle, ha="left", va="top", fontsize=8)

        # X-axis ticks (clean spacing, not glued to y-axis)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=60, ha="center", fontsize=7)
        ax.tick_params(axis="x", pad=2)
        ax.margins(x=0.01)
        ax.set_xlim(-0.8, 63.8)

        # Y-axis: plain numbers, correct formatting, smaller font
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.yaxis.get_offset_text().set_visible(False)

        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(round(v)):,}"))

        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.25)

        # Embed into Tk frame
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        widget = canvas.get_tk_widget()
        plot_frame.grid_rowconfigure(0, weight=1)
        plot_frame.grid_columnconfigure(0, weight=1)
        widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # HOVER (my style)
        if len(bars) > 0:
            default_color = bars[0].get_facecolor()
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
                        bars[hovered["idx"]].set_facecolor(default_color)
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

                # no change
                if found_idx == hovered["idx"]:
                    return

                # restore previous hover color
                if hovered["idx"] is not None:
                    bars[hovered["idx"]].set_facecolor(default_color)

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

            canvas.mpl_connect("motion_notify_event", on_motion)

        canvas.draw()

    @staticmethod
    def _count_kmers(seq, k):
        seq = (seq or "").upper()
        n = len(seq)
        if k <= 0:
            return np.array([], dtype=np.int64)

        m = 4 ** k
        if n < k:
            return np.zeros(m, dtype=np.int64)

        # ASCII -> 0..3 for A,C,G,T; invalid -> -1
        table = np.full(256, -1, dtype=np.int8)
        table[ord("A")] = 0
        table[ord("C")] = 1
        table[ord("G")] = 2
        table[ord("T")] = 3

        s = np.frombuffer(seq.encode("ascii", "ignore"), dtype=np.uint8)
        x = table[s].astype(np.int16)  # -1 invalid
        if x.size < k:
            return np.zeros(m, dtype=np.int64)

        # Valid windows: convolution over valid mask
        valid = (x >= 0).astype(np.int8)
        window_valid = np.convolve(valid, np.ones(k, dtype=np.int8), mode="valid") == k
        if not np.any(window_valid):
            return np.zeros(m, dtype=np.int64)

        # Rolling base-4 code for each window, O(n*k)
        # codes[i] = sum_{j=0..k-1} x[i+j] * 4^(k-1-j)
        pow4 = (4 ** np.arange(k - 1, -1, -1, dtype=np.int64))  # [4^(k-1), ..., 1]
        # Build windows via stride trick to avoid loops
        shape = (x.size - k + 1, k)
        strides = (x.strides[0], x.strides[0])
        windows = np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)

        codes = (windows.astype(np.int64) * pow4).sum(axis=1)
        codes = codes[window_valid]

        counts = np.bincount(codes, minlength=m).astype(np.int64)
        return counts

    @staticmethod
    def _labels_kmers(k):
        if k <= 0:
            return []

        bases = np.array(list("ACGT"))
        m = 4 ** k
        labels = [""] * m

        for code in range(m):
            c = code
            chars = []
            for _ in range(k):
                chars.append(bases[c % 4])
                c //= 4
            labels[code] = "".join(reversed(chars))
        return labels

    # --------------------------------------------------
    # Helper functions for CGR comparator tab
    # --------------------------------------------------
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
            self.t3_start_label.configure(text_color=COLORS["TEXT_DISABLE_COLOR"])
            self.t3_start_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
            self.t3_end_label.configure(text_color=COLORS["TEXT_DISABLE_COLOR"])
            self.t3_end_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
        else:
            # Disable the algo type combobox
            self.t3_rep_type_combobox.configure(state="disabled")
            self.t3_rep_n_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])
            # Enable start and end
            self.t3_start_label.configure(text_color=COLORS["TEXT_NORMAL_COLOR"])
            self.t3_start_entry.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])
            self.t3_end_label.configure(text_color=COLORS["TEXT_NORMAL_COLOR"])
            self.t3_end_entry.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])

    def t3_rep_algo_change_event(self, value):
        if value == "aRepSeg":
            self.t3_rep_number.set("30")
            self.t3_rep_n_entry.configure(state="normal", text_color=COLORS["TEXT_NORMAL_COLOR"])
        elif value == "RepSeg":
            self.t3_rep_number.set("1")
            self.t3_rep_n_entry.configure(state="disable", text_color=COLORS["TEXT_DISABLE_COLOR"])

    def t3_entry_change(self, which, event=None):
        start_raw = self.t3_ds['2'].start_txt.get().strip()
        end_raw = self.t3_ds['2'].end_txt.get().strip()

        # No sequence selected
        if self.t3_ds['2'].seq == '':
            if which == "start":
                self.t3_ds['2'].start_txt.set("")
            else:
                self.t3_ds['2'].end_txt.set("")
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
                self.t3_ds['2'].start_txt.set("")
            else:
                self.t3_ds['2'].end_txt.set("")
            return messagebox.showerror("Error", "Start and end values must be positive integers, "
                                                 "within sequence length.")

        seq_len = len(self.t3_ds['2'].seq)
        # Range validation
        if start < 0 or start > seq_len or end < 0 or end > seq_len or start >= end:
            if which == "start":
                self.t3_ds['2'].start_txt.set("")
            else:
                self.t3_ds['2'].end_txt.set("")
            return messagebox.showerror("Error", "Start and end values must be positive integers, "
                                                 "within sequence length, and start < end.")

        # All good so update stored values
        self.t3_ds['2'].start_seq.set(start)
        self.t3_ds['2'].end_seq.set(end)

        # Normalize formatting with commas so display is consistent
        self.t3_ds['2'].start_txt.set(self._format_int(start))
        self.t3_ds['2'].end_txt.set(self._format_int(end))

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
                self._t3_progress = (i + 3) / (t3_step_length + 2)
                b2 = i * seg_size
                e2 = (i + 1) * seg_size
                cgr2 = CGR(self.t3_ds["1"].seq[b2:e2], self.k_var.get())
                im2 = cgr2.get_fcgr()

                diff = im2 - im1
                dist = get_dist(im1, im2, dist_m=self.dist_metric.get())
                self.t3_cgr_distance_history.append(dist)

                fcgrs_dict[i] = {"fcgr": im2, "b": b2, "e": e2, "seq_len": len(self.t3_ds["2"].seq),
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
        self.t3_progress_bar.set(self._t3_progress)
        if foo_thread_2.is_alive():
            self.after(20, self.t3_check_thread)
        else:
            self.t3_progress_bar.set(1.0)
            self.t3_scale.configure(to=int(len(self.t3_cgr_distance_history) - 1))  # Update the scale range
            self.t3_pic_num.set(0)

            # Display the 3d plot, image, and the chart
            with open(f"{self.temp_output_path}/t3_run/t3_distance_matrix.pkl", 'rb') as handle:
                D = pickle.load(handle)
            # MDS (3d)
            self._t3_mds_drawn = False
            self._draw_panel(frame=self.t3_3d_display_frame, fig_attr="t3_3d_fig",
                             canvas_attr="t3_3d_canvas", save_btn_attr="t3_3d_save_btn",
                             save_command=lambda: self._save_figure("t3_3d_fig"),
                             placeholder_attr="t3_3d_placeholder_label", fcgrs_dict=None, index=0, panel_type="mds",
                             D=D)

            # Display image and chart
            self.after_idle(lambda: self.t3_change_images(0, None))

    def t3_change_images(self, index, value):
        index = round(value) if value is not None else index
        # MDS change color
        self._t3_mds_set_selected(index)
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

        if fig == self.t2_fig:
            ax1, ax2, ax3 = fig.subplots(1, 3)

            scale_1, scaling_1 = self._scaling(fcgrs["1"]["seq_len"])
            b1 = fcgrs["1"]["b"]
            e1 = fcgrs["1"]["e"]
            scale_2, scaling_2 = self._scaling(fcgrs["2"]["seq_len"])
            b2 = fcgrs["2"]["b"]
            e2 = fcgrs["2"]["e"]

            # plot the data on the subplots
            img1 = CGR.array2img(fcgrs["1"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            img1 = Image.fromarray(img1)
            ax1.imshow(img1, cmap='gray', extent=extent)  # Reds_r
            ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax1.set_title(f'Sequence 1\n{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')

            import matplotlib.colors as mcolors
            norm = mcolors.TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
            im2 = ax2.imshow(fcgrs['diff'], cmap='seismic', norm=norm, extent=extent)
            ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax2.set_title(f'Difference\ndistance = {round(fcgrs["distance"], 4)}')

            img2 = CGR.array2img(fcgrs["2"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            img2 = Image.fromarray(img2)
            ax3.imshow(img2, cmap='gray', extent=extent)  # Blues_r
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
            scale_2, scaling_2 = self._scaling(fcgrs[index]["seq_len"])
            b2 = fcgrs[index]["b"]
            e2 = fcgrs[index]["e"]

            # plot the data on the subplots
            img1 = CGR.array2img(fcgrs["ref"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            img1 = Image.fromarray(img1)
            ax1.imshow(img1, cmap='gray', extent=extent)  # Reds_r
            ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax1.set_title(f'Reference\n{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')

            img2 = CGR.array2img(fcgrs[index]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
            img2 = Image.fromarray(img2)
            ax3.imshow(img2, cmap='gray', extent=extent)  # Blues_r
            ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax3.set_title(f'Segment\n{round(b2 / scale_2, 2)} - {round(e2 / scale_2, 2)} {scaling_2}')

            # --- add distance text below both panels ---
            fig.subplots_adjust(bottom=0.12)  # make room for the text
            fig.text(0.5, 0.06, f"Distance = {round(fcgrs[index]['distance'], 4)}",
                     ha="center", va="center", fontsize=14)

        return fig

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
            if panel_type == "fcgr" or panel_type == "chart":
                save_btn = getattr(self, save_btn_attr, None)
                if (save_btn is None or not save_btn.winfo_exists() or save_btn.master is not frame):
                    save_btn = ctk.CTkButton(master=frame, text="💾", width=30, height=30,
                                             fg_color=COLORS["BORDER_COLOR"], hover_color=COLORS["FRAME_HOVER_COLOR"],
                                             command=save_command, )
                    save_btn.place(relx=0.01, rely=0.99, anchor="sw")
                    setattr(self, save_btn_attr, save_btn)
            else:
                # TODO: need to add save, zoom, pan, reset buttons for 3d plot
                pass
                # if getattr(self, "t3_mds_toolbar", None) is None or not self.t3_mds_toolbar.winfo_exists():
                #     bg = self._get_effective_bg_color(frame)  # MUST be hex after your converter
                #
                #     tb = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
                #     tb.update()
                #
                #     self._configure_mds_toolbar(tb, frame_bg=bg, keep_buttons=("Home", "Pan", "Zoom", "Save"),
                #                                 btn_bg=COLORS["BORDER_COLOR"], btn_active=COLORS["FRAME_HOVER_COLOR"])
                #     # Put toolbar
                #     frame.grid_columnconfigure(0, weight=1)
                #     tb.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
                #     tb.configure(background=bg, bd=0, relief="flat", highlightthickness=0)
                #
                #     self.t3_mds_toolbar = tb
                # # tb = NavigationToolbar2Tk(canvas, frame, pack_toolbar=False)
                # # tb.update()
                # # tb.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="ew")
                # # self.mds_toolbar = tb

        # --- 3) Clear figure and re-plot ---
        fig.clear()
        if panel_type == "fcgr":
            # If we are in third tab and no fcgrs_dict provided, load from pickle
            if fig_attr == "t3_fcgr_fig" and not fcgrs_dict:
                with open(f"{self.temp_output_path}/t3_run/t3_run.pkl", 'rb') as handle:
                    fcgrs_dict = pickle.load(handle)
            self._plot_fcgrs(fcgrs_dict, bg=bg, fig=fig, index=index)
        elif panel_type == "chart":
            dists = list(self.t3_cgr_distance_history)
            self._plot_charts(fig=fig, bg=bg, dists=dists, index=index, canvas=canvas)
        elif panel_type == "mds":
            self._plot_mds(fig=fig, bg=bg, D=D, index=index, canvas=canvas)

        # --- 4) Hide placeholder if present ---
        placeholder = getattr(self, placeholder_attr, None)
        if placeholder is not None and placeholder.winfo_exists():
            try:
                placeholder.place_forget()
            except Exception:
                pass

        # --- 5) Redraw canvas ---
        canvas.draw()

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
        # If already drawn, just update selection color and return
        if getattr(self, "_t3_mds_drawn", False):
            self._t3_mds_set_selected(index)
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
        ax.set_position([0.00, 0.05, 1.00, 0.95])

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

        # ---- Segment points: pickable ----
        xs, ys, zs = X_seg[:, 0], X_seg[:, 1], X_seg[:, 2]

        colors = np.tile(default_color, (n_seg, 1))
        colors[index] = selected_color

        sc_seg = ax.scatter(xs, ys, zs, s=30, picker=True, depthshade=False)
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
                            c=[ref_color], s=60, picker=False, zorder=5, marker='*')
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
        canvas.draw_idle()

        legend_elements = [Line2D([0], [0], marker='*', color='none', markerfacecolor=ref_color,
                                  markeredgecolor=ref_color, markersize=6, label='Reference'), ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=6,
                  frameon=False, bbox_to_anchor=(1.30, 1.0), handletextpad=0.3, handlelength=1.0)

    # --------------------------------------------------
    # Other functions
    # --------------------------------------------------
    def t3_gen_synth_seq_event(self):
        def _accept_sequence(seq):
            if len(seq) > 0:
                self.t3_ds["1"].specie.set("Synthetic")
                self.t3_ds["1"].invalidate_based_specie()
                self.t3_ds["1"].seq = seq
                self.t3_ds["1"].end_seq.set(len(self.t3_ds["1"].seq))
                self.t3_species_combobox_1.set("Synthetic")
                self.t3_chr_combobox_1.configure(values=[])
                self.t3_parts_name_combobox.configure(values=[])
                self.t3_start_seq_entry.configure(state="normal")
                self.t3_end_seq_entry.configure(state="normal")
                self.sync_text_vars(self.t3_ds, "1")
            else:
                messagebox.showerror("Error", "No sequence generated.")

        dialog = GenerateSyntheticSequence(self, on_save=_accept_sequence)
        self.wait_window(dialog)  # blocks until pop-up calls save_sequence() or is closed

    def open_popup(self):
        # Slightly larger than half size (e.g., 60%)
        popup_width = int(self.winfo_width() * 0.6)
        popup_height = int(self.winfo_height() * 0.6)

        # Center position
        x = self.winfo_x() + (self.winfo_width() - popup_width) // 2
        y = self.winfo_y() + (self.winfo_height() - popup_height) // 2

        # Create popup window
        popup = ctk.CTkToplevel(self)
        popup.title("Search and Download genomes from NCBI")
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

        # Centered content in popup is in two frames
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_columnconfigure(1, weight=5)
        popup.grid_rowconfigure(0, weight=1)
        # Frames
        popup_f1 = ctk.CTkFrame(popup, corner_radius=10, border_color="#333333", border_width=2)
        popup_f1.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        popup_f1.grid_columnconfigure(0, weight=1)
        popup_f1.grid_rowconfigure(0, weight=1)
        popup_f1.grid_rowconfigure(1, weight=10)
        popup_f2 = ctk.CTkFrame(popup, corner_radius=10, border_color="#333333", border_width=2)
        popup_f2.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        popup_f2.grid_columnconfigure(0, weight=1)
        popup_f2.grid_rowconfigure(0, weight=1)
        popup_f2.grid_rowconfigure(1, weight=1)
        popup_f2.grid_rowconfigure(2, weight=10)
        popup_f2.grid_rowconfigure(3, weight=1)

        # Designing the first frame
        # Download path and Browse button in a frame
        download_frame = ctk.CTkFrame(popup_f1, fg_color=popup_f1.cget("fg_color"))
        download_frame.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=(5, 5))
        download_frame.grid_columnconfigure(0, weight=5)
        download_frame.grid_columnconfigure(1, weight=1)

        # Bring a list of species available (downloaded) in the folder selected
        file_scrollable_frame = ctk.CTkScrollableFrame(popup_f1)
        file_scrollable_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 10), pady=(5, 5))
        # download path label and entry
        ctk.CTkLabel(download_frame, text="Enter download path:") \
            .grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(5, 0))
        download_entry = ctk.CTkEntry(download_frame, textvariable=self.download_path)
        self.display_downloaded_files(file_scrollable_frame, self.download_path.get())
        download_entry.bind('<FocusOut>', partial(self.display_downloaded_files, file_scrollable_frame))
        download_entry.bind('<Key-Return>', partial(self.display_downloaded_files, file_scrollable_frame))
        download_entry.grid(row=1, column=0, sticky="we", padx=(5, 0), pady=(5, 0))
        # Browse button
        browse_button = ctk.CTkButton(download_frame, text="Browse...", width=80,
                                      command=lambda: self.browse_folder(file_scrollable_frame))
        browse_button.grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(5, 0))

        # Designing the second frame
        # Enter email in a frame
        email_frame = ctk.CTkFrame(popup_f2, fg_color=popup_f2.cget("fg_color"))
        email_frame.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=(5, 5))
        email_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(email_frame, text="Enter your email (required by NCBI):") \
            .grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(5, 0))
        email_entry = ctk.CTkEntry(email_frame, textvariable=self.email_var)
        email_entry.grid(row=1, column=0, sticky="we", padx=(5, 5), pady=(5, 0))

        # Search label and entry in a frame
        search_frame = ctk.CTkFrame(popup_f2, fg_color=popup_f2.cget("fg_color"))
        search_frame.grid(row=1, column=0, sticky="ew", padx=(5, 5), pady=(5, 5))
        search_frame.grid_columnconfigure(0, weight=10)
        search_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(search_frame, text="Enter organism name:") \
            .grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(5, 0))
        search_organism = ctk.StringVar(value="")
        search_entry = ctk.CTkEntry(search_frame, textvariable=search_organism)
        search_entry.grid(row=1, column=0, sticky="we", padx=(5, 0), pady=(5, 0))
        # # Show the search results
        # scrollable_frame = ctk.CTkScrollableFrame(popup_f2)
        # scrollable_frame.grid(row=2, column=0, sticky="nsew", padx=(10, 10), pady=(5, 5))
        scrollable_container = ctk.CTkFrame(popup_f2, fg_color=popup_f2.cget("fg_color"))
        scrollable_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        scrollable_container.grid_rowconfigure(0, weight=1)
        scrollable_container.grid_columnconfigure(0, weight=1)
        # Create a Canvas with custom background
        canvas = tkinter.Canvas(scrollable_container, bg="#2b2b2b", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        # Add scrollbars
        v_scrollbar = ctk.CTkScrollbar(scrollable_container, orientation="vertical", command=canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar = ctk.CTkScrollbar(scrollable_container, orientation="horizontal", command=canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Frame inside canvas to hold widgets
        self.checkbox_frame = ctk.CTkFrame(canvas, fg_color=popup_f2.cget("fg_color"))
        canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")

        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.checkbox_frame.bind("<Configure>", configure_scroll_region)

        # search button
        search_button = ctk.CTkButton(search_frame, text="Search", width=50,
                                      command=lambda: self.search_ncbi(email_entry, search_organism))
        search_button.grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(5, 0))

        # download button
        download_button = ctk.CTkButton(popup_f2, text="Download selected fasta",
                                        command=lambda: self.download_genomes(email_entry, search_organism,
                                                                              file_scrollable_frame))
        download_button.grid(row=3, column=0, sticky="ew", padx=(10, 10), pady=(5, 5))

    def browse_folder(self, scrollable_frame):
        default_start_path = self.download_path.get()
        folder = filedialog.askdirectory(title="Choose Folder", initialdir=default_start_path)
        if folder:
            self.download_path.set(folder)
        self.display_downloaded_files(scrollable_frame, self.download_path.get())

    def display_downloaded_files(self, frame, value):
        # Clear old widgets
        for widget in frame.winfo_children():
            widget.destroy()

        path = self.download_path.get()

        if not os.path.exists(path):
            ctk.CTkLabel(frame, text="Path does not exist", text_color="red").pack(anchor="w", padx=5, pady=2)
            return

        # Filter only FASTA files
        all_files = os.listdir(path)
        fasta_files = [f for f in all_files if f.lower().endswith(('.fasta', '.fa', '.fna'))]

        if not fasta_files:
            ctk.CTkLabel(frame, text="No FASTA files found", text_color="red").pack(anchor="w", padx=5,
                                                                                    pady=2)
            return

        for file in sorted(fasta_files):
            entry = ctk.CTkEntry(frame, width=500)
            entry.insert(0, file)
            entry.configure(state="readonly")
            entry.pack(fill="x", padx=5, pady=2)

            # Bind left-click to enable editing
            entry.bind("<Button-1>", lambda e, ent=entry: ent.configure(state="normal"))

            # Bind Enter key to rename file
            entry.bind("<Return>",
                       lambda e, ent=entry, old_name=file, folder=path: self.rename_file(ent, old_name, folder))

    @staticmethod
    def rename_file(entry_widget, old_name, folder):
        new_name = entry_widget.get().strip()
        old_path = os.path.join(folder, old_name)
        new_path = os.path.join(folder, new_name)

        if not new_name:
            messagebox.showerror("Invalid Name", "Filename cannot be empty.")
            entry_widget.delete(0, "end")
            entry_widget.insert(0, old_name)
            entry_widget.configure(state="readonly")
            return

        if new_name == old_name:
            entry_widget.configure(state="readonly")
            return

        if os.path.exists(new_path):
            messagebox.showerror("File Exists", "A file with this name already exists.")
            entry_widget.delete(0, "end")
            entry_widget.insert(0, old_name)
        else:
            try:
                os.rename(old_path, new_path)
                messagebox.showinfo("Success", f"Renamed to: {new_name}")
            except Exception as e:
                messagebox.showerror("Rename Failed", str(e))
                entry_widget.delete(0, "end")
                entry_widget.insert(0, old_name)

        entry_widget.configure(state="readonly")

    def search_ncbi(self, email_entry, query, *_):
        email = email_entry.get().strip()
        if not email:
            messagebox.showwarning("Missing Email", "Please enter your email address.")
            return
        Entrez.email = email

        query = query.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter an organism name.")
            return

        try:
            handle = Entrez.esearch(db="nuccore", term=f"{query}[Organism] AND chromosome[Title] AND complete genome",
                                    retmax=20)
            record = Entrez.read(handle)
            ids = record["IdList"]
            handle.close()

            if not ids:
                messagebox.showinfo("No Results", "No matching entries found.")
                return

            handle = Entrez.esummary(db="nuccore", id=",".join(ids))
            summary = Entrez.read(handle)
            handle.close()

            # Clear existing checkboxes
            for widget in self.checkbox_frame.winfo_children():
                widget.destroy()

            self.checkbox_vars = {}
            self.id_map = {}

            for item in summary:
                title = item["Title"]
                gi = item["Id"]

                var = ctk.BooleanVar()
                cb = ctk.CTkCheckBox(self.checkbox_frame, text=title, variable=var)
                cb.pack(fill="x", padx=5, pady=2)

                self.checkbox_vars[title] = var
                self.id_map[title] = gi

        except Exception as e:
            messagebox.showerror("Search Error", str(e))

    def download_genomes(self, email_entry, query, file_scrollable_frame, *_):
        email = email_entry.get().strip()
        if not email:
            messagebox.showwarning("Missing Email", "Please enter your email address.")
            return
        Entrez.email = email

        query = query.get().strip()
        if not query:
            messagebox.showwarning("Input Error", "Please enter an organism name.")
            return

        selected_titles = [title for title, var in self.checkbox_vars.items() if var.get()]
        if not selected_titles:
            messagebox.showwarning("No Selection", "Please select at least one entry.")
            return

        folder = self.download_path.get()
        if not os.path.exists(folder):
            messagebox.showerror("Invalid Path", "The specified download path does not exist.")
            return

        errors = []
        for title in selected_titles:
            try:
                seq_id = self.id_map.get(title)
                if not seq_id:
                    continue  # Skip if ID is missing

                handle = Entrez.efetch(db="nuccore", id=seq_id, rettype="fasta", retmode="text")
                fasta_data = handle.read()
                handle.close()

                # Sanitize filename
                filename = "".join(c if c.isalnum() or c in " ._-" else "_" for c in title)
                file_path = os.path.join(folder, filename + ".fasta")

                with open(file_path, "w") as f:
                    f.write(fasta_data)

            except Exception as e:
                errors.append(f"{title}: {str(e)}")

        if errors:
            messagebox.showerror("Partial Download", f"Some files failed to download:\n\n" + "\n".join(errors))
        else:
            messagebox.showinfo("Download Complete", f"All selected FASTA files were saved to:\n{folder}")

        # update the downloaded files display
        self.display_downloaded_files(file_scrollable_frame, self.download_path.get())

        # update available species in the combobox
        # get all the folders name in DATA path
        folders = [str(f) for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]
        # update the combobox values
        self.t2_species_combobox["1"].configure(values=folders)
        self.t2_species_combobox["2"].configure(values=folders)
        self.t2_species_combobox.configure(values=folders)
        self.t3_species_combobox_1.configure(values=folders)
        self.t3_species_combobox_2.configure(values=folders)
        self.t4_species_combobox.configure(values=folders)
        # add a value of 8 to BITS_DICT for this species
        # BITS_DICT.update({folder: 8 for folder in folders})

    def _get_effective_bg_color(self, widget):
        w = widget
        while w is not None:
            try:
                raw = w.cget("fg_color")
                color = self._resolve_ctk_color(raw)
                if color is not None:
                    return color
            except Exception:
                pass
            w = w.master

        # fallback
        try:
            return self._resolve_ctk_color(self.cget("fg_color")) or "#FFFFFF"
        except Exception:
            return "#FFFFFF"

    def _resolve_ctk_color(self, color):
        if color in (None, "transparent"):
            return None

        if isinstance(color, (list, tuple)) and len(color) == 2:
            mode = ctk.get_appearance_mode()  # "Light" or "Dark"
            color = color[0] if mode == "Light" else color[1]

        # Convert Tk/CTk color name -> hex for Matplotlib
        return self._tk_color_to_hex(color)

    def _tk_color_to_hex(self, color):
        if color is None:
            return None

        # already hex
        if isinstance(color, str) and color.startswith("#") and len(color) in (7, 9):
            return color[:7]  # ignore alpha if present

        try:
            r, g, b = self.winfo_rgb(color)  # 0..65535
            return f"#{r // 256:02x}{g // 256:02x}{b // 256:02x}"
        except Exception:
            # fallback: return as-is (might still work for some names)
            return color

    def _configure_mds_toolbar(self, tb, frame_bg, keep_buttons=("Home", "Pan", "Zoom", "Save"),
                               btn_bg="#E6E6E6", btn_active="#F2F2F2"):
        try:
            tb.configure(background=frame_bg, bd=0, relief="flat", highlightthickness=0)
        except Exception:
            pass

        # Remove the status message area
        if hasattr(tb, "_message_label") and tb._message_label is not None:
            try:
                tb._message_label.configure(background=frame_bg)
                tb._message_label.pack_forget()
            except Exception:
                pass

        # Walk children once: filter + style
        for w in tb.winfo_children():
            cls = w.winfo_class()

            # Filter toolbar buttons
            if cls == "Button":
                txt = (w.cget("text") or "").strip()

                # Hide unwanted buttons
                if txt not in set(keep_buttons):
                    try:
                        w.pack_forget()
                    except Exception:
                        try:
                            w.grid_forget()
                        except Exception:
                            pass
                    continue

                # Style kept buttons (make icons visible)
                try:
                    w.configure(background=btn_bg, activebackground=btn_active, relief="flat",
                                borderwidth=0, highlightthickness=0, padx=4, pady=2)
                except Exception:
                    pass

            # Internal frames/spacers: flatten & match bg
            elif cls in ("Frame", "Labelframe"):
                try:
                    w.configure(background=frame_bg, bd=0, relief="flat", highlightthickness=0)
                except Exception:
                    pass

            # Everything else: match bg
            else:
                try:
                    w.configure(background=frame_bg)
                except Exception:
                    pass


class GenerateSyntheticSequence(ctk.CTkToplevel):
    def __init__(self, parent, on_save):
        super().__init__(parent)  # <-- IMPORTANT: Toplevel (not CTk)
        self.parent = parent
        self.on_save = on_save  # callback to send sequence back
        self.generated_sequence = ""
        self.title("Generate Synthetic Sequence")

        # make it modal & on top of parent
        self.transient(parent)
        self.grab_set()
        self.focus_set()

        # size/center relative to parent (not the screen)
        parent.update_idletasks()
        w = int(parent.winfo_width() * 0.5)
        h = int(parent.winfo_height() * 0.6)
        x = parent.winfo_rootx() + (parent.winfo_width() - w) // 2
        y = parent.winfo_rooty() + (parent.winfo_height() - h) // 2
        self.geometry(f"{w}x{h}+{x}+{y}")

        # We have different tabviews
        tabview = ctk.CTkTabview(self, width=w - 20, height=h - 20)
        tabview.pack(padx=10, pady=10, fill="both", expand=True)
        tab_names = ["Entropy", "2-mers", "k-mers"]
        for name in tab_names:
            tabview.add(name)

        '''
        Designing the first tab (Entropy method)
        '''
        # Content in two frames
        tabview.tab(tab_names[0]).grid_columnconfigure(0, weight=1)
        tabview.tab(tab_names[0]).grid_columnconfigure(1, weight=5)
        tabview.tab(tab_names[0]).grid_rowconfigure(0, weight=50)
        tabview.tab(tab_names[0]).grid_rowconfigure(1, weight=1)
        # Frames
        t1_config = ctk.CTkFrame(tabview.tab(tab_names[0]), corner_radius=10, border_color="#333333",
                                 border_width=2)
        t1_config.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        t1_config.grid_columnconfigure(0, weight=1)
        t1_config.grid_columnconfigure(1, weight=1)
        t1_config.grid_columnconfigure(2, weight=1)

        self.t1_frame = ctk.CTkFrame(tabview.tab(tab_names[0]), corner_radius=10, border_color="#333333",
                                     border_width=2, fg_color="white")
        self.t1_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        self.t1_frame.grid_columnconfigure(0, weight=1)
        self.t1_frame.grid_rowconfigure(0, weight=1)

        # Design the configuration frame
        # k-mer size
        ctk.CTkLabel(t1_config, text="k-mer: ").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        values_list = ["2", "3", "4", "5", "6"]
        self.t1_k_var = ctk.IntVar()
        k_mer_combobox = ctk.CTkComboBox(t1_config, values=values_list, width=80, variable=self.t1_k_var)
        k_mer_combobox.set("6")  # Default value
        k_mer_combobox.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(10, 0))

        # sequence length
        ctk.CTkLabel(t1_config, text="Sequence length: ").grid(row=1, column=0, sticky="w", padx=10,
                                                               pady=(10, 0))
        self.seq_len = tkinter.StringVar(value="10000")  # Default value
        (ctk.CTkEntry(t1_config, textvariable=self.seq_len)
         .grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ew"))

        # entropy scaling factor
        ctk.CTkLabel(t1_config, text="Entropy scaling factor: ").grid(row=2, column=0, sticky="w", padx=10,
                                                                      pady=(10, 0))
        self.t1_r_var = ctk.DoubleVar(value=1.0)  # Default value
        (ctk.CTkSlider(t1_config, from_=0.25, to=1.0, variable=self.t1_r_var, width=150)
         .grid(row=2, column=1, padx=(0, 10), pady=(10, 0), sticky="ew"))
        self.t1_r_value_label = ctk.CTkLabel(t1_config, text=f"{self.t1_r_var.get():.2f}")
        self.t1_r_value_label.grid(row=2, column=2, padx=(0, 10), pady=(10, 0), sticky="w")
        self.t1_r_var.trace_add("write", self.update_r_label)
        # Generate button
        (ctk.CTkButton(t1_config, text="Generate", command=lambda: self.generate_sequence("t1"))
         .grid(row=3, column=0, columnspan=3, padx=10, pady=(10, 10)))

        # Save button
        (ctk.CTkButton(tabview.tab(tab_names[0]), text="Save Sequence", command=self.save_sequence)
         .grid(row=1, column=1, padx=(5, 5), pady=(5, 5)))

        '''
            Designing the second tab (2-mer method)
        '''
        tabview.tab(tab_names[1]).grid_columnconfigure(0, weight=1)
        tabview.tab(tab_names[1]).grid_columnconfigure(1, weight=5)
        tabview.tab(tab_names[1]).grid_rowconfigure(0, weight=50)
        tabview.tab(tab_names[1]).grid_rowconfigure(1, weight=1)
        # Frames
        t2_config = ctk.CTkFrame(tabview.tab(tab_names[1]), corner_radius=10, border_color="#333333",
                                 border_width=2)
        t2_config.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        t2_config.grid_columnconfigure(0, weight=0)
        t2_config.grid_columnconfigure(1, weight=1)
        t2_config.grid_columnconfigure(2, weight=0)

        self.t2_frame = ctk.CTkFrame(tabview.tab(tab_names[1]), corner_radius=10, border_color="#333333",
                                     border_width=2, fg_color="white")
        self.t2_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        self.t2_frame.grid_columnconfigure(0, weight=1)
        self.t2_frame.grid_rowconfigure(0, weight=1)

        # Design the configuration frame
        last_row = 0
        self.k_var_dict = {}
        self.k_value_label_dict = {}
        self.t2_kmers = generate_kmers(2)
        for i, kmer in enumerate(self.t2_kmers):
            padding = 0 if i > 0 else 5
            t2_config.grid_rowconfigure(i, weight=1)
            ctk.CTkLabel(t2_config, text=f"{kmer}: ").grid(row=i, column=0, padx=(10, 0), pady=(padding, 0),
                                                           sticky="w")
            # Create a slider
            var = ctk.DoubleVar(value=0.0)
            self.k_var_dict[kmer] = var
            r_slider = ctk.CTkSlider(t2_config, from_=-3, to=3, variable=var, width=150, height=14)
            r_slider.grid(row=i, column=1, padx=(10, 0), pady=(padding, 0), sticky="ew")
            # Create a label for the slider
            self.k_value_label_dict[kmer] = ctk.CTkLabel(t2_config, text="0.0000", width=60)
            self.k_value_label_dict[kmer].grid(row=i, column=2, padx=(10, 10), pady=(padding, 0), sticky="w")
            # Bind the slider to update the label
            try:
                var.trace_add("write", self.update_all_k_labels)
            except AttributeError:
                var.trace("w", self.update_all_k_labels)
            last_row = i + 1
        self.update_all_k_labels()

        # put sequence length
        seq_frame = ctk.CTkFrame(t2_config, fg_color="transparent")
        seq_frame.grid(row=last_row, column=0, columnspan=3, padx=(10, 10), pady=(5, 0), sticky="w")

        ctk.CTkLabel(seq_frame, text="Sequence length:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        ctk.CTkEntry(seq_frame, textvariable=self.seq_len, width=150).grid(row=0, column=1, sticky="w")

        # Generate button
        (ctk.CTkButton(t2_config, text="Generate", command=lambda: self.generate_sequence("t2"))
         .grid(row=last_row + 1, column=0, columnspan=3, padx=10, pady=(10, 10)))

        # Save button
        (ctk.CTkButton(tabview.tab(tab_names[1]), text="Save Sequence", command=self.save_sequence)
         .grid(row=1, column=1, padx=(5, 5), pady=(5, 5)))

        '''
            Designing the third tab (k-mer method)
        '''
        tabview.tab(tab_names[2]).grid_columnconfigure(0, weight=1)
        tabview.tab(tab_names[2]).grid_columnconfigure(1, weight=10)
        tabview.tab(tab_names[2]).grid_rowconfigure(0, weight=50)
        tabview.tab(tab_names[2]).grid_rowconfigure(1, weight=1)
        # Frames
        t3_config = ctk.CTkFrame(tabview.tab(tab_names[2]), corner_radius=10, border_color="#333333",
                                 border_width=2)
        t3_config.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        t3_config.grid_columnconfigure(0, weight=1)
        t3_config.grid_columnconfigure(1, weight=1)
        for i in range(6):
            t3_config.grid_rowconfigure(i, weight=1)

        self.t3_frame = ctk.CTkFrame(tabview.tab(tab_names[2]), corner_radius=10, border_color="#333333",
                                     border_width=2, fg_color="white")
        self.t3_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        self.t3_frame.grid_columnconfigure(0, weight=1)
        self.t3_frame.grid_rowconfigure(0, weight=1)

        # ========== STATE ==========
        self.k = 2
        self.t3_kmers = generate_kmers(self.k)  # order defines p_input order
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.t3_kmers)}
        self.logits = np.zeros(len(self.t3_kmers), dtype=float)  # start all equal
        self.current_kmer = None

        # Design the configuration frame
        # k combobox
        ctk.CTkLabel(t3_config, text="k-mer length: ").grid(row=0, column=0, padx=10, pady=(10, 0),
                                                            sticky="w")
        k_mer_combobox = ctk.CTkComboBox(t3_config, values=values_list, width=80, variable=self.t1_k_var,
                                         command=self.set_kmers_event)
        k_mer_combobox.set("2")
        k_mer_combobox.grid(row=0, column=1, padx=(0, 10), pady=(10, 0), sticky="w")

        # sequence length
        ctk.CTkLabel(t3_config, text="Sequence length: ").grid(row=1, column=0, sticky="w", padx=10,
                                                               pady=(10, 0))
        (ctk.CTkEntry(t3_config, textvariable=self.seq_len)
         .grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ew"))

        # k-mer entry + checkmark
        # frame
        entry_frame = ctk.CTkFrame(t3_config, fg_color="transparent")
        entry_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(5, 5), sticky="ew")
        entry_frame.grid_columnconfigure(0, weight=0)
        entry_frame.grid_columnconfigure(1, weight=1)
        entry_frame.grid_columnconfigure(2, weight=0)
        entry_frame.grid_columnconfigure(3, weight=0)
        ctk.CTkLabel(entry_frame, text="k-mer").grid(row=0, column=0, sticky="w")

        # entry
        self.kmer_entry = ctk.CTkEntry(entry_frame, placeholder_text="e.g., ACAC", width=80)
        self.kmer_entry.grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.kmer_entry.bind("<Return>", lambda _e: self._load_kmer_into_slider())
        # Slider
        self.t3_slider_var = ctk.DoubleVar(value=0.0)
        self.t3_kmer_slider = ctk.CTkSlider(entry_frame, from_=-3.0, to=3.0, variable=self.t3_slider_var)
        self.t3_kmer_slider.grid(row=1, column=1, sticky="ew", pady=(0, 5))
        self.t3_kmer_label = ctk.CTkLabel(entry_frame, text=f"{self.t3_slider_var.get():.4f}")
        self.t3_kmer_label.grid(row=1, column=2, sticky="w", pady=(0, 5))
        self.t3_slider_var.trace_add("write", self.update_t3_kmer_label)
        self.slider_normal_color = self.t3_kmer_slider.cget("button_color")
        self.t3_kmer_slider.configure(state="disabled",
                                      button_color="#888888")  # disable the slider until a k-mer is loaded
        # Save button
        self.save_btn = ctk.CTkButton(entry_frame, text="✓", width=10, command=self._refresh_summary)
        self.save_btn.grid(row=1, column=3, sticky="w", padx=(10, 0), pady=(0, 5))

        # summary textbox
        ctk.CTkLabel(t3_config, text="Summary").grid(row=3, column=0, padx=10, pady=(5, 0), sticky="w")
        self.summary_box = ctk.CTkTextbox(t3_config, height=160)
        self.summary_box.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

        # buttons
        btn_frame = ctk.CTkFrame(t3_config, fg_color="transparent")
        btn_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.reset_btn = ctk.CTkButton(btn_frame, text="Reset logits", command=self._reset_logits)
        self.reset_btn.pack(side="left")
        self.gen_btn = ctk.CTkButton(btn_frame, text="Generate", command=lambda: self.generate_sequence("t3"))
        self.gen_btn.pack(side="right")

        # Save button
        (ctk.CTkButton(tabview.tab(tab_names[2]), text="Save Sequence", command=self.save_sequence)
         .grid(row=1, column=1, padx=(5, 5), pady=(5, 5)))

        # initial summary
        self._refresh_summary()

    def set_kmers_event(self, *args):
        try:
            k = self.t1_k_var.get()
        except ValueError:
            messagebox.showerror("Invalid k", "k must be an integer.")
            return
        self.t3_kmers = generate_kmers(k)
        self.kmer_to_idx = {kmer: i for i, kmer in enumerate(self.t3_kmers)}
        self.logits = np.zeros(len(self.t3_kmers), dtype=float)
        self.current_kmer = None
        self.kmer_entry.delete(0, tkinter.END)
        self.t3_slider_var.set(0.0)
        self.t3_kmer_slider.configure(state="disabled", button_color="#888888")
        self._refresh_summary()

    def _load_kmer_into_slider(self, *args):
        kmer = self.kmer_entry.get().strip().upper()
        if len(kmer) != self.t1_k_var.get() or any(c not in 'ACGT' for c in kmer):
            messagebox.showerror("Invalid k-mer",
                                 f"Please enter a length-{self.t1_k_var.get()} k-mer using only A,C,G,T.")
            self.t3_kmer_slider.configure(state="disabled", button_color="#888888")
            self.current_kmer = None
            self.t3_slider_var.set(0.0)
            self.t3_kmer_label.configure(text="0.0000")
            return
        self.t3_kmer_slider.configure(state="normal", button_color=self.slider_normal_color)
        self.current_kmer = kmer
        idx = self.kmer_to_idx[kmer]
        self.t3_slider_var.set(float(self.logits[idx]))

    def _softmax(self):
        x = self.logits - self.logits.max()
        e = np.exp(x)
        return e / e.sum()

    def update_t3_kmer_label(self, *args):
        if self.current_kmer is None:
            self.t3_kmer_label.configure(text=f"{self.t3_slider_var.get():.4f}")
            return
        idx = self.kmer_to_idx[self.current_kmer]
        self.logits[idx] = float(self.t3_slider_var.get())
        self.t3_kmer_label.configure(text=f"{self._softmax()[idx]:.4f}")

    def _reset_logits(self, *args):
        self.logits[:] = 0.0
        if self.current_kmer:
            idx = self.kmer_to_idx[self.current_kmer]
            self.t3_slider_var.set(float(self.logits[idx]))
        self._refresh_summary()

    def _refresh_summary(self):
        if self.kmer_entry.get().strip().upper() == "":
            self.t3_kmer_slider.configure(state="disabled", button_color="#888888")
            self.current_kmer = None
            self.t3_slider_var.set(0.0)
            self.t3_kmer_label.configure(text="0.0000")
        p = self._softmax()
        # summarize: if all equal
        if np.allclose(self.logits, self.logits[0]):
            txt = (f"All {len(self.t3_kmers)} k-mers have equal weight.\n"
                   f"Each probability = {1.0 / len(self.t3_kmers):.4f}\n")
        else:
            # list only k-mers whose logits differ from the median by > 1e-9 (assigned)
            med = np.median(self.logits)
            assigned = [(km, p[self.kmer_to_idx[km]]) for km in self.t3_kmers
                        if abs(self.logits[self.kmer_to_idx[km]] - med) > 1e-9]
            # keep the list short: show up to 20 explicitly
            assigned_sorted = sorted(assigned, key=lambda x: -x[1])[:20]
            others = 1.0 - sum(prob for _, prob in assigned_sorted)
            n_others = len(self.t3_kmers) - len(assigned_sorted)
            avg_other = (others / n_others) if n_others > 0 else 0.0

            lines = [f"Assigned k-mers (top {len(assigned_sorted)} shown):"]
            lines += [f"  {km}: {prob:.4f}" for km, prob in assigned_sorted]
            if n_others > 0:
                lines += [f"Others ({n_others}): avg ≈ {avg_other:.4f} (sum ≈ {others:.4f})"]
            txt = "\n".join(lines)

        self.summary_box.configure(state="normal")
        self.summary_box.delete("1.0", tkinter.END)
        self.summary_box.insert("1.0", txt)
        self.summary_box.configure(state="disabled")

    def update_r_label(self, *args):
        self.t1_r_value_label.configure(text=f"{self.t1_r_var.get():.2f}")

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

    def generate_sequence(self, frame_num):
        if frame_num == "t1":
            plot_frame = self.t1_frame
            k = self.t1_k_var.get()
            seq_len = self.seq_len.get()
            r = self.t1_r_var.get()
            if k not in [2, 3, 4, 5, 6]:
                messagebox.showerror("Input Error", "Please select a valid k-mer size (2-6).")
                return
            try:
                seq_len = int(seq_len)
                if seq_len <= 0:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Input Error", "Please enter a valid positive integer for sequence length.")
                return
            if not (0.25 <= r <= 1.0):
                messagebox.showerror("Input Error", "Entropy scaling factor must be between 0.25 and 1.0.")
                return
            # # If all inputs are valid, proceed with sequence generation
            # messagebox.showinfo("Success", f"Generating sequence with k={k}, length={window_size}, r={r:.2f}")
            # Here you would call your sequence generation function
            target_entropy = r * (2 * k)
            sequence, _, p = generate_dna_sequence(k, seq_len, target_entropy=target_entropy)
        elif frame_num == "t2":
            plot_frame = self.t2_frame
            k = 2
            seq_len = int(self.seq_len.get())
            # Collect the slider values
            slider_values = [self.k_var_dict[kmer].get() for kmer in self.t2_kmers]
            sequence, _, p = generate_dna_sequence(k, seq_len, p_input=slider_values)
        elif frame_num == "t3":
            plot_frame = self.t3_frame
            k = self.t1_k_var.get()
            seq_len = int(self.seq_len.get())
            if np.allclose(self.logits, self.logits[0]):
                p_input = None
            else:
                p_input = self._softmax()
            sequence, _, p = generate_dna_sequence(k, seq_len, p_input=p_input)
        else:
            return
        self.generated_sequence = sequence
        fcgr = CGR(sequence, k).get_fcgr()
        # Generate the CGR image and display it
        fig, (ax1) = plt.subplots(1, 1)
        extent = 0, 1, 0, 1
        # Assuming extent = [xmin, xmax, ymin, ymax]
        xmin, xmax, ymin, ymax = extent
        offset = 0.01 * (xmax - xmin)  # 1% of the width/height as padding

        ax1.text(xmin - offset, ymax + offset, 'C', fontsize=12, fontweight='bold', ha='right', va='bottom')
        ax1.text(xmax + offset, ymax + offset, 'G', fontsize=12, fontweight='bold', ha='left', va='bottom')
        ax1.text(xmin - offset, ymin - offset, 'A', fontsize=12, fontweight='bold', ha='right', va='top')
        ax1.text(xmax + offset, ymin - offset, 'T', fontsize=12, fontweight='bold', ha='left', va='top')

        display_frame_color = plot_frame.cget("fg_color")
        fig.patch.set_facecolor(display_frame_color)

        fcgr_image = CGR.array2img(fcgr, bits=8, resolution=2)
        fcgr_image = Image.fromarray(fcgr_image, 'L')
        ax1.imshow(fcgr_image, cmap='gray', extent=extent)
        ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)

        # Clear the previous figure from the display frame if any
        for widget in plot_frame.winfo_children():
            widget.destroy()

        # Create a canvas and add the figure to it
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()

        # Set the canvas size explicitly
        canvas_width = plot_frame.cget("width")
        canvas_height = plot_frame.cget("height")
        canvas.get_tk_widget().config(width=canvas_width, height=canvas_height)
        # Use grid to place the canvas
        canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        plt.close()

    def save_sequence(self):
        if hasattr(self, "generated_sequence") and self.on_save:
            self.on_save(self.generated_sequence)
        self.destroy()  # close the pop-up


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
