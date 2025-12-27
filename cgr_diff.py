import os
import pickle
import sys
import threading
import tkinter
import tkinter.messagebox
from functools import partial
import random

import customtkinter as ctk
from tkinter import filedialog, messagebox
import tkinter.filedialog as fd

import numpy as np
from Bio import Entrez
from PIL import Image

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib import pyplot as plt

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
    FRAME_HOVER_COLOR="#444444",
    BORDER_COLOR="#333333", )
KMERS = [str(i) for i in range(1, 9)]
DISTANCES = ["Normalized Euclidean", "Cosine", "Manhattan", "Descriptor", "DSSIM", "K-S", "Wasserstein"]
RESOLUTION_DICT = {2: 2, 3: 2, 4: 2, 5: 2, 6: 2, 7: 2, 8: 4, 9: 4, 10: 4, 11: 4, 12: 4}


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

        # Variables for page 2
        self.t2_ds = {'1': GUIDataStructure(), '2': GUIDataStructure()}
        self.window_s_toggle = tkinter.IntVar(value=0)  # 0: Variable, 1: Fix
        self.window_s = tkinter.StringVar(value="")  # Window size entry variable
        self.window_entry = None  # Entry widget for window size
        self.k_var = ctk.IntVar(value=6)  # k-mer selection variable
        self.dist_metric = tkinter.StringVar(value="DSSIM")  # distance metric selection variable
        self.checkbox_RC = {}  # reverse complement checkbox dictionary
        self.checkbox_Random = {}  # random sequence checkbox dictionary

        self.start_seq_scale = {}
        self.end_seq_scale = {}
        self.start_seq_entry = {}
        self.end_seq_entry = {}

        self.t2_fig = None
        self.t2_canvas = None
        self.t2_save_btn = None

        # ------------------------- Build UI -------------------------
        self._create_top_navbar()
        self._build_main_content()

        self._upload_files(hard_coded=True)

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
        # elif self.active_tab == "Common Reference":
        #     self._build_common_reference(main)
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

    def _toggle_theme(self):
        new_mode = "Light" if self.appearance.get() == "Dark" else "Dark"
        self.appearance.set(new_mode)
        ctk.set_appearance_mode(new_mode)
        # update icon
        if new_mode == "Dark":
            self.theme_button.configure(text="☀️")
        else:
            self.theme_button.configure(text="🌙")

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
        display_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 0), sticky="nsew")
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
        slider_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        for i in range(4):
            slider_frame.grid_rowconfigure(i, weight=1)
        slider_frame.grid_columnconfigure(2, weight=1)
        slider_frame.grid_propagate(False)
        display_frame = ctk.CTkFrame(parent, corner_radius=8, border_width=1, border_color=COLORS["BORDER_COLOR"],
                                     fg_color=COLORS["FRAME_COLOR"])
        display_frame.grid(row=1, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        display_frame.grid_columnconfigure(0, weight=1)
        display_frame.grid_rowconfigure(0, weight=1)
        display_frame.grid_propagate(False)

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

            t2_seq_combobox[f"{i + 1}"] = ctk.CTkComboBox(seq_frame, values=self.file_names,
                                                          variable=self.t2_ds[str(i + 1)].seq_name,
                                                          command=partial(self.t2_sequence_selection_event, f"{i + 1}"))
            t2_seq_combobox[f"{i + 1}"].grid(row=i, column=1, sticky="ew", padx=(0, 10), pady=(10, 10 * i))

        # Radio Button (Window size)
        # Frame for window size settings
        window_size_frame = ctk.CTkFrame(config_frame, fg_color="transparent", border_color=COLORS["BORDER_COLOR"],
                                         border_width=1, corner_radius=8)
        window_size_frame.grid(row=1, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        window_size_frame.grid_columnconfigure(0, weight=1)
        window_size_frame.grid_columnconfigure(1, weight=1)
        for i in range(3):
            window_size_frame.grid_rowconfigure(i, weight=1)

        (ctk.CTkLabel(window_size_frame, text="Window Size", font=HEADER_FONT_BOLD)
         .grid(row=0, column=0, sticky="w", padx=5))
        ctk.CTkRadioButton(window_size_frame, text="Variable", variable=self.window_s_toggle, value=0,
                           command=self.window_size_toggle_event).grid(row=1, column=0, padx=5, pady=5, sticky="w")
        ctk.CTkRadioButton(window_size_frame, text="Fix", variable=self.window_s_toggle, value=1,
                           command=self.window_size_toggle_event).grid(row=1, column=1, padx=5, pady=5, sticky="w")

        self.window_entry = ctk.CTkEntry(window_size_frame, textvariable=self.window_s)
        self.window_entry.bind('<FocusOut>', partial(self.sequence_value_change, "0"))
        self.window_entry.bind('<Key-Return>', partial(self.sequence_value_change, "0"))
        if self.window_s_toggle.get() == 0:
            self.window_entry.configure(state="disabled")
        self.window_entry.grid(row=2, columnspan=2, padx=(5, 5), pady=(10, 10), sticky="ew")

        # Frame for k-mer selection and distance selection
        # k-mer selection
        kmer_frame = ctk.CTkFrame(config_frame, fg_color="transparent")
        kmer_frame.grid(row=2, column=0, padx=(10, 10), pady=(10, 10), sticky="nsew")
        kmer_frame.grid_columnconfigure(0, weight=1)
        kmer_frame.grid_columnconfigure(1, weight=1)
        kmer_frame.grid_rowconfigure(0, weight=1)
        kmer_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(kmer_frame, text="k-mer: ", font=HEADER_FONT_BOLD).grid(row=0, column=0, sticky="w", padx=(5, 0))
        (ctk.CTkComboBox(kmer_frame, values=KMERS, state="normal", variable=self.k_var)
         .grid(row=0, column=1, sticky="ew", padx=(0, 5)))

        # distance measure selection
        (ctk.CTkLabel(kmer_frame, text="Distance measure: ", font=HEADER_FONT_BOLD)
         .grid(row=1, column=0, padx=(5, 0), pady=(10, 0), sticky="w"))
        (ctk.CTkComboBox(kmer_frame, values=DISTANCES, variable=self.dist_metric)
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
            self.checkbox_RC[str(i + 1)] = ctk.CTkCheckBox(master=rv_frame, text="Reverse Complement")
            self.checkbox_RC[str(i + 1)].grid(row=(i * 2), column=1, padx=(10, 0), pady=(10, 0), sticky="w")
            self.checkbox_Random[str(i + 1)] = ctk.CTkCheckBox(master=rv_frame, text="Shuffle")
            self.checkbox_Random[str(i + 1)].grid(row=(i * 2) + 1, column=1, padx=(10, 0), pady=(10, 10), sticky="w")

        # plot button
        plot_button = ctk.CTkButton(config_frame, text="Plot", corner_radius=8, height=35, font=HEADER_FONT,
                                    command=partial(self.t2_plot, display_frame))
        plot_button.grid(row=4, column=0, pady=(10, 10))

        # ---------- Design the slider frame ----------
        # TODO: When the window size is fixed, start slider should have a limit and entry cannot go beyond the limit
        #  also end entry cannot go beyond the limit of the sequence length
        for i in range(2):
            (ctk.CTkLabel(slider_frame, text=f'Sequence {i + 1}:', font=HEADER_FONT_BOLD)
             .grid(row=(i * 2), column=0, padx=(10, 0), pady=(10, 0)))
            (ctk.CTkLabel(slider_frame, text='Start').grid(row=(i * 2), column=1, padx=(5, 0), pady=(10, 0)))
            (ctk.CTkLabel(slider_frame, text='End').grid(row=(i * 2) + 1, column=1, padx=(5, 0), pady=(10, 0)))

            # The start slider and entry
            seq_length = len(self.t2_ds[str(i + 1)].seq)
            to_value = seq_length if seq_length > 0 else 1
            self.start_seq_scale[str(i + 1)] = ctk.CTkSlider(slider_frame, from_=0, to=to_value,
                                                             orientation="horizontal",
                                                             variable=self.t2_ds[str(i + 1)].start_seq,
                                                             command=partial(self.sequence_value_change, str(i + 1)))
            if self.t2_ds[str(i + 1)].seq == '':
                self.start_seq_scale[str(i + 1)].set(0)
                self.start_seq_scale[str(i + 1)].configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
            self.start_seq_scale[str(i + 1)].grid(row=(i * 2), column=2, padx=(5, 0), pady=(10, 0), sticky="ew")

            self.start_seq_entry[str(i + 1)] = ctk.CTkEntry(slider_frame, textvariable=self.t2_ds[str(i + 1)].start_txt)
            self.start_seq_entry[str(i + 1)].bind('<FocusOut>', partial(self.sequence_value_change, "3"))
            self.start_seq_entry[str(i + 1)].bind('<Key-Return>', partial(self.sequence_value_change, "3"))
            if self.t2_ds[str(i + 1)].seq == '':
                self.start_seq_entry[str(i + 1)].configure(state="disabled")
            self.start_seq_entry[str(i + 1)].grid(row=(i * 2), column=3, padx=(5, 0), pady=(10, 0))
            ctk.CTkLabel(slider_frame, text='bp').grid(row=(i * 2), column=4, padx=(5, 10), pady=(10, 0))

            # The end slider and entry
            self.end_seq_scale[str(i + 1)] = ctk.CTkSlider(slider_frame, from_=0, to=to_value,
                                                           orientation="horizontal",
                                                           variable=self.t2_ds[str(i + 1)].end_seq,
                                                           command=partial(self.sequence_value_change, str(i + 1)))
            if self.t2_ds[str(i + 1)].seq == '':
                self.end_seq_scale[str(i + 1)].set(0)
                self.end_seq_scale[str(i + 1)].configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
            if self.window_s_toggle.get() == 1:
                self.end_seq_scale[str(i + 1)].configure(state="disabled", button_color=COLORS["DISABLED_BTN_COLOR"])
            self.end_seq_scale[str(i + 1)].grid(row=(i * 2) + 1, column=2, padx=(5, 0), pady=(10, 0), sticky="ew")

            self.end_seq_entry[str(i + 1)] = ctk.CTkEntry(slider_frame, textvariable=self.t2_ds[str(i + 1)].end_txt)
            self.end_seq_entry[str(i + 1)].bind('<FocusOut>', partial(self.sequence_value_change, "3"))
            self.end_seq_entry[str(i + 1)].bind('<Key-Return>', partial(self.sequence_value_change, "3"))
            if self.t2_ds[str(i + 1)].seq == '':
                self.end_seq_entry[str(i + 1)].configure(state="disabled")
            self.end_seq_entry[str(i + 1)].grid(row=(i * 2) + 1, column=3, padx=(5, 0), pady=(10, 0))
            ctk.CTkLabel(slider_frame, text='bp').grid(row=(i * 2) + 1, column=4, padx=(5, 10), pady=(10, 0))

    def _build_common_reference(self, parent):
        pass

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
        self.hist_frame = ctk.CTkFrame(self.display_content_frame, corner_radius=8, border_width=1, height=300)
        self.hist_frame.grid(row=0, column=0, columnspan=2, sticky="nsew", padx=10, pady=(0, 5), )
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
        # 1) k-mer (k=3) frequency analysis
        # self.kmer_freq = self._analyze_kmer_frequency(seq, k=3)
        # self._draw_kmer_histogram()

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
                self.start_seq_scale[sender].configure(state="normal", button_color=COLORS["BTN_COLOR"])
                self.end_seq_scale[sender].configure(state="normal", button_color=COLORS["BTN_COLOR"])
            else:
                self.start_seq_scale[sender].configure(state="disable", button_color=COLORS["DISABLED_BTN_COLOR"])
                self.end_seq_scale[sender].configure(state="disable", button_color=COLORS["DISABLED_BTN_COLOR"])
            self.start_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.start_seq_scale[sender].set(0)
            self.end_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.end_seq_scale[sender].set(len(self.t2_ds[sender].seq))
        except Exception as e:
            messagebox.showerror("Sequence is Empty!")

        # Set start and end in entries
        self.sync_text_vars(self.t2_ds, sender)
        # Clear window_s
        self.window_s_toggle.set(0)
        self.window_s.set("")
        self.window_entry.configure(state="disable")
        # Make end scales and entries normal
        for key, value in self.end_seq_scale.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal", button_color=COLORS["BTN_COLOR"])
        for key, value in self.end_seq_entry.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal")
        for key, value in self.start_seq_entry.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal")

    def window_size_toggle_event(self):
        if self.window_s_toggle.get() == 0:
            self.window_s.set("")
            self.window_entry.configure(state="disable")

            # end scales
            for key, value in self.end_seq_scale.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal", button_color=COLORS["BTN_COLOR"])
            for key, value in self.end_seq_entry.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal")

        else:
            self.window_entry.configure(state="normal")
            self.window_s.set("500_000")

            # end scales
            for key, value in self.end_seq_scale.items():
                value.configure(state="disable", button_color=COLORS["DISABLED_BTN_COLOR"])
            for key, value in self.end_seq_entry.items():
                value.configure(state="disable")
            for key, value in self.t2_ds.items():
                if len(self.t2_ds[key].seq) > 0:
                    self.t2_ds[key].end_seq.set(self.t2_ds[key].start_seq.get() + int(self.window_s.get()))

        for key, value in self.t2_ds.items():
            self.sync_text_vars(self.t2_ds, key)

    def sequence_value_change(self, sender, value):
        if sender == "0":  # Window size changed
            for key, value in self.t2_ds.items():
                if self.t2_ds[key].seq == '':
                    continue
                # If window size is out of range, send an error and set to 500,000
                if int(self.window_s.get()) < 1 or int(self.window_s.get()) > len(self.t2_ds[key].seq):
                    self.window_s.set("500_000")
                    messagebox.showerror("Error", "Window size is out of range.")
                self.t2_ds[key].end_seq.set(self.t2_ds[key].start_seq.get() + int(self.window_s.get()))
        elif sender in ["1", "2"]:  # Scale changed
            if self.window_s_toggle.get() == 1 and self.t2_ds[sender].seq != '':
                self.t2_ds[sender].end_seq.set(self.t2_ds[sender].start_seq.get() + int(self.window_s.get()))
        elif sender in ["3"]:  # Entry changed
            for key, value in self.t2_ds.items():
                self.reverse_sync_text_vars(self.t2_ds, key)
            if self.window_s_toggle.get() == 1:
                for key, value in self.t2_ds.items():
                    if self.t2_ds[key].seq == '':
                        continue
                    self.t2_ds[key].end_seq.set(self.t2_ds[key].start_seq.get() + int(self.window_s.get()))

        for key, value in self.t2_ds.items():
            self.sync_text_vars(self.t2_ds, key)

    @staticmethod
    def sync_text_vars(ds, sender):
        ds[sender].start_txt.set(f"{ds[sender].start_seq.get()}")
        ds[sender].end_txt.set(f"{ds[sender].end_seq.get()}")

    @staticmethod
    def reverse_sync_text_vars(ds, sender):
        # Prevent error if the entry is not digits (if its "" it is okay)
        if not ds[sender].start_txt.get().isdigit() and ds[sender].start_txt.get() != "":
            ds[sender].start_txt.set(0)
            return messagebox.showerror("Error", "Please enter a valid integer value.")
        if not ds[sender].end_txt.get().isdigit() and ds[sender].end_txt.get() != "":
            ds[sender].end_txt.set(len(ds[sender].seq))
            return messagebox.showerror("Error", "Please enter a valid integer value.")
        # Prevent error if the entry is out of range
        if int(ds[sender].start_txt.get()) < 0 or int(ds[sender].start_txt.get()) > len(ds[sender].seq):
            ds[sender].start_txt.set(0)
            return messagebox.showerror("Error", "The value is out of range.")
        if int(ds[sender].end_txt.get()) < 0 or int(ds[sender].end_txt.get()) > len(ds[sender].seq):
            ds[sender].end_txt.set(len(ds[sender].seq))
            return messagebox.showerror("Error", "The value is out of range.")
        ds[sender].start_seq.set(int(ds[sender].start_txt.get()))
        ds[sender].end_seq.set(int(ds[sender].end_txt.get()))

    def t2_plot(self, display_frame):
        if self.t2_ds["1"].seq == "" or self.t2_ds["2"].seq == "":
            messagebox.showerror("Error", "Please upload or choose the sequences first")
            return
        if self.k_var.get() == 0:
            messagebox.showerror("Error", "Please choose the k-mer value")
            return
        if self.dist_metric.get() == "":
            messagebox.showerror("Error", "Please choose the distance measure")
            return
        fcgrs_dict = {}
        for key in self.t2_ds.keys():
            fcgrs_dict[key] = {}
            seq = self.t2_ds[key].seq[self.t2_ds[key].start_seq.get():self.t2_ds[key].end_seq.get()]
            if self.checkbox_RC[key].get():
                seq = self.get_reverse_complement(seq)
            if self.checkbox_Random[key].get():
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

        # Visualize the FCGRs
        # First time: create figure, draw into it off-screen, then attach canvas
        bg = display_frame.cget("fg_color")
        if self.t2_fig is None:
            # Let Tk lay out the frame so grid sizes are sane
            display_frame.update_idletasks()
            # Create a reasonably sized figure
            self.t2_fig = plt.Figure(dpi=120)
            self.t2_fig.patch.set_facecolor(bg)
            # Draw everything into the figure *before* the canvas exists
            self.t2_fig.clear()
            self._plot_fcgrs(fcgrs_dict, colormap=True, background_color=bg, name="Sequence", fig=self.t2_fig, )

            # Now create and grid the canvas once the figure is ready
            self.t2_canvas = FigureCanvasTkAgg(self.t2_fig, master=display_frame)
            widget = self.t2_canvas.get_tk_widget()
            widget.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

            # Create tiny save button once, bottom-left over the figure
            if self.t2_save_btn is None:
                self.t2_save_btn = ctk.CTkButton(master=display_frame, text="💾", width=30, height=30,
                                                 fg_color=COLORS["BORDER_COLOR"],
                                                 hover_color=COLORS["FRAME_HOVER_COLOR"],
                                                 command=self.save_t2_figure, )
                # place() with relative coords: bottom-left corner of the frame
                self.t2_save_btn.place(relx=0.01, rely=0.99, anchor="sw")
        else:
            # Subsequent plots: reuse same figure & canvas
            self.t2_fig.clear()
            self._plot_fcgrs(fcgrs_dict, colormap=True, background_color=bg, name="Sequence", fig=self.t2_fig, )

        # Redraw (both first and later calls)
        self.t2_canvas.draw()

    def _plot_fcgrs(self, fcgrs, colormap=False, background_color=None, name="Sequence", fig=None):
        if fig is None:
            fig = plt.Figure()
        ax1, ax2, ax3 = fig.subplots(1, 3)
        extent = 0, 1, 0, 1

        if background_color is not None:
            fig.patch.set_facecolor(background_color)

        scale_1, scaling_1 = self.get_scaling(fcgrs["1"]["seq_len"])
        b1 = fcgrs["1"]["b"]
        e1 = fcgrs["1"]["e"]
        scale_2, scaling_2 = self.get_scaling(fcgrs["2"]["seq_len"])
        b2 = fcgrs["2"]["b"]
        e2 = fcgrs["2"]["e"]

        # plot the data on the subplots
        img1 = CGR.array2img(fcgrs["1"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
        img1 = Image.fromarray(img1)
        ax1.imshow(img1, cmap='gray', extent=extent)  # Reds_r
        ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax1.set_title(f'{name} 1\n{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')

        im2 = ax2.imshow(fcgrs['diff'], cmap='RdBu', norm=plt.Normalize(-100, 100), extent=extent)
        ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax2.set_title(f'Difference\ndistance = {round(fcgrs["distance"], 4)}')

        img2 = CGR.array2img(fcgrs["2"]["fcgr"], bits=8, resolution=RESOLUTION_DICT[self.k_var.get()])
        img2 = Image.fromarray(img2)
        ax3.imshow(img2, cmap='gray', extent=extent)  # Blues_r
        ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax3.set_title(f'{name} 2\n{round(b2 / scale_2, 2)} - {round(e2 / scale_2, 2)} {scaling_2}')

        if colormap:
            fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
            cbar_ax2 = fig.add_axes([0.36, 0.1, 0.3, 0.02])  # Adjust position as needed
            cbar = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
            cbar.set_label(f'Red: Greater k-mer value in {name} 1 , Blue: Greater k-mer value in {name} 2', fontsize=10)
            cbar.ax.xaxis.set_label_position('top')  # Position label at top of colorbar
            cbar.ax.xaxis.labelpad = 5
            cbar.ax.tick_params(labelsize=8)

        return fig

    def save_t2_figure(self):
        if self.t2_fig is None:
            messagebox.showerror("Error", "No figure to save. Please plot first.")
            return

        file_path = fd.asksaveasfilename(defaultextension=".png",
                                         filetypes=[("PNG Image", "*.png"), ("PDF Document", "*.pdf"),
                                                    ("SVG Image", "*.svg"), ("All Files", "*.*"), ],
                                         title="Save figure")
        if not file_path:
            return

        try:
            self.t2_fig.savefig(file_path, dpi=300, bbox_inches="tight")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save figure:\n{e}")

    @staticmethod
    def get_scaling(chromosome_length):
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
    def get_reverse_complement(sequence):
        complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
        bases = [complement[base] for base in sequence]
        bases = reversed(bases)
        return ''.join(bases)

    @staticmethod
    def tk_to_hex(widget, color_name):
        r, g, b = widget.winfo_rgb(color_name)
        return f"#{r // 256:02x}{g // 256:02x}{b // 256:02x}"

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

    # --------------------------------------------------
    #
    # --------------------------------------------------
    def _change_images(self, index, tab_name, value):
        # plot distance results bar and first index is red
        index = round(value) if value is not None else index
        self._plot_chart(index, tab_name)
        # Load and display the first image set in next plot
        self.__plot_fcgrs(index, tab_name)

    def _plot_chart(self, highlighted_index, tab_name):
        dist_history = None
        frame = None
        if tab_name == "t2":
            dist_history = self.cgr_distance_history
            frame = self.t2_plot_frame
        elif tab_name == "t3":
            dist_history = self.t3_cgr_distance_history
            frame = self.t3_plot_frame
        elif tab_name == "t4":
            dist_history = self.t4_cgr_distance_history
            frame = self.t4_plot_frame

        fig, ax1 = plt.subplots(figsize=(100, 2))
        # Set x-axis limits
        # ax1.set_xlim(0, len(dist_history) + 1)  # Set the x-axis to start at 1
        x = np.arange(1, len(dist_history) + 1)  # Start from 1 instead of 0
        y = np.asarray(dist_history)

        plot_type = self.plot_type_var.get()
        if plot_type == "Bar plot":
            ax1.set_xlim(x[0] - 0.9, x[-1] + 0.9)
            mask1 = x == highlighted_index + 1
            mask2 = x != highlighted_index + 1
            # bar_width = 0.5
            ax1.bar(x[mask1], y[mask1], color='red')  # , width=bar_width)
            ax1.bar(x[mask2], y[mask2], color='blue')  # , width=bar_width)

            # Set titles for x and y axes
            ax1.set_xlabel('Segment number')
            ax1.set_ylabel('Distance value')
        elif plot_type == "Histogram plot":
            n_bins = 30
            ax1.hist(y, bins=n_bins, color='blue')
            highlighted_value = dist_history[highlighted_index]
            ax1.axvline(highlighted_value, color='red', linestyle='--', linewidth=1)
            ax1.set_xlabel('Distance value')
            ax1.set_ylabel('Frequency')
        else:
            messagebox.showerror("Error", "Please choose a plot type")
            return

        # Adjust layout to make room for x-axis title
        fig.subplots_adjust(bottom=0.2)  # Increase the bottom margin

        # Clear the previous figure from the display frame if any
        for widget in frame.winfo_children():
            widget.destroy()

        # Create a canvas and add the figure to it
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        # Set the canvas size explicitly
        canvas_width = frame.cget("width")
        canvas_height = frame.cget("height")
        canvas.get_tk_widget().config(width=canvas_width, height=canvas_height)
        # Use grid to place the canvas
        canvas.get_tk_widget().grid(row=0, column=0, padx=(8, 8), pady=5, sticky='nsew')
        plt.close()

    def __plot_fcgrs(self, image_index, tab_name):
        frame, fig = None, None
        if tab_name == "t2":
            with open(f"{self.temp_output_path}/consecutive/pickle/{image_index}.pkl", 'rb') as handle:
                dictionary = pickle.load(handle)
            frame = self.t2_display_frame

            display_frame_color = frame.cget("fg_color")
            fig = self.plot_fcgrs(dictionary, colormap=True, background_color=display_frame_color, name="Segment")

        elif tab_name == "t3":
            with open(f"{self.temp_output_path}/common_ref/pickle/{image_index}.pkl", 'rb') as handle:
                dictionary = pickle.load(handle)
            frame = self.t3_display_frame_2

            fig, (ax2, ax3) = plt.subplots(1, 2)
            fig.subplots_adjust(top=0.85)
            extent = 0, 1, 0, 1

            display_frame_color = frame.cget("fg_color")
            fig.patch.set_facecolor(display_frame_color)

            scale_2, scaling_2 = self.get_scaling(dictionary["chr_len"])
            b2 = dictionary["b"]
            e2 = dictionary["e"]

            # plot the data on the subplots
            im2 = ax2.imshow(dictionary['diff'], cmap='RdBu', norm=plt.Normalize(-100, 100), extent=extent)
            ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax2.set_title(f'Difference\ndistance = {round(dictionary["distance"], 4)}')

            img2 = CGR.array2img(dictionary["(f)cgr"], bits=8,  # BITS_DICT[dictionary["species"]]
                                 resolution=RESOLUTION_DICT[self.k_var.get()])
            img2 = Image.fromarray(img2, 'L')
            ax3.imshow(img2, cmap='gray', extent=extent)
            ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax3.set_title(f'Segment\n{round(b2 / scale_2, 2)} - {round(e2 / scale_2, 2)} {scaling_2}')

            fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
            cbar_ax2 = fig.add_axes([0.36, 0.1, 0.3, 0.02])  # Adjust position as needed
            cbar = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
            cbar.set_label('Red: Greater k-mer value in Reference , Blue: Greater k-mer value in Segment', fontsize=10)
            cbar.ax.xaxis.set_label_position('top')  # Position label at top of colorbar
            cbar.ax.xaxis.labelpad = 5
            cbar.ax.tick_params(labelsize=8)

        elif tab_name == "t4":
            with open(f"{self.temp_output_path}/representative/pickle/{image_index}.pkl", 'rb') as handle:
                dictionary = pickle.load(handle)
            frame = self.t4_display_frame_2

            fig, (ax2, ax3) = plt.subplots(1, 2)
            fig.subplots_adjust(top=0.85)
            extent = 0, 1, 0, 1

            display_frame_color = frame.cget("fg_color")
            fig.patch.set_facecolor(display_frame_color)

            scale_2, scaling_2 = self.get_scaling(dictionary["chr_len"])
            b2 = dictionary["b"]
            e2 = dictionary["e"]

            # plot the data on the subplots
            im2 = ax2.imshow(dictionary['diff'], cmap='RdBu', norm=plt.Normalize(-100, 100), extent=extent)
            ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax2.set_title(f'Difference\ndistance = {round(dictionary["distance"], 4)}')

            img2 = CGR.array2img(dictionary["(f)cgr"], bits=8,  # BITS_DICT[dictionary["species"]]
                                 resolution=RESOLUTION_DICT[self.k_var.get()])
            img2 = Image.fromarray(img2, 'L')
            ax3.imshow(img2, cmap='gray', extent=extent)
            ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax3.set_title(f'Segment\n{round(b2 / scale_2, 2)} - {round(e2 / scale_2, 2)} {scaling_2}')

            fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
            cbar_ax2 = fig.add_axes([0.36, 0.1, 0.3, 0.02])  # Adjust position as needed
            cbar = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
            cbar.set_label('Red: Greater k-mer value in Representative , Blue: Greater k-mer value in Segment',
                           fontsize=10)
            cbar.ax.xaxis.set_label_position('top')  # Position label at top of colorbar
            cbar.ax.xaxis.labelpad = 5
            cbar.ax.tick_params(labelsize=8)

        # Clear the previous figure from the display frame if any
        for widget in frame.winfo_children():
            widget.destroy()

        # Create a canvas and add the figure to it
        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()

        # Set the canvas size explicitly
        canvas_width = frame.cget("width")
        canvas_height = frame.cget("height")
        canvas.get_tk_widget().config(width=canvas_width, height=canvas_height)
        # Use grid to place the canvas
        canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=10, sticky='nsew')
        plt.close()

    def move_previous(self, tab_name, value):
        pic_num = None
        if tab_name == "t2":
            pic_num = self.t2_pic_num
        elif tab_name == "t3":
            pic_num = self.t3_pic_num
        elif tab_name == "t4":
            pic_num = self.t4_pic_num

        if pic_num.get() > 0:
            pic_num.set(pic_num.get() - 1)
            self._change_images(pic_num.get(), tab_name, None)

    def move_next(self, tab_name, value):
        pic_num, dist_history_len = None, None
        if tab_name == "t2":
            pic_num = self.t2_pic_num
            dist_history_len = len(self.cgr_distance_history)
        elif tab_name == "t3":
            pic_num = self.t3_pic_num
            dist_history_len = len(self.t3_cgr_distance_history)
        elif tab_name == "t4":
            pic_num = self.t4_pic_num
            dist_history_len = len(self.t4_cgr_distance_history)

        if pic_num.get() < dist_history_len - 1:
            pic_num.set(pic_num.get() + 1)
            self._change_images(pic_num.get(), tab_name, None)

    def t3_upload_event(self, sender, value):
        file_path = filedialog.askopenfilename()
        _, sequence = ChromosomesHolder.read_fasta(file_path)
        if len(sequence) > 0:
            self.t3_ds[sender].specie.set("Custom")
            self.t3_ds[sender].invalidate_based_specie()
            self.t3_ds[sender].seq = sequence
            self.t3_ds[sender].end_seq.set(len(self.t3_ds[sender].seq))
            if sender == "1":
                self.t3_species_combobox_1.set("Custom")
                self.t3_chr_combobox_1.configure(values=[])
                self.t3_parts_name_combobox.configure(values=[])
                self.t3_start_seq_entry.configure(state="normal")
                self.t3_end_seq_entry.configure(state="normal")
            elif sender == "2":
                self.t3_species_combobox_2.set("Custom")
                self.t3_chr_combobox_2.configure(values=[])
                self.t3_window_s.set("")
                self.t3_window_entry.configure(state="normal")
            self.sync_text_vars(self.t3_ds, sender)

    def t3_specie_change_event(self, sender, value):
        if sender == "1":
            self.t3_species_combobox_1.set(value)
        elif sender == "2":
            self.t3_species_combobox_2.set(value)
        self.t3_ds[sender].specie.set(value)

        specie = self.t3_ds[sender].specie.get()
        if sender == "1":
            self.t3_chr_combobox_1.configure(
                values=ChromosomesHolder(specie, self.data_path).get_all_chromosomes_name(include_whole_genome=True))
            # disable annotation combobox
            self.t3_parts_name_combobox.configure(values=[])
            self.t3_parts_name_combobox.configure(state="disable")
            self.t3_start_seq_entry.configure(state="disable")
            self.t3_end_seq_entry.configure(state="disable")
        elif sender == "2":
            self.t3_chr_combobox_2.configure(
                values=ChromosomesHolder(specie, self.data_path).get_all_chromosomes_name(include_whole_genome=True))
            self.t3_window_s.set("")
            self.t3_window_entry.configure(state="disable")
        self.t3_ds[sender].invalidate_based_specie()
        self.sync_text_vars(self.t3_ds, sender)

    def t3_chromosome_change_event(self, sender, value):
        specie = self.t3_ds[sender].specie.get()
        chromosome = self.t3_ds[sender].chromosome.get()
        # set its sequence
        self.t3_ds[sender].seq = ChromosomesHolder(specie, self.data_path).get_chromosome_sequence(chromosome)
        # set the annotation combobox
        if sender == "1":
            self.t3_start_seq_entry.configure(state="normal")
            self.t3_end_seq_entry.configure(state="normal")
            self.t3_parts_name_combobox.configure(state="normal")
            self.t3_parts_name_combobox.configure(
                values=ChromosomesHolder(specie, self.data_path).cytobands[chromosome])
        elif sender == "2":
            self.t3_window_entry.configure(state="normal")
        self.t3_ds[sender].invalidate_based_chromosome()
        # set end
        self.t3_ds[sender].end_seq.set(len(self.t3_ds[sender].seq))
        # sync with text

        self.sync_text_vars(self.t3_ds, sender)

    def t3_annotation_change_event(self, sender, value):
        annotation = self.t3_ds[sender].annotation.get()
        chromosome_name = self.t3_ds[sender].chromosome.get()
        annotation_info = ChromosomesHolder(self.t3_ds[sender].specie.get(), self.data_path).cytobands[chromosome_name][
            annotation]
        self.t3_ds[sender].start_seq.set(annotation_info.start)
        self.t3_ds[sender].end_seq.set(annotation_info.end)
        self.sync_text_vars(self.t3_ds, sender, keep_annotation=True)

    def t3_entry_change(self, value):
        self.t3_ds["1"].annotation.set("")
        self.t3_ds["1"].start_seq.set(int(self.t3_ds["1"].start_txt.get()))
        self.t3_ds["1"].end_seq.set(int(self.t3_ds["1"].end_txt.get()))

    def run_common_ref(self, event):
        if self.t3_ds["1"].seq == "" or self.t3_ds["2"].seq == "":
            messagebox.showerror("Error", "Please upload or choose the sequences first")
            return
        if self.k_var.get() == 0:
            messagebox.showerror("Error", "Please choose the k-mer value")
            return
        if self.dist_metric.get() == "":
            messagebox.showerror("Error", "Please choose the distance measure")
            return
        global foo_thread_2
        foo_thread_2 = threading.Thread(target=self.t3_run)
        foo_thread_2.daemon = True
        foo_thread_2.start()
        self.after(20, self.t3_check_thread)

    def t3_check_thread(self):
        self.t3_progress_bar.set(self._t3_progress)
        if foo_thread_2.is_alive():
            self.after(20, self.t3_check_thread)
        else:
            self.t3_progress_bar.set(1.0)
            self.t3_pic_num.set(0)
            self.t3_scale.configure(to=int(len(self.t3_cgr_distance_history) - 1))  # Update the scale range

            # Display the reference image
            with open(f"{self.temp_output_path}/common_ref/pickle/ref.pkl", 'rb') as handle:
                dictionary = pickle.load(handle)

            fig, (ax1) = plt.subplots(1, 1)
            extent = 0, 1, 0, 1

            display_frame_color = self.t3_display_frame_1.cget("fg_color")
            fig.patch.set_facecolor(display_frame_color)

            img1 = CGR.array2img(dictionary["(f)cgr"], bits=8,  # BITS_DICT[dictionary["species"]]
                                 resolution=RESOLUTION_DICT[self.k_var.get()])
            img1 = Image.fromarray(img1, 'L')
            ax1.imshow(img1, cmap='gray', extent=extent)
            ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            scale_1, scaling_1 = self.get_scaling(dictionary["chr_len"])
            b1 = int(self.t3_ds["1"].start_seq.get())
            e1 = int(self.t3_ds["1"].end_seq.get())
            if self.t3_ds["1"].specie.get() == "Custom":
                ax1.set_title(f'Reference\nCustom / '
                              f'{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')
            else:
                if self.t3_ds["1"].chromosome.get() == "Whole Genome":
                    chromosome = "Genome"
                else:
                    chromosome = f'chr {self.t3_ds["1"].chromosome.get()}'
                ax1.set_title(f'Reference\n{self.t3_ds["1"].specie.get()} / {chromosome} / '
                              f'{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')

            # Clear the previous figure from the display frame if any
            for widget in self.t3_display_frame_1.winfo_children():
                widget.destroy()

            # Create a canvas and add the figure to it
            canvas = FigureCanvasTkAgg(fig, master=self.t3_display_frame_1)
            canvas.draw()

            # Set the canvas size explicitly
            canvas_width = self.t3_display_frame_1.cget("width")
            canvas_height = self.t3_display_frame_1.cget("height")
            canvas.get_tk_widget().config(width=canvas_width, height=canvas_height)
            # Use grid to place the canvas
            canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
            plt.close()

            # Display the plot and the first set
            self._change_images(0, "t3", None)

    def t3_run(self):
        self.t3_cgr_distance_history = []
        t3_step_length = np.floor(len(self.t3_ds["2"].seq) / int(self.t3_window_s.get()))
        # self.t3_progress_bar.set(0)
        # self.t3_pic_num.set(0)
        self._t3_progress = 0.0

        ref_b = int(self.t3_ds["1"].start_seq.get())
        ref_e = int(self.t3_ds["1"].end_seq.get())
        ref_cgr = CGR(self.t3_ds["1"].seq[ref_b:ref_e], self.k_var.get())

        # self.t3_progress_bar.set(1 / (int(t3_step_length) + 2))  # start progress bar
        self._t3_progress = 1.0 / (int(t3_step_length) + 2)

        # if self.fcgr.get() == 1:
        im1 = ref_cgr.get_fcgr()
        # else:
        #     im1 = ref_cgr.get_cgr()

        # self.t3_progress_bar.set(2 / (int(t3_step_length) + 2))  # update progress bar
        self._t3_progress = 2.0 / (int(t3_step_length) + 2)

        ref_dict = {"(f)cgr": im1, "species": self.t3_ds["1"].specie.get(),
                    "chr_len": len(self.t3_ds["1"].seq)}

        path = f"{self.temp_output_path}/common_ref/pickle"
        if not os.path.exists(path):
            os.makedirs(path)
        with open(f"{path}/ref.pkl", 'wb') as f:
            pickle.dump(ref_dict, f)

        # the sliding sequence
        for i in range(int(t3_step_length)):
            # self.t3_progress_bar.set((i + 3) / (int(t3_step_length) + 2))
            self._t3_progress = (i + 3) / (int(t3_step_length) + 2)
            b2 = i * int(self.t3_window_s.get())
            e2 = (i + 1) * int(self.t3_window_s.get())

            cgr2 = CGR(self.t3_ds["2"].seq[b2:e2], self.k_var.get())
            # if self.fcgr.get() == 1:
            im2 = cgr2.get_fcgr()
            # else:
            #     im2 = cgr2.get_cgr()

            diff = im2 - im1

            dist = get_dist(im1, im2, dist_m=self.dist_metric.get())

            self.t3_cgr_distance_history.append(dist)

            dictionary = {"(f)cgr": im2, "b": b2, "e": e2, "chr_len": len(self.t3_ds["2"].seq),
                          "diff": diff, "distance": dist, "species": self.t3_ds["2"].specie.get()}

            with open(f"{path}/{i}.pkl", 'wb') as f:
                pickle.dump(dictionary, f)


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
