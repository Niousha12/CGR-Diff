import os
import pickle
import sys
import threading
import tkinter
import tkinter.messagebox
from functools import partial
import random

import customtkinter
from tkinter import filedialog, messagebox

import numpy as np
from Bio import Entrez
from PIL import Image

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib import pyplot as plt

from chaos_game_representation import CGR
from chromosomes_holder import ChromosomesHolder
from constants import DISTANCE_METRICS_LIST, RESOLUTION_DICT, BITS_DICT
from distances.distance_metrics import get_dist
from representative_selection import ChromosomeRepresentativeSelection
from sequence_generation.sampling import generate_kmers
from sequence_generation.sequence_generation import generate_dna_sequence

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class GUIDataStructure:
    def __init__(self):
        self.specie = customtkinter.StringVar()
        self.chromosome = customtkinter.StringVar()
        self.seq = ""
        self.start_seq = customtkinter.IntVar()
        self.start_txt = customtkinter.StringVar()
        self.end_seq = customtkinter.IntVar()
        self.end_txt = customtkinter.StringVar()
        self.annotation = customtkinter.StringVar()

    def invalidate_based_specie(self):
        self.chromosome.set("")
        self.seq = ""
        self.start_seq.set(0)
        self.end_seq.set(0)
        self.annotation.set("")

    def invalidate_based_chromosome(self):
        self.start_seq.set(0)
        self.end_seq.set(0)
        self.annotation.set("")


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.temp_output_path = self.resource_path(".gui_temp_outputs")
        # self.temp_output_path = "./.gui_temp_outputs"
        if not os.path.exists(self.temp_output_path):
            os.makedirs(self.temp_output_path)
        # self.assets_path = "./assets"
        self.assets_path = self.resource_path("assets")
        self.data_path = self.resource_path("Data")

        self.title("CGR-Diff.py")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{int(screen_width)}x{int(screen_height)}")
        # self.geometry(f"{2300}x{1500}")
        self.header_font = ('Cambria', 14, 'bold')

        # configure grid layout (4x4)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # create tabview
        self.tabview = customtkinter.CTkTabview(self)
        self.tabview.grid(padx=(20, 20), pady=(20, 20), sticky="nsew")
        tab_names = ["CGR Analysis", "CGR Comparator", "Common Reference", "Multispecies Comparator"]
        for tab_name in tab_names:
            self.tabview.add(tab_name)
        # start from tab 1
        self.tabview.set(tab_names[1])

        # ------- images --------
        prev_im = customtkinter.CTkImage(light_image=Image.open(f"{self.assets_path}/back_arrow.png"), size=(20, 20))
        next_im = customtkinter.CTkImage(light_image=Image.open(f"{self.assets_path}/next_arrow.png"), size=(20, 20))
        search_im = customtkinter.CTkImage(light_image=Image.open(f"{self.assets_path}/search.png"), size=(12, 12))
        upload_im = customtkinter.CTkImage(light_image=Image.open(f"{self.assets_path}/upload-sign.png"), size=(12, 12))

        # ------- variables --------
        values_list = ["2", "3", "4", "5", "6", "7", "8", "9"]
        PLOT_TYPE_LIST = ["Bar plot", "Histogram plot"]
        species_list = [str(f) for f in os.listdir(self.data_path) if os.path.isdir(os.path.join(self.data_path, f))]

        # common variables
        self.k_var = customtkinter.IntVar()
        self.dist_metric = customtkinter.StringVar()
        appearance = customtkinter.StringVar(value="Dark")
        self.plot_type_var = customtkinter.StringVar()

        # NCBI variables
        self.download_path = customtkinter.StringVar(value=str(self.resource_path("Data")))
        self.email_var = customtkinter.StringVar(value="")
        self.checkbox_frame = None
        self.id_map = {}
        self.checkbox_vars = {}  # dict to store which boxes are selected

        '''
            Second Page (CGR Comparator tab)
            Configuring 
        '''
        self.tabview.tab(tab_names[1]).grid_columnconfigure(0, weight=1)
        self.tabview.tab(tab_names[1]).grid_columnconfigure(1, weight=10)
        self.tabview.tab(tab_names[1]).grid_rowconfigure(0, weight=1)
        self.tabview.tab(tab_names[1]).grid_rowconfigure(1, weight=4)

        # Frames
        t2_config_frame = customtkinter.CTkFrame(self.tabview.tab(tab_names[1]), corner_radius=20,
                                                 border_color="#333333", border_width=2)
        t2_config_frame.grid(row=0, column=0, rowspan=2, sticky="ns")
        t2_slider_frame = customtkinter.CTkFrame(self.tabview.tab(tab_names[1]), corner_radius=20,
                                                 border_color="#333333", border_width=2)
        t2_slider_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        self.t2_display_frame = customtkinter.CTkFrame(self.tabview.tab(tab_names[1]), corner_radius=20,
                                                       fg_color="#707370")  # , width=600, height=200)
        self.t2_display_frame.grid(row=1, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")

        # Designing the config frame (F1)
        for row in range(10):  # Increased row count to account for empty rows
            t2_config_frame.grid_rowconfigure(row, weight=1)

        # Search button
        customtkinter.CTkButton(t2_config_frame, image=search_im, width=180, height=25, command=self.open_popup,
                                text="Search and Download Genomes").grid(row=0, columnspan=2, pady=(5, 0))
        # Creating frames for chromosome 1 and chromosome 2
        t2_chr_frame = customtkinter.CTkFrame(t2_config_frame, corner_radius=20)
        t2_chr_frame.grid(row=1, columnspan=2, padx=5, pady=(5, 5), sticky="ew")

        # Chromosomes Widget
        # get all the folders name in DATA path
        self.t2_ds = {'1': GUIDataStructure(), '2': GUIDataStructure()}
        self.t2_species_combobox = {}
        self.t2_chr_combobox = {}
        for i in range(2):
            customtkinter.CTkLabel(t2_chr_frame, text=f"Genome {i + 1}: ", font=self.header_font) \
                .grid(row=i * 3, column=0, sticky="w", padx=10, pady=(5, 0))

            # upload button
            customtkinter.CTkButton(t2_chr_frame, image=upload_im, text="", width=15, height=10,
                                    command=partial(self.t1_upload_event, f"{str(i + 1)}", None)) \
                .grid(row=i * 3, column=1, sticky="w", padx=10, pady=(5, 0))

            customtkinter.CTkLabel(t2_chr_frame, text="Species: ", font=self.header_font) \
                .grid(row=(i * 3) + 1, column=0, sticky="w", padx=10)
            customtkinter.CTkLabel(t2_chr_frame, text="Chromosome name: ", font=self.header_font) \
                .grid(row=(i * 3) + 1, column=1, sticky="w", padx=10)
            self.t2_species_combobox[f"{i + 1}"] = customtkinter.CTkComboBox(t2_chr_frame, values=species_list,
                                                                             width=100,
                                                                             command=partial(self.specie_change_event,
                                                                                             f"{i + 1}"))

            self.t2_species_combobox[f"{i + 1}"].grid(row=(i * 3) + 2, column=0, sticky="w", padx=10, pady=(0, 12))
            self.t2_species_combobox[f"{i + 1}"].set("")

            self.t2_chr_combobox[f"{i + 1}"] = customtkinter.CTkComboBox(t2_chr_frame, values=[],
                                                                         variable=self.t2_ds[str(i + 1)].chromosome,
                                                                         width=100,
                                                                         command=partial(self.chromosome_change_event,
                                                                                         f"{i + 1}"))
            self.t2_chr_combobox[f"{i + 1}"].grid(row=(i * 3) + 2, column=1, sticky="w", padx=10, pady=(0, 12))
            self.t2_chr_combobox[f"{i + 1}"].set("")

        # Radio Button (Window size)
        # Frame for window size settings
        t2_config_frame_color = t2_config_frame.cget("fg_color")
        t2_window_size_frame = customtkinter.CTkFrame(t2_config_frame, fg_color=t2_config_frame_color)
        t2_window_size_frame.grid(row=2, columnspan=2, padx=10, pady=5, sticky="w")

        customtkinter.CTkLabel(t2_window_size_frame, text="Segment Size:", font=self.header_font) \
            .grid(row=0, column=0, padx=10)
        self.window_s_toggle = tkinter.IntVar(value=0)
        t2_window_s_1 = customtkinter.CTkRadioButton(t2_window_size_frame, text="Variable",
                                                     variable=self.window_s_toggle,
                                                     value=0, command=self.window_size_toggle_event)
        t2_window_s_1.grid(row=1, column=0, padx=10, pady=5)
        t2_window_s_2 = customtkinter.CTkRadioButton(t2_window_size_frame, text="Fix",
                                                     variable=self.window_s_toggle,
                                                     value=1, command=self.window_size_toggle_event)
        t2_window_s_2.grid(row=1, column=1, padx=10, pady=5, sticky="w")
        self.window_s = tkinter.StringVar(value="")
        self.window_entry = customtkinter.CTkEntry(t2_window_size_frame, textvariable=self.window_s)
        self.window_entry.bind('<FocusOut>', partial(self.sequence_value_change, "0"))
        self.window_entry.bind('<Key-Return>', partial(self.sequence_value_change, "0"))
        self.window_entry.configure(state="disable")
        self.window_entry.grid(row=2, columnspan=2, padx=(10, 10), pady=(10, 10), sticky="ew")

        # k_mer combo box
        customtkinter.CTkLabel(t2_config_frame, text="k-mer: ", font=self.header_font) \
            .grid(row=3, column=0, padx=10, pady=(10, 10))
        k_mer_combobox = customtkinter.CTkComboBox(t2_config_frame, values=values_list, width=100,
                                                   state="normal", variable=self.k_var)
        k_mer_combobox.grid(row=3, column=1, sticky="w", pady=(10, 10), padx=10)

        # reverse complement and random
        t2_seq_rv = customtkinter.CTkFrame(t2_config_frame, fg_color=t2_config_frame_color)
        t2_seq_rv.grid(row=4, columnspan=2, padx=10, pady=5, sticky="ew")

        self.checkbox_RC = {}
        self.checkbox_Random = {}
        for i in range(2):
            seq_label = customtkinter.CTkLabel(t2_seq_rv, text=f'Sequence {i + 1} :', font=self.header_font)
            seq_label.grid(row=(i * 2), column=0, padx=10, pady=(5, 0), sticky="w")

            self.checkbox_RC[str(i + 1)] = customtkinter.CTkCheckBox(master=t2_seq_rv, text="Reverse Complement")
            self.checkbox_RC[str(i + 1)].grid(row=(i * 2), column=1, padx=10, pady=5, sticky="w")
            self.checkbox_Random[str(i + 1)] = customtkinter.CTkCheckBox(master=t2_seq_rv, text="Shuffle")
            self.checkbox_Random[str(i + 1)].grid(row=(i * 2) + 1, column=1, padx=10, pady=5, sticky="w")

        # Distance metrics
        customtkinter.CTkLabel(t2_config_frame, text="Distance\n Measure: ", font=self.header_font) \
            .grid(row=6, column=0, padx=10, pady=(10, 10))
        dist_metric_combobox = customtkinter.CTkComboBox(t2_config_frame, values=DISTANCE_METRICS_LIST,
                                                         width=120, variable=self.dist_metric)
        dist_metric_combobox.grid(row=6, column=1, sticky="w", padx=10, pady=(10, 10))
        dist_metric_combobox.set("DSSIM")

        # cgr/fcgr option
        # self.fcgr = customtkinter.IntVar(value=1)
        # switch = customtkinter.CTkSwitch(t2_config_frame, text=f"Frequency CGR", variable=self.fcgr)
        # switch.grid(row=7, columnspan=2, pady=(10, 10))

        # plot button
        self.t2_display_frame.grid_rowconfigure(0, weight=1)
        self.t2_display_frame.grid_columnconfigure(0, weight=1)
        t2_plot_button = customtkinter.CTkButton(t2_config_frame, text="Plot", width=120, command=self.t1_plot)
        t2_plot_button.grid(row=8, columnspan=2, pady=(10, 5))

        # Appearance mode
        customtkinter.CTkLabel(t2_config_frame, text="Theme: ", font=self.header_font) \
            .grid(row=9, column=0, padx=10, pady=(20, 10))
        appearance_mode = customtkinter.CTkOptionMenu(t2_config_frame, values=["Dark", "Light"], width=100,
                                                      variable=appearance, command=self.change_appearance_mode_event)
        appearance_mode.grid(row=9, column=1, sticky="w", pady=(20, 10))

        # First Sequence scale
        _pad_size = (20, 0)
        self.t2_parts_name_combobox = {}
        self.start_seq_scale = {}
        self.end_seq_scale = {}
        self.t2_start_seq_entry = {}
        self.t2_end_seq_entry = {}
        for i in range(2):
            customtkinter.CTkLabel(t2_slider_frame, text=f'Sequence {i + 1} :', font=self.header_font) \
                .grid(row=(i * 2), column=0, padx=10, pady=_pad_size)

            # Sequence part names combo box
            self.t2_parts_name_combobox[str(i + 1)] = \
                customtkinter.CTkComboBox(t2_slider_frame, width=100, values=[],
                                          command=partial(self.annotation_change_event, str(i + 1)),
                                          variable=self.t2_ds[str(i + 1)].annotation, state="disable")
            self.t2_parts_name_combobox[str(i + 1)].grid(row=(i * 2) + 1, column=0, padx=10, pady=_pad_size)

            customtkinter.CTkLabel(t2_slider_frame, text='Start').grid(row=(i * 2), column=1, padx=5, pady=_pad_size)
            customtkinter.CTkLabel(t2_slider_frame, text='End').grid(row=(i * 2) + 1, column=1, padx=5, pady=_pad_size)

            seq_length = len(self.t2_ds[str(i + 1)].seq)
            to_value = seq_length if seq_length > 0 else 1

            self.start_seq_scale[str(i + 1)] = customtkinter.CTkSlider(t2_slider_frame, from_=0, to=to_value,
                                                                       orientation="horizontal", width=700,
                                                                       variable=self.t2_ds[str(i + 1)].start_seq,
                                                                       command=partial(self.sequence_value_change,
                                                                                       str(i + 1)))
            self.start_seq_scale[str(i + 1)].set(0)
            self.scale_normal_color = self.start_seq_scale[str(i + 1)].cget("button_color")
            self.start_seq_scale[str(i + 1)].configure(state="disabled", button_color="#888888")
            self.start_seq_scale[str(i + 1)].grid(row=(i * 2), column=2, pady=_pad_size)

            self.t2_start_seq_entry[str(i + 1)] = customtkinter.CTkEntry(t2_slider_frame,
                                                                         textvariable=self.t2_ds[str(i + 1)].start_txt)
            self.t2_start_seq_entry[str(i + 1)].bind('<FocusOut>', partial(self.sequence_value_change, "3"))
            self.t2_start_seq_entry[str(i + 1)].bind('<Key-Return>', partial(self.sequence_value_change, "3"))
            self.t2_start_seq_entry[str(i + 1)].grid(row=(i * 2), column=3, padx=5, pady=_pad_size)
            seq_s_e_label = customtkinter.CTkLabel(t2_slider_frame, text='bp')
            seq_s_e_label.grid(row=(i * 2), column=4, pady=_pad_size)

            end_seq_length = self.t2_ds[str(i + 1)].end_seq.get()
            to_value = end_seq_length if end_seq_length > 0 else 1

            self.end_seq_scale[str(i + 1)] = customtkinter.CTkSlider(t2_slider_frame, from_=0, to=to_value,
                                                                     orientation="horizontal", width=700,
                                                                     variable=self.t2_ds[str(i + 1)].end_seq,
                                                                     command=partial(self.sequence_value_change,
                                                                                     str(i + 1)))
            self.end_seq_scale[str(i + 1)].set(0)
            self.end_seq_scale[str(i + 1)].configure(state="disabled", button_color="#888888")
            self.end_seq_scale[str(i + 1)].grid(row=(i * 2) + 1, column=2, pady=_pad_size)

            self.t2_end_seq_entry[str(i + 1)] = customtkinter.CTkEntry(t2_slider_frame,
                                                                       textvariable=self.t2_ds[str(i + 1)].end_txt)
            self.t2_end_seq_entry[str(i + 1)].bind('<FocusOut>', partial(self.sequence_value_change, "3"))
            self.t2_end_seq_entry[str(i + 1)].bind('<Key-Return>', partial(self.sequence_value_change, "3"))
            self.t2_end_seq_entry[str(i + 1)].grid(row=(i * 2) + 1, column=3, padx=5, pady=_pad_size)
            seq_s_e_label_d = customtkinter.CTkLabel(t2_slider_frame, text='bp')
            seq_s_e_label_d.grid(row=(i * 2) + 1, column=4, pady=_pad_size)

        '''
            Third Page (Common Reference)
            Configuring 
        '''
        self.tabview.tab(tab_names[2]).grid_columnconfigure(0, weight=1)
        self.tabview.tab(tab_names[2]).grid_columnconfigure(1, weight=10)

        self.tabview.tab(tab_names[2]).grid_rowconfigure(0, weight=1)
        self.tabview.tab(tab_names[2]).grid_rowconfigure(1, weight=20)
        self.tabview.tab(tab_names[2]).grid_rowconfigure(2, weight=50)
        self.tabview.tab(tab_names[2]).grid_rowconfigure(3, weight=1)

        # Frames
        t3_config_frame = customtkinter.CTkFrame(self.tabview.tab(tab_names[2]), corner_radius=20,
                                                 border_color="#333333", border_width=2)
        t3_config_frame.grid(row=0, column=0, rowspan=4, sticky="ns")
        self.t3_plot_frame = customtkinter.CTkFrame(self.tabview.tab(tab_names[2]), corner_radius=20, fg_color="white")
        self.t3_plot_frame.grid(row=1, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        t3_display_frame = customtkinter.CTkFrame(self.tabview.tab(tab_names[2]), corner_radius=20)
        t3_display_frame.grid(row=2, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        # frames in the display frame
        t3_display_frame.grid_rowconfigure(0, weight=1)
        t3_display_frame.grid_columnconfigure(0, weight=2)
        t3_display_frame.grid_columnconfigure(1, weight=8)
        self.t3_display_frame_1 = customtkinter.CTkFrame(t3_display_frame, corner_radius=20, fg_color="#707370")
        self.t3_display_frame_1.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        self.t3_display_frame_2 = customtkinter.CTkFrame(t3_display_frame, corner_radius=20, fg_color="#707370")
        self.t3_display_frame_2.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")

        self.t3_plot_frame.grid_rowconfigure(0, weight=1)
        self.t3_plot_frame.grid_columnconfigure(0, weight=1)

        self.t3_display_frame_1.grid_rowconfigure(0, weight=1)
        self.t3_display_frame_1.grid_columnconfigure(0, weight=1)
        self.t3_display_frame_2.grid_rowconfigure(0, weight=1)
        self.t3_display_frame_2.grid_columnconfigure(0, weight=1)

        # Designing the config frame (F3)
        for row in range(10):
            t3_config_frame.grid_rowconfigure(row, weight=1)

        # Search button
        search_button = customtkinter.CTkButton(t3_config_frame, image=search_im, width=180, height=25,
                                                text="Search and Download Genomes", command=self.open_popup)
        search_button.grid(row=0, columnspan=2, pady=(5, 0))

        self.t3_ds = {'1': GUIDataStructure(), '2': GUIDataStructure()}
        # Creating frame for reference sequence
        t3_chr_frame = customtkinter.CTkFrame(t3_config_frame, corner_radius=20)
        t3_chr_frame.grid(row=1, columnspan=2, padx=5, pady=5, sticky="ew")

        customtkinter.CTkLabel(t3_chr_frame, text=f"Reference: ", font=self.header_font) \
            .grid(row=0, column=0, sticky="w", padx=10, pady=(5, 0))
        # upload button
        upload_button = customtkinter.CTkButton(t3_chr_frame, image=upload_im, text="",
                                                command=partial(self.t3_upload_event, "1", None),
                                                width=15, height=10)
        upload_button.grid(row=0, column=1, sticky="w", padx=10, pady=(5, 0))
        # Button for synthetic sequence
        customtkinter.CTkButton(t3_chr_frame, text="Generate", command=self.t3_gen_synth_seq_event,
                                width=15, height=10).grid(row=0, column=1, sticky="w", padx=(50, 10), pady=(5, 0))
        customtkinter.CTkLabel(t3_chr_frame, text="Species: ", font=self.header_font) \
            .grid(row=1, column=0, sticky="w", padx=10)
        customtkinter.CTkLabel(t3_chr_frame, text="Chromosome name: ", font=self.header_font) \
            .grid(row=1, column=1, sticky="w", padx=10)
        self.t3_species_combobox_1 = customtkinter.CTkComboBox(t3_chr_frame, values=species_list, width=100,
                                                               command=partial(self.t3_specie_change_event, "1"))

        self.t3_species_combobox_1.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))
        self.t3_species_combobox_1.set("")

        self.t3_chr_combobox_1 = customtkinter.CTkComboBox(t3_chr_frame, values=[],
                                                           variable=self.t3_ds["1"].chromosome, width=100,
                                                           command=partial(self.t3_chromosome_change_event, "1"))
        self.t3_chr_combobox_1.grid(row=2, column=1, sticky="w", padx=10, pady=(0, 10))
        self.t3_chr_combobox_1.set("")

        # start and end
        customtkinter.CTkLabel(t3_chr_frame, text="Start: ", font=self.header_font).grid(row=3, column=0, sticky="w",
                                                                                         padx=10)
        customtkinter.CTkLabel(t3_chr_frame, text="End: ", font=self.header_font) \
            .grid(row=3, column=1, sticky="w", padx=10)

        self.t3_start_seq_entry = customtkinter.CTkEntry(t3_chr_frame, textvariable=self.t3_ds["1"].start_txt)
        self.t3_start_seq_entry.bind('<FocusOut>', partial(self.t3_entry_change))
        self.t3_start_seq_entry.bind('<Key-Return>', partial(self.t3_entry_change))
        self.t3_start_seq_entry.configure(state="disable")
        self.t3_start_seq_entry.grid(row=4, column=0, padx=10, pady=(0, 10))

        self.t3_end_seq_entry = customtkinter.CTkEntry(t3_chr_frame, textvariable=self.t3_ds["1"].end_txt)
        self.t3_end_seq_entry.bind('<FocusOut>', partial(self.t3_entry_change))
        self.t3_end_seq_entry.bind('<Key-Return>', partial(self.t3_entry_change))
        self.t3_end_seq_entry.configure(state="disable")
        self.t3_end_seq_entry.grid(row=4, column=1, padx=10, pady=(0, 10))

        customtkinter.CTkLabel(t3_chr_frame, text="Annotations: ", font=self.header_font) \
            .grid(row=5, column=0, sticky="w", padx=10, pady=(0, 10))
        self.t3_parts_name_combobox = customtkinter.CTkComboBox(t3_chr_frame, values=[], state="disable",
                                                                variable=self.t3_ds["1"].annotation, width=120,
                                                                command=partial(self.t3_annotation_change_event, "1"))
        self.t3_parts_name_combobox.grid(row=5, column=1, sticky="w", padx=10, pady=(0, 10))

        # Creating frame for the other sequence
        t3_chr_frame_2 = customtkinter.CTkFrame(t3_config_frame, corner_radius=20)
        t3_chr_frame_2.grid(row=2, columnspan=2, padx=5, pady=5, sticky="ew")

        customtkinter.CTkLabel(t3_chr_frame_2, text=f"Genome: ", font=self.header_font) \
            .grid(row=0, columnspan=2, sticky="w", padx=10, pady=(5, 0))
        upload_button = customtkinter.CTkButton(t3_chr_frame_2, image=upload_im, text="",
                                                command=partial(self.t3_upload_event, "2", None),
                                                width=15, height=10)
        upload_button.grid(row=0, column=1, sticky="w", padx=10, pady=(5, 0))
        customtkinter.CTkLabel(t3_chr_frame_2, text="Species: ", font=self.header_font) \
            .grid(row=1, column=0, sticky="w", padx=10)
        customtkinter.CTkLabel(t3_chr_frame_2, text="Chromosome name: ", font=self.header_font) \
            .grid(row=1, column=1, sticky="w", padx=10)
        self.t3_species_combobox_2 = customtkinter.CTkComboBox(t3_chr_frame_2, values=species_list, width=100,
                                                               command=partial(self.t3_specie_change_event, "2"))

        self.t3_species_combobox_2.grid(row=2, column=0, sticky="w", padx=10, pady=(0, 10))
        self.t3_species_combobox_2.set("")

        self.t3_chr_combobox_2 = customtkinter.CTkComboBox(t3_chr_frame_2, values=[],
                                                           variable=self.t3_ds["2"].chromosome, width=100,
                                                           command=partial(self.t3_chromosome_change_event, "2"))
        self.t3_chr_combobox_2.grid(row=2, column=1, sticky="w", padx=10, pady=(0, 10))
        self.t3_chr_combobox_2.set("")

        # Window size
        customtkinter.CTkLabel(t3_chr_frame_2, text="Segment Size:", font=self.header_font) \
            .grid(row=3, column=0, padx=10, pady=(0, 10))
        self.t3_window_s = tkinter.StringVar(value="")
        self.t3_window_entry = customtkinter.CTkEntry(t3_chr_frame_2, textvariable=self.t3_window_s)
        self.t3_window_entry.configure(state="disable")
        self.t3_window_entry.grid(row=3, column=1, pady=(0, 10), padx=10, sticky="w")

        # k_mer combo box
        customtkinter.CTkLabel(t3_config_frame, text="k-mer: ", font=self.header_font) \
            .grid(row=3, column=0, padx=10, pady=(10, 10))
        k_mer_combobox = customtkinter.CTkComboBox(t3_config_frame, values=values_list, width=100,
                                                   state="normal", variable=self.k_var)
        k_mer_combobox.grid(row=3, column=1, sticky="w", padx=10, pady=(10, 10))

        # Distance metrics
        customtkinter.CTkLabel(t3_config_frame, text="Distance\n Measure: ", font=self.header_font) \
            .grid(row=4, column=0, pady=(20, 5), padx=(5, 0))
        dist_metric_combobox = customtkinter.CTkComboBox(t3_config_frame, values=DISTANCE_METRICS_LIST,
                                                         width=120, variable=self.dist_metric)
        dist_metric_combobox.grid(row=4, column=1, pady=(20, 5), sticky="w")
        dist_metric_combobox.set("DSSIM")

        # plot type
        customtkinter.CTkLabel(t3_config_frame, text="Plot Type: ", font=self.header_font) \
            .grid(row=5, column=0, pady=(20, 5), padx=(5, 0))
        plot_type_combobox = customtkinter.CTkComboBox(t3_config_frame, values=PLOT_TYPE_LIST,
                                                       width=120, variable=self.plot_type_var)
        plot_type_combobox.grid(row=5, column=1, pady=(20, 5), sticky="w")
        plot_type_combobox.set("Bar plot")

        # switch = customtkinter.CTkSwitch(t3_config_frame, text=f"Frequency CGR", variable=self.fcgr)
        # switch.grid(row=6, columnspan=2, pady=(20, 5))

        # run button
        run_button = customtkinter.CTkButton(t3_config_frame, text="Run", command=partial(self.run_common_ref, None))
        run_button.grid(row=8, columnspan=2)  # , sticky="ns")

        # Appearance mode
        customtkinter.CTkLabel(t3_config_frame, text="Theme: ", font=self.header_font) \
            .grid(row=9, column=0, padx=10, pady=(20, 10))
        appearance_mode = customtkinter.CTkOptionMenu(t3_config_frame, values=["Dark", "Light"], width=100,
                                                      variable=appearance, command=self.change_appearance_mode_event)
        appearance_mode.grid(row=9, column=1, sticky="w", pady=(20, 10))

        # placing the progress bars
        self.t3_cgr_distance_history = None

        self.t3_progress_bar = customtkinter.CTkProgressBar(self.tabview.tab(tab_names[2]))
        self.t3_progress_bar.set(0)
        self.t3_progress_bar.grid(row=0, column=1, padx=(10, 10), pady=(10, 10), sticky="nsew")

        # placing the slider bar
        t3_changing_frame = customtkinter.CTkFrame(self.tabview.tab(tab_names[2]), corner_radius=20)
        t3_changing_frame.grid(row=3, column=1, sticky="nsew")

        # Designing the changing frame
        t3_changing_frame.grid_columnconfigure(0, weight=1)
        t3_changing_frame.grid_columnconfigure(1, weight=10)
        t3_changing_frame.grid_columnconfigure(2, weight=1)

        self.t3_pic_num = customtkinter.IntVar(value=0)
        self.t3_scale = customtkinter.CTkSlider(t3_changing_frame, from_=0, orientation=customtkinter.HORIZONTAL,
                                                variable=self.t3_pic_num,
                                                command=partial(self._change_images, self.t3_pic_num.get(), "t3"))
        self.t3_scale.grid(row=0, column=1, pady=(10, 10), sticky="nsew")

        # previous-next button
        customtkinter.CTkButton(t3_changing_frame, image=prev_im, text="", width=10,
                                command=partial(self.move_previous, "t3", None)).grid(row=0, column=0)

        customtkinter.CTkButton(t3_changing_frame, image=next_im, text="", width=10,
                                command=partial(self.move_next, "t3", None)).grid(row=0, column=2)

        self.after_idle(self.bring_to_front)

    def bring_to_front(self):
        """Ensure window gets focus and is brought to front on macOS."""
        self.lift()
        self.attributes('-topmost', True)
        self.after_idle(self.attributes, '-topmost', False)
        self.focus_force()

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
        popup = customtkinter.CTkToplevel(self)
        popup.title("Search and Download genomes from NCBI")
        popup.geometry(f"{popup_width}x{popup_height}+{x}+{y}")

        # Centered content in popup is in two frames
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_columnconfigure(1, weight=5)
        popup.grid_rowconfigure(0, weight=1)
        # Frames
        popup_f1 = customtkinter.CTkFrame(popup, corner_radius=10, border_color="#333333", border_width=2)
        popup_f1.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        popup_f1.grid_columnconfigure(0, weight=1)
        popup_f1.grid_rowconfigure(0, weight=1)
        popup_f1.grid_rowconfigure(1, weight=10)
        popup_f2 = customtkinter.CTkFrame(popup, corner_radius=10, border_color="#333333", border_width=2)
        popup_f2.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        popup_f2.grid_columnconfigure(0, weight=1)
        popup_f2.grid_rowconfigure(0, weight=1)
        popup_f2.grid_rowconfigure(1, weight=1)
        popup_f2.grid_rowconfigure(2, weight=10)
        popup_f2.grid_rowconfigure(3, weight=1)

        # Designing the first frame
        # Download path and Browse button in a frame
        download_frame = customtkinter.CTkFrame(popup_f1, fg_color=popup_f1.cget("fg_color"))
        download_frame.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=(5, 5))
        download_frame.grid_columnconfigure(0, weight=5)
        download_frame.grid_columnconfigure(1, weight=1)

        # Bring a list of species available (downloaded) in the folder selected
        file_scrollable_frame = customtkinter.CTkScrollableFrame(popup_f1)
        file_scrollable_frame.grid(row=1, column=0, sticky="nsew", padx=(10, 10), pady=(5, 5))
        # download path label and entry
        customtkinter.CTkLabel(download_frame, text="Enter download path:") \
            .grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(5, 0))
        download_entry = customtkinter.CTkEntry(download_frame, textvariable=self.download_path)
        self.display_downloaded_files(file_scrollable_frame, self.download_path.get())
        download_entry.bind('<FocusOut>', partial(self.display_downloaded_files, file_scrollable_frame))
        download_entry.bind('<Key-Return>', partial(self.display_downloaded_files, file_scrollable_frame))
        download_entry.grid(row=1, column=0, sticky="we", padx=(5, 0), pady=(5, 0))
        # Browse button
        browse_button = customtkinter.CTkButton(download_frame, text="Browse...", width=80,
                                                command=lambda: self.browse_folder(file_scrollable_frame))
        browse_button.grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(5, 0))

        # Designing the second frame
        # Enter email in a frame
        email_frame = customtkinter.CTkFrame(popup_f2, fg_color=popup_f2.cget("fg_color"))
        email_frame.grid(row=0, column=0, sticky="ew", padx=(5, 5), pady=(5, 5))
        email_frame.grid_columnconfigure(0, weight=1)
        customtkinter.CTkLabel(email_frame, text="Enter your email (required by NCBI):") \
            .grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(5, 0))
        email_entry = customtkinter.CTkEntry(email_frame, textvariable=self.email_var)
        email_entry.grid(row=1, column=0, sticky="we", padx=(5, 5), pady=(5, 0))

        # Search label and entry in a frame
        search_frame = customtkinter.CTkFrame(popup_f2, fg_color=popup_f2.cget("fg_color"))
        search_frame.grid(row=1, column=0, sticky="ew", padx=(5, 5), pady=(5, 5))
        search_frame.grid_columnconfigure(0, weight=10)
        search_frame.grid_columnconfigure(1, weight=1)

        customtkinter.CTkLabel(search_frame, text="Enter organism name:") \
            .grid(row=0, column=0, sticky="w", padx=(5, 0), pady=(5, 0))
        search_organism = customtkinter.StringVar(value="")
        search_entry = customtkinter.CTkEntry(search_frame, textvariable=search_organism)
        search_entry.grid(row=1, column=0, sticky="we", padx=(5, 0), pady=(5, 0))
        # # Show the search results
        # scrollable_frame = customtkinter.CTkScrollableFrame(popup_f2)
        # scrollable_frame.grid(row=2, column=0, sticky="nsew", padx=(10, 10), pady=(5, 5))
        scrollable_container = customtkinter.CTkFrame(popup_f2, fg_color=popup_f2.cget("fg_color"))
        scrollable_container.grid(row=2, column=0, sticky="nsew", padx=10, pady=5)
        scrollable_container.grid_rowconfigure(0, weight=1)
        scrollable_container.grid_columnconfigure(0, weight=1)
        # Create a Canvas with custom background
        canvas = tkinter.Canvas(scrollable_container, bg="#2b2b2b", highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")
        # Add scrollbars
        v_scrollbar = customtkinter.CTkScrollbar(scrollable_container, orientation="vertical", command=canvas.yview)
        v_scrollbar.grid(row=0, column=1, sticky="ns")
        h_scrollbar = customtkinter.CTkScrollbar(scrollable_container, orientation="horizontal", command=canvas.xview)
        h_scrollbar.grid(row=1, column=0, sticky="ew")
        canvas.configure(yscrollcommand=v_scrollbar.set, xscrollcommand=h_scrollbar.set)

        # Frame inside canvas to hold widgets
        self.checkbox_frame = customtkinter.CTkFrame(canvas, fg_color=popup_f2.cget("fg_color"))
        canvas.create_window((0, 0), window=self.checkbox_frame, anchor="nw")

        def configure_scroll_region(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.checkbox_frame.bind("<Configure>", configure_scroll_region)

        # search button
        search_button = customtkinter.CTkButton(search_frame, text="Search", width=50,
                                                command=lambda: self.search_ncbi(email_entry, search_organism))
        search_button.grid(row=1, column=1, sticky="w", padx=(5, 0), pady=(5, 0))

        # download button
        download_button = customtkinter.CTkButton(popup_f2, text="Download selected fasta",
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
            customtkinter.CTkLabel(frame, text="Path does not exist", text_color="red").pack(anchor="w", padx=5, pady=2)
            return

        # Filter only FASTA files
        all_files = os.listdir(path)
        fasta_files = [f for f in all_files if f.lower().endswith(('.fasta', '.fa', '.fna'))]

        if not fasta_files:
            customtkinter.CTkLabel(frame, text="No FASTA files found", text_color="red").pack(anchor="w", padx=5,
                                                                                              pady=2)
            return

        for file in sorted(fasta_files):
            entry = customtkinter.CTkEntry(frame, width=500)
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

                var = customtkinter.BooleanVar()
                cb = customtkinter.CTkCheckBox(self.checkbox_frame, text=title, variable=var)
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

    @staticmethod
    def resource_path(relative_path):
        """ Get absolute path to resource, works for dev and for PyInstaller """
        if getattr(sys, 'frozen', False):  # If running as a bundled .app
            base_path = sys._MEIPASS
        else:
            base_path = os.path.abspath(".")

        return os.path.join(base_path, relative_path)

    @staticmethod
    def change_appearance_mode_event(new_appearance_mode: str):
        customtkinter.set_appearance_mode(new_appearance_mode)

    @staticmethod
    def sync_text_vars(ds, sender, keep_annotation=False):
        ds[sender].start_txt.set(f"{ds[sender].start_seq.get()}")
        ds[sender].end_txt.set(f"{ds[sender].end_seq.get()}")
        if keep_annotation is False:
            ds[sender].annotation.set("")

    @staticmethod
    def reverse_sync_text_vars(ds, sender):
        ds[sender].start_seq.set(int(ds[sender].start_txt.get()))
        ds[sender].end_seq.set(int(ds[sender].end_txt.get()))
        ds[sender].annotation.set("")

    # '''events'''
    def t1_upload_event(self, sender, value):
        file_path = filedialog.askopenfilename()
        _, sequence = ChromosomesHolder.read_fasta(file_path)
        if len(sequence) > 0:
            self.t2_ds[sender].specie.set("Custom")
            self.t2_species_combobox[sender].set("Custom")
            self.t2_ds[sender].invalidate_based_specie()
            self.t2_chr_combobox[sender].configure(values=[])
            self.t2_parts_name_combobox[sender].configure(values=[])
            # clear window_s
            self.window_s_toggle.set(0)
            self.window_s.set("")
            self.window_entry.configure(state="disable")
            # end scales
            for key, value in self.end_seq_scale.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal", button_color=self.scale_normal_color)
            for key, value in self.t2_end_seq_entry.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal")
            # for key, value in self.t1_ds.items():
            self.sync_text_vars(self.t2_ds, sender, keep_annotation=False)

            self.t2_ds[sender].seq = sequence
            self.t2_ds[sender].end_seq.set(len(self.t2_ds[sender].seq))
            self.start_seq_scale[sender].configure(state="normal", button_color=self.scale_normal_color)
            self.start_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.end_seq_scale[sender].configure(state="normal", button_color=self.scale_normal_color)
            self.end_seq_scale[sender].set(0)
            self.end_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.end_seq_scale[sender].set(len(self.t2_ds[sender].seq))
            self.sync_text_vars(self.t2_ds, sender)

    def specie_change_event(self, sender, value):
        try:
            self.t2_species_combobox[sender].set(value)
            self.t2_ds[sender].specie.set(value)

            specie = self.t2_ds[sender].specie.get()
            self.t2_chr_combobox[sender].configure(
                values=ChromosomesHolder(specie, self.data_path).get_all_chromosomes_name(include_whole_genome=True))
            self.t2_ds[sender].invalidate_based_specie()

            # clear window_s
            self.window_s_toggle.set(0)
            # end scales
            for key, value in self.end_seq_scale.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal", button_color=self.scale_normal_color)
            for key, value in self.t2_end_seq_entry.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal")
            self.window_s.set("")
            self.window_entry.configure(state="disable")
            self.sync_text_vars(self.t2_ds, sender, keep_annotation=False)
            # for key, value in self.t1_ds.items():
            #     self.sync_text_vars(self.t1_ds, key, keep_annotation=False)

            self.start_seq_scale[sender].configure(state="disabled", button_color="#888888")
            self.end_seq_scale[sender].configure(state="disabled", button_color="#888888")

            # Empty the list of annotations
            self.t2_parts_name_combobox[sender].configure(values=[])
        except Exception as e:
            messagebox.showerror("Error", f"Error: {e}")

    def chromosome_change_event(self, sender, value):
        specie = self.t2_ds[sender].specie.get()
        chromosome = self.t2_ds[sender].chromosome.get()
        # set its sequence
        self.t2_ds[sender].seq = ChromosomesHolder(specie, self.data_path).get_chromosome_sequence(chromosome)
        # set the annotation combobox
        self.t2_parts_name_combobox[sender].configure(state="normal")
        self.t2_parts_name_combobox[sender].configure(
            values=ChromosomesHolder(specie, self.data_path).cytobands[chromosome])
        self.t2_ds[sender].invalidate_based_chromosome()
        # set start and end
        try:
            self.t2_ds[sender].end_seq.set(len(self.t2_ds[sender].seq))
            if len(self.t2_ds[sender].seq) > 0:
                self.start_seq_scale[sender].configure(state="normal", button_color=self.scale_normal_color)
                self.end_seq_scale[sender].configure(state="normal", button_color=self.scale_normal_color)
            else:
                self.start_seq_scale[sender].configure(state="disable", button_color="#888888")
                self.end_seq_scale[sender].configure(state="disable", button_color="#888888")
            self.start_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.end_seq_scale[sender].set(0)
            self.end_seq_scale[sender].configure(to=len(self.t2_ds[sender].seq))
            self.end_seq_scale[sender].set(len(self.t2_ds[sender].seq))
        except Exception as e:
            messagebox.showerror("Sequence is Empty!")

        self.sync_text_vars(self.t2_ds, sender, keep_annotation=False)
        # clear window_s
        self.window_s_toggle.set(0)
        self.window_s.set("")
        self.window_entry.configure(state="disable")
        # end scales
        for key, value in self.end_seq_scale.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal", button_color=self.scale_normal_color)
        for key, value in self.t2_end_seq_entry.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal")
        # for key, value in self.t1_ds.items():
        #     self.sync_text_vars(self.t1_ds, key, keep_annotation=False)

    def annotation_change_event(self, sender, value):
        annotation = self.t2_ds[sender].annotation.get()
        chromosome_name = self.t2_ds[sender].chromosome.get()
        annotation_info = ChromosomesHolder(self.t2_ds[sender].specie.get(), self.data_path).cytobands[chromosome_name][
            annotation]
        self.t2_ds[sender].start_seq.set(annotation_info.start)
        self.t2_ds[sender].end_seq.set(annotation_info.end)
        self.sync_text_vars(self.t2_ds, sender, keep_annotation=True)
        self.end_seq_scale[sender].configure(state="normal", button_color=self.scale_normal_color)

        # clear window_s
        self.window_s_toggle.set(0)
        self.window_s.set("")
        self.window_entry.configure(state="disable")
        # end scales
        for key, value in self.end_seq_scale.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal", button_color=self.scale_normal_color)
        for key, value in self.t2_end_seq_entry.items():
            if len(self.t2_ds[key].seq) > 0:
                value.configure(state="normal")

    def window_size_toggle_event(self, keep_annotation=False):
        if self.window_s_toggle.get() == 0:
            self.window_s.set("")
            self.window_entry.configure(state="disable")

            # end scales
            for key, value in self.end_seq_scale.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal", button_color=self.scale_normal_color)
            for key, value in self.t2_end_seq_entry.items():
                if len(self.t2_ds[key].seq) > 0:
                    value.configure(state="normal")
        else:
            self.window_entry.configure(state="normal")
            self.window_s.set("500000")

            # end scales
            for key, value in self.end_seq_scale.items():
                value.configure(state="disable", button_color="#888888")
            for key, value in self.t2_end_seq_entry.items():
                value.configure(state="disable")
            for key, value in self.t2_ds.items():
                if len(self.t2_ds[key].seq) > 0:
                    self.t2_ds[key].end_seq.set(self.t2_ds[key].start_seq.get() + int(self.window_s.get()))

        for key, value in self.t2_ds.items():
            self.sync_text_vars(self.t2_ds, key, keep_annotation)

    def sequence_value_change(self, sender, value):
        if sender == "0":  # Window size changed
            for key, value in self.t2_ds.items():
                self.t2_ds[key].end_seq.set(self.t2_ds[key].start_seq.get() + int(self.window_s.get()))
        elif sender in ["1", "2"]:  # Scale changed
            if self.window_s_toggle.get() == 1:
                self.t2_ds[sender].end_seq.set(self.t2_ds[sender].start_seq.get() + int(self.window_s.get()))
        elif sender in ["3"]:
            for key, value in self.t2_ds.items():
                self.reverse_sync_text_vars(self.t2_ds, key)
            if self.window_s_toggle.get() == 1:
                for key, value in self.t2_ds.items():
                    self.t2_ds[key].end_seq.set(self.t2_ds[key].start_seq.get() + int(self.window_s.get()))

        for key, value in self.t2_ds.items():
            self.sync_text_vars(self.t2_ds, key)

    def t1_plot(self):
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
                seq = ChromosomesHolder.get_reverse_complement(seq)
            if self.checkbox_Random[key].get():
                seq = list(seq)
                random.shuffle(seq)
                seq = ''.join(seq)
            cgr = CGR(seq, self.k_var.get())
            # if self.fcgr.get() == 1:
            fcgrs_dict[key]["(f)cgr"] = cgr.get_fcgr()
            # else:
            #     fcgrs_dict[key]["(f)cgr"] = cgr.get_cgr()

            fcgrs_dict[key]["chr_len"] = len(self.t2_ds[key].seq)
            fcgrs_dict[key]["b"] = self.t2_ds[key].start_seq.get()
            fcgrs_dict[key]["e"] = self.t2_ds[key].end_seq.get()
            fcgrs_dict[key]["species"] = self.t2_ds[key].specie.get()

        diff = fcgrs_dict["2"]["(f)cgr"] - fcgrs_dict["1"]["(f)cgr"]
        fcgrs_dict["diff"] = diff
        distance_value = get_dist(fcgrs_dict["1"]["(f)cgr"], fcgrs_dict["2"]["(f)cgr"], dist_m=self.dist_metric.get())
        fcgrs_dict["distance"] = distance_value

        # Visualize the FCGRs
        display_frame_color = self.t2_display_frame.cget("fg_color")
        fig = self.plot_fcgrs(fcgrs_dict, colormap=True, background_color=display_frame_color)

        # Clear the previous figure from the display frame if any
        for widget in self.t2_display_frame.winfo_children():
            widget.destroy()

        # Create a canvas and add the figure to it
        canvas = FigureCanvasTkAgg(fig, master=self.t2_display_frame)
        canvas.draw()

        # Set the canvas size explicitly
        canvas_width = self.t2_display_frame.cget("width")
        canvas_height = self.t2_display_frame.cget("height")
        canvas.get_tk_widget().config(width=canvas_width, height=canvas_height)

        # Use grid to place the canvas
        canvas.get_tk_widget().grid(row=0, column=0, padx=10, pady=10, sticky='nsew')
        plt.close()

    def plot_fcgrs(self, fcgrs, colormap=False, background_color=None, name="Sequence"):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))
        fig.subplots_adjust(top=0.85)
        extent = 0, 1, 0, 1

        if background_color is not None:
            fig.patch.set_facecolor(background_color)

        scale_1, scaling_1 = self.get_scaling(fcgrs["1"]["chr_len"])
        b1 = fcgrs["1"]["b"]
        e1 = fcgrs["1"]["e"]
        scale_2, scaling_2 = self.get_scaling(fcgrs["2"]["chr_len"])
        b2 = fcgrs["2"]["b"]
        e2 = fcgrs["2"]["e"]

        # plot the data on the subplots
        img1 = CGR.array2img(fcgrs["1"]["(f)cgr"], bits=8,  # BITS_DICT[dictionary["species"]]
                             resolution=RESOLUTION_DICT[self.k_var.get()])
        img1 = Image.fromarray(img1, 'L')
        ax1.imshow(img1, cmap='gray', extent=extent)  # Reds_r
        ax1.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax1.set_title(f'{name} 1\n{round(b1 / scale_1, 2)} - {round(e1 / scale_1, 2)} {scaling_1}')

        im2 = ax2.imshow(fcgrs['diff'], cmap='RdBu', norm=plt.Normalize(-100, 100), extent=extent)
        ax2.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax2.set_title(f'Difference\ndistance = {round(fcgrs["distance"], 4)}')

        img2 = CGR.array2img(fcgrs["2"]["(f)cgr"], bits=8,  # BITS_DICT[dictionary["species"]]
                             resolution=RESOLUTION_DICT[self.k_var.get()])
        img2 = Image.fromarray(img2, 'L')
        ax3.imshow(img2, cmap='gray', extent=extent)  # Blues_r
        ax3.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
        ax3.set_title(f'{name} 2\n{round(b2 / scale_2, 2)} - {round(e2 / scale_2, 2)} {scaling_2}')

        if colormap:
            fig.subplots_adjust(bottom=0.2)  # Adjust the bottom margin
            cbar_ax2 = fig.add_axes([0.36, 0.1, 0.3, 0.02])  # Adjust position as needed
            cbar = fig.colorbar(im2, cax=cbar_ax2, orientation='horizontal')
            # Red: Greater k-mer value in Sequence 1, Blue: Greater k-mer value in Sequence 2
            cbar.set_label(f'Red: Greater k-mer value in {name} 1 , Blue: Greater k-mer value in {name} 2', fontsize=10)
            cbar.ax.xaxis.set_label_position('top')  # Position label at top of colorbar
            cbar.ax.xaxis.labelpad = 5
            cbar.ax.tick_params(labelsize=8)
        # fig.subplots_adjust(bottom=0.0001)  # Adjust bottom margin
        return fig

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

    def _change_images(self, index, tab_name, value):
        # plot distance results bar and first index is red
        index = round(value) if value is not None else index
        self._plot_chart(index, tab_name)
        # Load and display the first image set in next plot
        self._plot_fcgrs(index, tab_name)

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

    def _plot_fcgrs(self, image_index, tab_name):
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


class GenerateSyntheticSequence(customtkinter.CTkToplevel):
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
        tabview = customtkinter.CTkTabview(self, width=w - 20, height=h - 20)
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
        t1_config = customtkinter.CTkFrame(tabview.tab(tab_names[0]), corner_radius=10, border_color="#333333",
                                           border_width=2)
        t1_config.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        t1_config.grid_columnconfigure(0, weight=1)
        t1_config.grid_columnconfigure(1, weight=1)
        t1_config.grid_columnconfigure(2, weight=1)

        self.t1_frame = customtkinter.CTkFrame(tabview.tab(tab_names[0]), corner_radius=10, border_color="#333333",
                                               border_width=2, fg_color="white")
        self.t1_frame.grid(row=0, column=1, padx=(5, 5), pady=(5, 5), sticky="nsew")
        self.t1_frame.grid_columnconfigure(0, weight=1)
        self.t1_frame.grid_rowconfigure(0, weight=1)

        # Design the configuration frame
        # k-mer size
        customtkinter.CTkLabel(t1_config, text="k-mer: ").grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        values_list = ["2", "3", "4", "5", "6"]
        self.t1_k_var = customtkinter.IntVar()
        k_mer_combobox = customtkinter.CTkComboBox(t1_config, values=values_list, width=80, variable=self.t1_k_var)
        k_mer_combobox.set("6")  # Default value
        k_mer_combobox.grid(row=0, column=1, sticky="w", padx=(0, 10), pady=(10, 0))

        # sequence length
        customtkinter.CTkLabel(t1_config, text="Sequence length: ").grid(row=1, column=0, sticky="w", padx=10,
                                                                         pady=(10, 0))
        self.seq_len = tkinter.StringVar(value="10000")  # Default value
        (customtkinter.CTkEntry(t1_config, textvariable=self.seq_len)
         .grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ew"))

        # entropy scaling factor
        customtkinter.CTkLabel(t1_config, text="Entropy scaling factor: ").grid(row=2, column=0, sticky="w", padx=10,
                                                                                pady=(10, 0))
        self.t1_r_var = customtkinter.DoubleVar(value=1.0)  # Default value
        (customtkinter.CTkSlider(t1_config, from_=0.25, to=1.0, variable=self.t1_r_var, width=150)
         .grid(row=2, column=1, padx=(0, 10), pady=(10, 0), sticky="ew"))
        self.t1_r_value_label = customtkinter.CTkLabel(t1_config, text=f"{self.t1_r_var.get():.2f}")
        self.t1_r_value_label.grid(row=2, column=2, padx=(0, 10), pady=(10, 0), sticky="w")
        self.t1_r_var.trace_add("write", self.update_r_label)
        # Generate button
        (customtkinter.CTkButton(t1_config, text="Generate", command=lambda: self.generate_sequence("t1"))
         .grid(row=3, column=0, columnspan=3, padx=10, pady=(10, 10)))

        # Save button
        (customtkinter.CTkButton(tabview.tab(tab_names[0]), text="Save Sequence", command=self.save_sequence)
         .grid(row=1, column=1, padx=(5, 5), pady=(5, 5)))

        '''
            Designing the second tab (2-mer method)
        '''
        tabview.tab(tab_names[1]).grid_columnconfigure(0, weight=1)
        tabview.tab(tab_names[1]).grid_columnconfigure(1, weight=5)
        tabview.tab(tab_names[1]).grid_rowconfigure(0, weight=50)
        tabview.tab(tab_names[1]).grid_rowconfigure(1, weight=1)
        # Frames
        t2_config = customtkinter.CTkFrame(tabview.tab(tab_names[1]), corner_radius=10, border_color="#333333",
                                           border_width=2)
        t2_config.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        t2_config.grid_columnconfigure(0, weight=0)
        t2_config.grid_columnconfigure(1, weight=1)
        t2_config.grid_columnconfigure(2, weight=0)

        self.t2_frame = customtkinter.CTkFrame(tabview.tab(tab_names[1]), corner_radius=10, border_color="#333333",
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
            customtkinter.CTkLabel(t2_config, text=f"{kmer}: ").grid(row=i, column=0, padx=(10, 0), pady=(padding, 0),
                                                                     sticky="w")
            # Create a slider
            var = customtkinter.DoubleVar(value=0.0)
            self.k_var_dict[kmer] = var
            r_slider = customtkinter.CTkSlider(t2_config, from_=-3, to=3, variable=var, width=150, height=14)
            r_slider.grid(row=i, column=1, padx=(10, 0), pady=(padding, 0), sticky="ew")
            # Create a label for the slider
            self.k_value_label_dict[kmer] = customtkinter.CTkLabel(t2_config, text="0.0000", width=60)
            self.k_value_label_dict[kmer].grid(row=i, column=2, padx=(10, 10), pady=(padding, 0), sticky="w")
            # Bind the slider to update the label
            try:
                var.trace_add("write", self.update_all_k_labels)
            except AttributeError:
                var.trace("w", self.update_all_k_labels)
            last_row = i + 1
        self.update_all_k_labels()

        # put sequence length
        seq_frame = customtkinter.CTkFrame(t2_config, fg_color="transparent")
        seq_frame.grid(row=last_row, column=0, columnspan=3, padx=(10, 10), pady=(5, 0), sticky="w")

        customtkinter.CTkLabel(seq_frame, text="Sequence length:").grid(row=0, column=0, padx=(0, 5), sticky="w")
        customtkinter.CTkEntry(seq_frame, textvariable=self.seq_len, width=150).grid(row=0, column=1, sticky="w")

        # Generate button
        (customtkinter.CTkButton(t2_config, text="Generate", command=lambda: self.generate_sequence("t2"))
         .grid(row=last_row + 1, column=0, columnspan=3, padx=10, pady=(10, 10)))

        # Save button
        (customtkinter.CTkButton(tabview.tab(tab_names[1]), text="Save Sequence", command=self.save_sequence)
         .grid(row=1, column=1, padx=(5, 5), pady=(5, 5)))

        '''
            Designing the third tab (k-mer method)
        '''
        tabview.tab(tab_names[2]).grid_columnconfigure(0, weight=1)
        tabview.tab(tab_names[2]).grid_columnconfigure(1, weight=10)
        tabview.tab(tab_names[2]).grid_rowconfigure(0, weight=50)
        tabview.tab(tab_names[2]).grid_rowconfigure(1, weight=1)
        # Frames
        t3_config = customtkinter.CTkFrame(tabview.tab(tab_names[2]), corner_radius=10, border_color="#333333",
                                           border_width=2)
        t3_config.grid(row=0, column=0, rowspan=2, padx=(5, 5), pady=(5, 5), sticky="nsew")
        t3_config.grid_columnconfigure(0, weight=1)
        t3_config.grid_columnconfigure(1, weight=1)
        for i in range(6):
            t3_config.grid_rowconfigure(i, weight=1)

        self.t3_frame = customtkinter.CTkFrame(tabview.tab(tab_names[2]), corner_radius=10, border_color="#333333",
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
        customtkinter.CTkLabel(t3_config, text="k-mer length: ").grid(row=0, column=0, padx=10, pady=(10, 0),
                                                                      sticky="w")
        k_mer_combobox = customtkinter.CTkComboBox(t3_config, values=values_list, width=80, variable=self.t1_k_var,
                                                   command=self.set_kmers_event)
        k_mer_combobox.set("2")
        k_mer_combobox.grid(row=0, column=1, padx=(0, 10), pady=(10, 0), sticky="w")

        # sequence length
        customtkinter.CTkLabel(t3_config, text="Sequence length: ").grid(row=1, column=0, sticky="w", padx=10,
                                                                         pady=(10, 0))
        (customtkinter.CTkEntry(t3_config, textvariable=self.seq_len)
         .grid(row=1, column=1, padx=(0, 10), pady=(10, 0), sticky="ew"))

        # k-mer entry + checkmark
        # frame
        entry_frame = customtkinter.CTkFrame(t3_config, fg_color="transparent")
        entry_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=(5, 5), sticky="ew")
        entry_frame.grid_columnconfigure(0, weight=0)
        entry_frame.grid_columnconfigure(1, weight=1)
        entry_frame.grid_columnconfigure(2, weight=0)
        entry_frame.grid_columnconfigure(3, weight=0)
        customtkinter.CTkLabel(entry_frame, text="k-mer").grid(row=0, column=0, sticky="w")

        # entry
        self.kmer_entry = customtkinter.CTkEntry(entry_frame, placeholder_text="e.g., ACAC", width=80)
        self.kmer_entry.grid(row=1, column=0, sticky="w", pady=(0, 5))
        self.kmer_entry.bind("<Return>", lambda _e: self._load_kmer_into_slider())
        # Slider
        self.t3_slider_var = customtkinter.DoubleVar(value=0.0)
        self.t3_kmer_slider = customtkinter.CTkSlider(entry_frame, from_=-3.0, to=3.0, variable=self.t3_slider_var)
        self.t3_kmer_slider.grid(row=1, column=1, sticky="ew", pady=(0, 5))
        self.t3_kmer_label = customtkinter.CTkLabel(entry_frame, text=f"{self.t3_slider_var.get():.4f}")
        self.t3_kmer_label.grid(row=1, column=2, sticky="w", pady=(0, 5))
        self.t3_slider_var.trace_add("write", self.update_t3_kmer_label)
        self.slider_normal_color = self.t3_kmer_slider.cget("button_color")
        self.t3_kmer_slider.configure(state="disabled",
                                      button_color="#888888")  # disable the slider until a k-mer is loaded
        # Save button
        self.save_btn = customtkinter.CTkButton(entry_frame, text="✓", width=10, command=self._refresh_summary)
        self.save_btn.grid(row=1, column=3, sticky="w", padx=(10, 0), pady=(0, 5))

        # summary textbox
        customtkinter.CTkLabel(t3_config, text="Summary").grid(row=3, column=0, padx=10, pady=(5, 0), sticky="w")
        self.summary_box = customtkinter.CTkTextbox(t3_config, height=160)
        self.summary_box.grid(row=4, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="nsew")

        # buttons
        btn_frame = customtkinter.CTkFrame(t3_config, fg_color="transparent")
        btn_frame.grid(row=5, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.reset_btn = customtkinter.CTkButton(btn_frame, text="Reset logits", command=self._reset_logits)
        self.reset_btn.pack(side="left")
        self.gen_btn = customtkinter.CTkButton(btn_frame, text="Generate", command=lambda: self.generate_sequence("t3"))
        self.gen_btn.pack(side="right")

        # Save button
        (customtkinter.CTkButton(tabview.tab(tab_names[2]), text="Save Sequence", command=self.save_sequence)
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
