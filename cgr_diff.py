import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import tkinter as tk
from tkinter import filedialog, messagebox
import os

# -------------------- FONTS --------------------
# FONT_LG = ("Cambria", 16, "bold")
# FONT_MD = ("Cambria", 13, "bold")
FONT_SM = ("Cambria", 18)


# -------------------- THEME MANAGER --------------------
class ThemeManager:
    mode = "dark"  # "dark" or "light"

    @staticmethod
    def get_colors():
        """Return a dict of colors for the current theme."""
        if ThemeManager.mode == "dark":
            # very dark, almost black palette
            return dict(
                BG_MAIN="#242424",
                BG_PANEL="#2B2B2B",
                CARD="#333333",
                BORDER="#262626",
                BUTTON="#696969",
                BUTTON_HL="#3668A0",  # blue accent
                TEXT_MAIN="#f5f5f5",
                TEXT_MUTED="#a3a3a3",
            )
        else:
            # light theme approximation
            return dict(
                BG_MAIN="#EBEBEB",
                BG_PANEL="#DBDBDB",
                CARD="#989DA1",
                BORDER="#d4d4d8",
                BUTTON="#e5e5e5",
                BUTTON_HL="#2563eb",
                TEXT_MAIN="#111827",
                TEXT_MUTED="#6b7280",
            )


# -------------------- MAIN APP --------------------
class CGRApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # initial theme
        ctk.set_appearance_mode(ThemeManager.mode)
        self.colors = ThemeManager.get_colors()

        self.tab_names = ["CGR Analysis", "CGR Comparator", "Common Reference", "Multispecies Comparator"]
        self.active_tab = self.tab_names[0]

        self.title("CGR Diff")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{int(screen_width)}x{int(screen_height)}")
        self.configure(fg_color=self.colors["BG_MAIN"])

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        # ----------------------- placeholder for dynamic content -----------------------

        self.uploaded_files = []  # list of uploaded fasta files (full paths)
        self.file_cards = []  # list of card widgets corresponding to uploaded files

        self.uploaded_seq_lists_frame = None  # frame that holds the uploaded file entries
        self.selected_file_index = None  # index of currently selected file in self.uploaded_files (or None)

        # ----------------------- Build UI -----------------------
        self._build_top_nav()
        self._build_main_content()

    # --------------------------------------------------
    # UI REFRESH (used by theme toggle, tab switch, upload)
    # --------------------------------------------------
    def _refresh_ui(self, full_rebuild: bool = False):
        self.colors = ThemeManager.get_colors()
        self.configure(fg_color=self.colors["BG_MAIN"])

        if full_rebuild:
            for widget in self.winfo_children():
                widget.destroy()
            self._build_top_nav()
        else:
            self.winfo_children()[1].destroy()
        self._build_main_content()

    # --------------------------------------------------
    # NAV BAR
    # --------------------------------------------------
    def _build_top_nav(self):
        nav = ctk.CTkFrame(self, fg_color=self.colors["BG_PANEL"], corner_radius=100, border_width=1,
                           border_color=self.colors["BORDER"], height=60)
        nav.grid(row=0, column=0, sticky="nsew", padx=(5, 5), pady=(5, 5))
        nav.grid_columnconfigure(0, weight=1)

        # left nav buttons
        btn_frame = ctk.CTkFrame(nav, fg_color="transparent")
        btn_frame.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        def nav_button(text, tab_key, col):
            active = (self.active_tab == tab_key)
            btn = ctk.CTkButton(btn_frame, text=text, fg_color=self.colors["BUTTON_HL"] if active else "transparent",
                                hover_color=self.colors["BUTTON_HL"] if active else None, border_width=0,
                                corner_radius=100, font=FONT_SM, width=150, height=32,
                                text_color="white" if active else self.colors["TEXT_MUTED"],
                                command=lambda t=tab_key: self._set_tab(t))
            btn.grid(row=0, column=col, padx=(0, 10))
            return btn

        for tab in self.tab_names:
            col_index = self.tab_names.index(tab)
            nav_button(tab, tab, col_index)

        # theme toggle button (moon/sun)
        theme_icon = "🌙" if ThemeManager.mode == "light" else "☀️"
        theme_btn = ctk.CTkButton(nav, text=theme_icon, width=40, height=40, fg_color=self.colors["CARD"],
                                  hover_color=self.colors["BUTTON"], corner_radius=20, font=("Helvetica", 16),
                                  command=self._toggle_theme)
        theme_btn.grid(row=0, column=1, padx=20, sticky="e")

    # --------------------------------------------------
    # THEME TOGGLE
    # --------------------------------------------------
    def _toggle_theme(self):
        ThemeManager.mode = "light" if ThemeManager.mode == "dark" else "dark"
        ctk.set_appearance_mode(ThemeManager.mode)
        self._refresh_ui()

    # --------------------------------------------------
    # TAB SWITCH
    # --------------------------------------------------
    def _set_tab(self, tab_key: str):
        if tab_key == self.active_tab:
            return
        self.active_tab = tab_key
        self._refresh_ui()

    # --------------------------------------------------
    # MAIN CONTENT
    # --------------------------------------------------
    def _build_main_content(self):
        main = ctk.CTkFrame(self, fg_color=self.colors["BG_MAIN"], corner_radius=0)
        main.grid(row=1, column=0, sticky="nsew")

        if self.active_tab == "CGR Analysis":
            main.grid_columnconfigure(0, weight=0, minsize=320)  # left panel
            main.grid_columnconfigure(1, weight=1)  # right panel
            main.grid_rowconfigure(0, weight=1)

            self._build_cgr_analysis(main)
        else:
            main.grid_columnconfigure(0, weight=1)
            main.grid_rowconfigure(0, weight=1)

            placeholder = ctk.CTkFrame(main, fg_color=self.colors["BG_MAIN"])
            placeholder.grid(row=0, column=0, sticky="nsew")
            placeholder.grid_columnconfigure(0, weight=1)
            placeholder.grid_rowconfigure(0, weight=1)

            label = ctk.CTkLabel(placeholder, text=f"{self.active_tab} (empty)", text_color=self.colors["TEXT_MUTED"])
            label.grid(row=0, column=0)

    # --------------------------------------------------
    # LEFT PANEL of CGR Analysis
    # --------------------------------------------------
    def _build_cgr_analysis(self, parent):
        # ---------- Design left panel ----------
        left = ctk.CTkFrame(parent, fg_color=self.colors["BG_PANEL"], border_width=1,
                            border_color=self.colors["BORDER"], corner_radius=10, )
        left.grid(row=0, column=0, padx=(5, 5), pady=(5, 5), sticky="nsew")
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(1, weight=1)  # row 1 is list_frame
        left.grid_propagate(False)

        # top buttons
        top_btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        top_btn_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        top_btn_frame.grid_columnconfigure((0, 1), weight=1)

        search_btn = ctk.CTkButton(top_btn_frame, text="Search and Download", fg_color=self.colors["BUTTON_HL"],
                                   corner_radius=8, height=35, font=FONT_SM, text_color="white", )
        search_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        upload_btn = ctk.CTkButton(top_btn_frame, text="Upload", fg_color=self.colors["BUTTON_HL"],
                                   corner_radius=8, height=35, font=FONT_SM, text_color="white",
                                   command=self._upload_files)
        upload_btn.grid(row=1, column=0, sticky="ew")

        generate_btn = ctk.CTkButton(top_btn_frame, text="Generate", fg_color=self.colors["BUTTON_HL"],
                                     corner_radius=8, height=35, font=FONT_SM, text_color="white", )
        generate_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        # list of genomes
        list_frame = ctk.CTkFrame(left, fg_color=self.colors["CARD"], corner_radius=8, border_width=1,
                                  border_color=self.colors["BORDER"], )
        list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_columnconfigure(0, weight=1)
        list_frame.grid_rowconfigure(1, weight=1)

        label = ctk.CTkLabel(list_frame, text="List of available sequences:", font=FONT_SM,
                             text_color=self.colors["TEXT_MAIN"], anchor="w", )
        label.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 5))

        # ---------- SCROLLABLE REGION (both directions) ----------
        scroll_container = ctk.CTkFrame(list_frame, fg_color=self.colors["CARD"])
        scroll_container.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))
        scroll_container.grid_columnconfigure(0, weight=1)
        scroll_container.grid_rowconfigure(0, weight=1)

        # plain Tk Canvas for scrolling
        canvas = tk.Canvas(scroll_container, background=self.colors["BG_MAIN"], highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        # vertical scrollbar
        v_scroll = ctk.CTkScrollbar(scroll_container, orientation="vertical", command=canvas.yview, )
        v_scroll.grid(row=0, column=1, sticky="ns", padx=(4, 0))
        # horizontal scrollbar
        h_scroll = ctk.CTkScrollbar(scroll_container, orientation="horizontal", command=canvas.xview, )
        h_scroll.grid(row=1, column=0, sticky="ew", pady=(4, 0))
        canvas.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        # inner frame that actually holds the file entries
        self.uploaded_seq_lists_frame = ctk.CTkFrame(canvas, fg_color=self.colors["BG_MAIN"])
        canvas.create_window((0, 0), window=self.uploaded_seq_lists_frame, anchor="nw")

        def _on_inner_configure(event):
            # update scroll region to fit inner frame (both width and height)
            canvas.configure(scrollregion=canvas.bbox("all"))

        self.uploaded_seq_lists_frame.bind("<Configure>", _on_inner_configure)

        self._refresh_uploaded_file_list()

        # bottom buttons
        bottom_btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        bottom_btn_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        bottom_btn_frame.grid_columnconfigure((0, 1), weight=1)

        remove_btn = ctk.CTkButton(bottom_btn_frame, text="Remove", fg_color=self.colors["BUTTON_HL"],
                                   corner_radius=8, height=35, font=FONT_SM, text_color="white",
                                   command=self._remove_selected_file, )
        remove_btn.grid(row=1, column=0, sticky="ew")

        run_btn = ctk.CTkButton(bottom_btn_frame, text="Run Analysis", fg_color=self.colors["BUTTON_HL"],
                                corner_radius=8, height=35, font=FONT_SM, text_color="white",
                                command=self._run_analysis_selected_file, )
        run_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

    # --------------------------------------------------
    # UPLOAD HANDLER
    # --------------------------------------------------
    def _upload_files(self):
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

        if not self.uploaded_files:
            no_file_label = tk.Label(self.uploaded_seq_lists_frame, text="No files uploaded yet.",
                                     bg=self.colors["BG_MAIN"], fg=self.colors["TEXT_MUTED"])
            no_file_label.grid(row=0, column=0, padx=5, pady=5)
            self.selected_file_index = None
            return

        for i, path in enumerate(self.uploaded_files):
            fname = os.path.basename(path)

            # card for each file
            card = ctk.CTkFrame(self.uploaded_seq_lists_frame, fg_color=self.colors["BG_MAIN"])
            card.grid(row=i, column=0, padx=5, pady=5, sticky="w")

            # make everything inside the card clickable
            def _make_on_click(index):
                def _on_click(event=None):
                    return self._set_selected_uploaded(index)

                return _on_click

            on_click = _make_on_click(i)
            card.bind("<Button-1>", on_click)

            name_label = tk.Label(card, text=fname, bg=self.colors["BG_MAIN"], fg=self.colors["TEXT_MAIN"], anchor="w")
            name_label.grid(row=0, column=0, padx=(1, 1), pady=(2, 0), sticky="w")
            name_label.bind("<Button-1>", on_click)

            path_label = tk.Label(card, text=path, bg=self.colors["BG_MAIN"], fg=self.colors["TEXT_MUTED"], anchor="w")
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
                card.configure(fg_color=self.colors["BUTTON_HL"], corner_radius=0)
                for child in card.winfo_children():
                    if child.grid_info().get("row") == 0:
                        child.configure(bg=self.colors["BUTTON_HL"], fg="white")
                    else:
                        child.configure(bg=self.colors["BUTTON_HL"], fg=self.colors["TEXT_MUTED"])
            else:
                card.configure(fg_color=self.colors["BG_MAIN"])
                for child in card.winfo_children():
                    if child.grid_info().get("row") == 0:
                        child.configure(bg=self.colors["BG_MAIN"], fg=self.colors["TEXT_MAIN"])
                    else:
                        child.configure(bg=self.colors["BG_MAIN"], fg=self.colors["TEXT_MUTED"])

    def _remove_selected_file(self):
        if self.selected_file_index is None:
            messagebox.showinfo("No selection", "Please select a file to remove.")
            return

        removed_path = self.uploaded_files.pop(self.selected_file_index)  # remove from list
        self.selected_file_index = None  # reset selection
        self._refresh_uploaded_file_list()  # refresh GUI

    def _run_analysis_selected_file(self):
        pass


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app = CGRApp()
    app.mainloop()
