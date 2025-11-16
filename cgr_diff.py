import customtkinter as ctk
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

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
        # active tab
        self.active_tab = self.tab_names[0]

        self.title("CGR Diff")
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        self.geometry(f"{int(screen_width)}x{int(screen_height)}")
        self.configure(fg_color=self.colors["BG_MAIN"])

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._build_top_nav()
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
        # flip mode
        ThemeManager.mode = "light" if ThemeManager.mode == "dark" else "dark"

        # apply to CTk & colors
        ctk.set_appearance_mode(ThemeManager.mode)
        self.colors = ThemeManager.get_colors()
        self.configure(fg_color=self.colors["BG_MAIN"])

        # rebuild UI
        for widget in self.winfo_children():
            widget.destroy()
        self._build_top_nav()
        self._build_main_content()

    # --------------------------------------------------
    # TAB SWITCH
    # --------------------------------------------------
    def _set_tab(self, tab_key: str):
        if tab_key == self.active_tab:
            return
        self.active_tab = tab_key

        for widget in self.winfo_children():
            widget.destroy()
        self._build_top_nav()
        self._build_main_content()

    # --------------------------------------------------
    # MAIN CONTENT
    # --------------------------------------------------
    def _build_main_content(self):
        main = ctk.CTkFrame(self, fg_color=self.colors["BG_MAIN"], corner_radius=0)
        main.grid(row=1, column=0, sticky="nsew")

        # CGR Analysis has the full layout; other tabs are empty
        if self.active_tab == "CGR Analysis":
            main.grid_columnconfigure(0, weight=0, minsize=320)  # left panel
            main.grid_columnconfigure(1, weight=1)  # right panel
            main.grid_rowconfigure(0, weight=1)

            self._build_cgr_analysis(main)

            # self._build_cgr_analysis_left_panel(main)
            # self._build_cgr_analysis_right_panel(main)
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
        left.grid_rowconfigure(1, weight=1)

        # top buttons
        top_btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        top_btn_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="ew")
        top_btn_frame.grid_columnconfigure((0, 1), weight=1)

        search_btn = ctk.CTkButton(top_btn_frame, text="Search and Download", fg_color=self.colors["BUTTON_HL"],
                                   corner_radius=8, height=35, font=FONT_SM, text_color="white", )
        search_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 10))

        upload_btn = ctk.CTkButton(top_btn_frame, text="Upload", fg_color=self.colors["BUTTON_HL"],
                                   corner_radius=8, height=35, font=FONT_SM, text_color="white", )
        upload_btn.grid(row=1, column=0, sticky="ew")

        generate_btn = ctk.CTkButton(top_btn_frame, text="Generate", fg_color=self.colors["BUTTON_HL"],
                                     corner_radius=8, height=35, font=FONT_SM, text_color="white", )
        generate_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        # list of genomes
        list_frame = ctk.CTkFrame(left, fg_color=self.colors["CARD"], corner_radius=8, border_width=1,
                                  border_color=self.colors["BORDER"], )
        list_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        list_frame.grid_columnconfigure(0, weight=1)

        label = ctk.CTkLabel(list_frame, text="List of available Sequences:", font=FONT_SM,
                             text_color=self.colors["TEXT_MAIN"], anchor="w", )
        label.grid(row=0, column=0, sticky="ew", padx=10, pady=(5, 5))

        # genome_card = ctk.CTkFrame(
        #     list_frame,
        #     fg_color=self.colors["BG_MAIN"],
        #     corner_radius=6,
        # )
        # genome_card.grid(row=1, column=0, padx=10, pady=(4, 10), sticky="ew")
        # genome_card.grid_columnconfigure(0, weight=1)
        #
        # genome_name = ctk.CTkLabel(
        #     genome_card,
        #     text="CP068277.2",
        #     font=FONT_SM,
        #     text_color=self.colors["TEXT_MAIN"],
        #     anchor="w",
        # )
        # genome_name.grid(row=0, column=0, padx=10, pady=(6, 0), sticky="ew")
        #
        # genome_len = ctk.CTkLabel(
        #     genome_card,
        #     text="248,387,328 bp",
        #     font=("Helvetica", 10),
        #     text_color=self.colors["TEXT_MUTED"],
        #     anchor="w",
        # )
        # genome_len.grid(row=1, column=0, padx=10, pady=(0, 6), sticky="ew")

        # bottom buttons
        bottom_btn_frame = ctk.CTkFrame(left, fg_color="transparent")
        bottom_btn_frame.grid(row=3, column=0, padx=10, pady=(0, 10), sticky="ew")
        bottom_btn_frame.grid_columnconfigure((0, 1), weight=1)

        remove_btn = ctk.CTkButton(bottom_btn_frame, text="Remove", fg_color=self.colors["BUTTON_HL"],
                                   corner_radius=8, height=35, font=FONT_SM, text_color="white", )
        remove_btn.grid(row=1, column=0, sticky="ew")

        run_btn = ctk.CTkButton(bottom_btn_frame, text="Run Analysis", fg_color=self.colors["BUTTON_HL"],
                                corner_radius=8, height=35, font=FONT_SM, text_color="white", )
        run_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

    # # --------------------------------------------------
    # # RIGHT PANEL (only for CGR Analysis)
    # # --------------------------------------------------
    # def _build_right_panel(self, parent):
    #     right = ctk.CTkFrame(
    #         parent,
    #         fg_color="transparent",
    #     )
    #     right.grid(row=0, column=1, padx=(0, 10), pady=10, sticky="nsew")
    #     right.grid_rowconfigure(0, weight=3)
    #     right.grid_rowconfigure(1, weight=1)
    #     right.grid_columnconfigure(0, weight=1)


# -------------------- RUN APP --------------------
if __name__ == "__main__":
    app = CGRApp()
    app.mainloop()
