import gc
import logging
from pathlib import Path
from typing import Literal

import dearpygui.dearpygui as dpg
from attrs import define, field

from src.utils import Project, loading_indicator, log_exec_time

logger = logging.getLogger(__name__)


TOOLTIP_DELAY_SEC = 0.1
LABEL_PAD = 23


@define(repr=False, eq=False)
class UI:
    project: Project = field(init=False)
    window: Literal["primary"] = field(init=False, default="primary")
    sidebar_width: Literal[350] = 350
    global_theme: int = field(init=False, default=0)
    button_theme: int = field(init=False, default=0)

    def __attrs_post_init__(self):
        dpg.create_context()
        dpg.create_viewport(title="hsistat", width=1920, height=1080, vsync=True)
        dpg.configure_app(wait_for_input=False)
        self.setup_themes()
        self.bind_themes()
        self.setup_handler_registries()
        self.setup_layout()
        self.bind_item_handlers()

    def start(self, dev=False):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_vsync(True)
        dpg.set_primary_window(self.window, True)
        try:
            if dev:
                dpg.set_frame_callback(1, self.setup_dev)
            dpg.start_dearpygui()
        except Exception as e:
            logger.fatal(e)
        finally:
            self.stop()

    def stop(self):
        dpg.stop_dearpygui()
        dpg.destroy_context()

    def setup_dev(self):
        dpg.set_value(
            "project_directory",
            "/home/puglet5/Documents/PROJ/libsthing/src/sandbox/libs/GR",
        )
        self.setup_project()

    def setup_themes(self):
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_ButtonTextAlign, 0.5, category=dpg.mvThemeCat_Core
                )

        with dpg.theme() as self.button_theme:
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(
                    dpg.mvThemeCol_Button,
                    (37, 37, 38, -255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonActive,
                    (37, 37, 38, -255),
                    category=dpg.mvThemeCat_Core,
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_ButtonHovered,
                    (37, 37, 38, -255),
                    category=dpg.mvThemeCat_Core,
                )

    def bind_themes(self):
        dpg.bind_theme(self.global_theme)

    def setup_handler_registries(self):
        with dpg.handler_registry():
            dpg.add_key_down_handler(dpg.mvKey_Control, callback=self.on_key_ctrl)

    def on_key_ctrl(self):
        if dpg.is_key_pressed(dpg.mvKey_Q):
            dpg.stop_dearpygui()
            dpg.destroy_context()

    def bind_item_handlers(self):
        pass

    def clear_plots(self):
        dpg.delete_item("libs_y_axis", children_only=True)

    def setup_project(self):
        if hasattr(self, "project"):
            del self.project
            self.clear_plots()
            gc.collect()

        dpg.hide_item("project_directory_error_message")

        try:
            dir_str = dpg.get_value("project_directory")
            if not dir_str:
                return

            project_dir = Path(dir_str)
            self.project = Project(project_dir)
        except ValueError:
            dpg.show_item("project_directory_error_message")
            return

        self.show_libs_plots()

    def directory_picker_callback(self, _sender, data):
        dpg.set_value("project_directory", data["file_path_name"])
        self.setup_project()

    def show_libs_plots(self):
        self.clear_plots()

        normalized = dpg.get_value("libs_normalized")
        corrected = dpg.get_value("libs_baseline_corrected")

        with dpg.mutex():
            for s in self.project.series:
                spectrum = s.averaged
                assert spectrum.raw_spectral_data is not None
                spectrum.process_spectral_data(normalized, corrected)
                x, y = spectrum.xy.tolist()

                dpg.add_line_series(x, y, parent="libs_y_axis", label=str(s.name))

            dpg.fit_axis_data("libs_x_axis")
            dpg.fit_axis_data("libs_y_axis")

    def setup_layout(self):
        with dpg.window(
            label="libsthing",
            tag=self.window,
            horizontal_scrollbar=False,
            no_scrollbar=True,
            min_size=[160, 90],
        ):
            with dpg.menu_bar(tag="menu_bar"):
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Open new project", shortcut="(Ctrl+O)")
                    dpg.add_menu_item(label="Save", shortcut="(Ctrl+S)")
                    dpg.add_menu_item(label="Save As", shortcut="(Ctrl+Shift+S)")
                    dpg.add_menu_item(label="Quit", shortcut="(Ctrl+Q)")

                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(
                        label="Preferences",
                        shortcut="(Ctrl+,)",
                    )

                with dpg.menu(label="Window"):
                    dpg.add_menu_item(
                        label="Wait For Input",
                        check=True,
                        tag="wait_for_input_menu",
                        shortcut="(Ctrl+Shift+Alt+W)",
                        callback=lambda s, a: dpg.configure_app(wait_for_input=a),
                    )
                    dpg.add_menu_item(
                        label="Toggle Fullscreen",
                        shortcut="(Win+F)",
                        callback=lambda _s, _d: dpg.toggle_viewport_fullscreen(),
                    )
                with dpg.menu(label="Tools"):
                    with dpg.menu(label="Developer"):
                        dpg.add_menu_item(
                            label="Show About",
                            callback=lambda: dpg.show_tool(dpg.mvTool_About),
                        )
                        dpg.add_menu_item(
                            label="Show Metrics",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Metrics),
                            shortcut="(Ctrl+Shift+Alt+M)",
                        )
                        dpg.add_menu_item(
                            label="Show Documentation",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Doc),
                        )
                        dpg.add_menu_item(
                            label="Show Debug",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Debug),
                        )
                        dpg.add_menu_item(
                            label="Show Style Editor",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Style),
                        )
                        dpg.add_menu_item(
                            label="Show Font Manager",
                            callback=lambda: dpg.show_tool(dpg.mvTool_Font),
                        )
                        dpg.add_menu_item(
                            label="Show Item Registry",
                            callback=lambda: dpg.show_tool(dpg.mvTool_ItemRegistry),
                        )

            with dpg.group(horizontal=True):
                with dpg.child_window(
                    border=False, width=self.sidebar_width, tag="sidebar"
                ):
                    dpg.add_progress_bar(tag="table_progress", width=-1, height=19)
                    with dpg.tooltip("table_progress", delay=TOOLTIP_DELAY_SEC):
                        dpg.add_text("Current operation progress")

                    with dpg.child_window(
                        label="Project",
                        width=-1,
                        height=200,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="Project", enabled=False):
                                pass

                        with dpg.group(horizontal=True):
                            dpg.add_text("Project directory".rjust(LABEL_PAD))
                            dpg.add_input_text(
                                tag="project_directory",
                                width=100,
                                callback=lambda s, d: self.setup_project(),
                                on_enter=True,
                            )
                            dpg.add_button(
                                label="Browse",
                                width=-1,
                                callback=lambda: dpg.show_item(
                                    "project_directory_picker"
                                ),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                default_value="Chosen directory is not valid!".rjust(
                                    LABEL_PAD
                                ),
                                tag="project_directory_error_message",
                                show=False,
                                color=(200, 20, 20, 255),
                            )

                    with dpg.child_window(
                        label="Plots",
                        width=-1,
                        height=-1,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="Plots", enabled=False):
                                pass

                        with dpg.group(horizontal=True):
                            dpg.add_text("Normalize".rjust(LABEL_PAD))
                            dpg.add_checkbox(
                                tag="libs_normalized",
                                default_value=False,
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text("Normalize to range:".rjust(LABEL_PAD))
                            with dpg.group(horizontal=False):
                                with dpg.group(horizontal=True):
                                    dpg.add_text("from".rjust(4))
                                    dpg.add_input_int(
                                        tag="from",
                                        width=-1,
                                        max_value=10000,
                                        min_value=1,
                                        min_clamped=True,
                                        max_clamped=True,
                                        default_value=1,
                                        on_enter=True,
                                    )
                                with dpg.group(horizontal=True):
                                    dpg.add_text("to".rjust(4))
                                    with dpg.tooltip(
                                        dpg.last_item(), delay=TOOLTIP_DELAY_SEC
                                    ):
                                        dpg.add_text(
                                            "Value of -1 indicates no upper limit (up to last row)",
                                            wrap=400,
                                        )
                                    dpg.add_input_int(
                                        tag="to",
                                        width=-1,
                                        max_value=10000,
                                        min_value=-1,
                                        default_value=-1,
                                        min_clamped=True,
                                        max_clamped=True,
                                        on_enter=True,
                                    )
                        with dpg.group(horizontal=True):
                            dpg.add_text("Remove baseline".rjust(LABEL_PAD))

                            dpg.add_combo(
                                tag="libs_baseline_corrected",
                                default_value="SNIP",
                                items=["None", "SNIP"],
                                width=-1,
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )

                        with dpg.group(horizontal=True):
                            dpg.add_text("Clip to zero".rjust(LABEL_PAD))
                            dpg.add_checkbox(
                                default_value=True,
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )

                with dpg.child_window(border=False, width=-1, tag="data"):
                    with dpg.plot(
                        tag="libs_plots",
                        crosshairs=True,
                        anti_aliased=True,
                        height=800,
                        width=-1,
                    ):
                        dpg.add_plot_legend(location=9)
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            label="Wavelength, nm",
                            tag="libs_x_axis",
                        )
                        dpg.add_plot_axis(
                            dpg.mvYAxis,
                            label="Line Intensity (arb. unit of energy flux)",
                            tag="libs_y_axis",
                        )

                    with dpg.plot(
                        tag="calibration_plots",
                        crosshairs=True,
                        anti_aliased=True,
                        height=-1,
                        width=-1,
                    ):
                        dpg.add_plot_legend(location=9)
                        dpg.add_plot_axis(
                            dpg.mvXAxis,
                            label="",
                            tag="calibration_x_axis",
                        )
                        dpg.add_plot_axis(
                            dpg.mvYAxis, label="", tag="calibration_y_axis"
                        )

        with dpg.file_dialog(
            modal=True,
            show=False,
            directory_selector=True,
            callback=self.directory_picker_callback,
            width=800,
            height=600,
            tag="project_directory_picker",
        ):
            pass

        with dpg.window(
            label="Settings",
            tag="settings_modal",
            show=False,
            no_move=True,
            no_collapse=True,
            modal=True,
            width=700,
            height=400,
            no_resize=True,
        ):
            pass

        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        with dpg.window(
            modal=True,
            no_background=True,
            no_move=True,
            no_scrollbar=True,
            no_title_bar=True,
            no_close=True,
            no_resize=True,
            tag="loading_indicator",
            show=False,
            pos=(w // 2 - 100, h // 2 - 100),
        ):
            dpg.add_loading_indicator(radius=20)
            dpg.add_button(
                label="Loading hyperspectral datapass",
                indent=30,
                tag="loading_indicator_message",
            )
            dpg.bind_item_theme(dpg.last_item(), self.button_theme)
