import gc
import logging
from operator import call
from pathlib import Path
from re import L
from typing import Literal

import dearpygui.dearpygui as dpg
import numpy as np
from attrs import define, field
from natsort import natsorted

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
        self.window_resize_callback()

    def setup_themes(self):
        with dpg.theme() as self.global_theme:
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvStyleVar_ButtonTextAlign, 0.5, category=dpg.mvThemeCat_Core
                )
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots
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
            dpg.add_key_down_handler(dpg.mvKey_Control, callback=self.on_key_ctrl_down)
            dpg.add_key_release_handler(
                dpg.mvKey_Control, callback=self.on_key_ctrl_release
            )

        with dpg.item_handler_registry(tag="window_resize_handler"):
            dpg.add_item_resize_handler(callback=self.window_resize_callback)

    def on_key_ctrl_release(self):
        dpg.configure_item("libs_plots", pan_button=dpg.mvMouseButton_Left)
        if not dpg.is_plot_queried("libs_plots"):
            dpg.set_value("plot_selected_region_text", "None")

    def on_key_ctrl_down(self):
        dpg.configure_item("libs_plots", pan_button=dpg.mvMouseButton_Middle)
        if dpg.is_key_pressed(dpg.mvKey_Q):
            dpg.stop_dearpygui()
            dpg.destroy_context()

    def bind_item_handlers(self):
        dpg.bind_item_handler_registry(self.window, "window_resize_handler")

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
        range_from = dpg.get_value("normalize_from")
        range_to = dpg.get_value("normalize_to")
        if range_to < range_from and range_to != -1:
            dpg.set_value("normalize_to", range_from)
            range_to = range_from

        normalized = dpg.get_value("libs_normalized")
        baseline_removal = dpg.get_value("libs_baseline_corrected")
        normalization_range = (
            dpg.get_value("normalize_from"),
            dpg.get_value("normalize_to"),
        )

        selected_series = [s for s in self.project.series.values() if s.selected]
        selected_series = natsorted(selected_series, key=lambda s: s.name)

        with dpg.mutex():
            for id, s in enumerate(selected_series):
                spectrum = s.averaged
                assert spectrum.raw_spectral_data is not None
                spectrum.process_spectral_data(
                    normalized,
                    normalization_range=normalization_range,
                    baseline_removal=baseline_removal,
                    baseline_clip=dpg.get_value("clip_baseline"),
                    baseline_params={
                        "max_half_window": dpg.get_value("max_half_window"),
                        "filter_order": int(dpg.get_value("filter_order")),
                    },
                )
                x, y = spectrum.xy.tolist()

                if s.color is None:
                    s.color = (
                        np.array(
                            dpg.sample_colormap(
                                dpg.mvPlotColormap_Spectral,
                                id / (len(self.project.series)),
                            )
                        )
                        * [255, 255, 255, 200]
                    ).tolist()

                with dpg.theme() as plot_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(
                            dpg.mvPlotCol_Line,
                            s.color,
                            category=dpg.mvThemeCat_Plots,
                        )

                if dpg.does_item_exist(f"series_plot_{s.id}"):
                    dpg.configure_item(f"series_plot_{s.id}", x=x, y=y)
                    dpg.bind_item_theme(f"series_plot_{s.id}", plot_theme)
                else:
                    dpg.add_line_series(
                        x,
                        y,
                        parent="libs_y_axis",
                        label=str(s.name),
                        tag=f"series_plot_{s.id}",
                    )
                    dpg.bind_item_theme(f"series_plot_{s.id}", plot_theme)

                self.project.plotted_series_ids.add(s.id)

            if dpg.get_value("fit_to_axes"):
                dpg.fit_axis_data("libs_x_axis")
                dpg.fit_axis_data("libs_y_axis")

            for s in self.project.series.values():
                if s.id in self.project.plotted_series_ids and not s.id in [
                    s.id for s in selected_series
                ]:
                    self.project.plotted_series_ids.discard(s.id)
                    dpg.delete_item(f"series_plot_{s.id}")

    def toggle_series_list(self, state):
        dpg.delete_item("series_list_wrapper", children_only=True)
        if state == "Select":
            for s in self.project.series.values():
                dpg.add_selectable(
                    label=str(s.name).rjust(LABEL_PAD + 5),
                    parent="series_list_wrapper",
                    default_value=s.selected,
                    user_data=s.id,
                    callback=lambda s, d: self.toggle_series_selection(s, d),
                )

        if state == "All":
            for s in self.project.series.values():
                s.selected = True
        self.show_libs_plots()

    def toggle_series_selection(self, sender, state: bool):
        id = dpg.get_item_user_data(sender)
        assert isinstance(id, str)

        self.project.series[id].selected = state

        self.show_libs_plots()
        self.refresh_fitting_windows()

    def perform_fit(self):
        dpg.show_item("fitting_results")

    def plot_query_callback(self, sender, data):
        region = f"{(data[0]):.2f}..{(data[1]):.2f} nm"
        dpg.set_value("plot_selected_region_text", region)

    def window_resize_callback(self, _sender=None, _data=None):
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        if dpg.does_item_exist("loading_indicator"):
            dpg.configure_item("loading_indicator", pos=(w // 2 - 100, h // 2 - 100))

    def toggle_fitting_windows(self, state: bool):
        if not state:
            dpg.delete_item("libs_plots", children_only=True, slot=0)

        else:
            selected_series = [s for s in self.project.series.values() if s.selected]
            selected_series = natsorted(selected_series, key=lambda s: s.name)

            with dpg.mutex():
                for id, s in enumerate(selected_series):
                    spectrum = s.averaged
                    x_threshold = dpg.get_value("fitting_windows_x_threshold")
                    y_threshold = dpg.get_value("fitting_windows_y_threshold")
                    threshold_type = dpg.get_value("fitting_windows_y_threshold_type")
                    spectrum.generate_fitting_windows(
                        x_threshold=x_threshold,
                        y_threshold=y_threshold,
                        threshold_type=threshold_type,
                    )
                    windows = spectrum.fitting_windows
                    x, _ = windows.T.tolist()

                    assert s.color

                    for i, e in enumerate(x):
                        dpg.add_drag_line(
                            color=s.color,
                            default_value=e,
                            parent="libs_plots",
                            label=f"{s.name}_fitting_window_{i}",
                            tag=f"{s.id}_fitting_window_{i}",
                        )

    def refresh_fitting_windows(self):
        if dpg.get_value("toggle_fitting_windows_checkbox"):
            self.toggle_fitting_windows(False)
            self.toggle_fitting_windows(True)

    def change_fitting_windows_threshold_type(self, t_type):
        if t_type == "Absolute":
            dpg.configure_item("fitting_windows_y_threshold", max_value=500)
        elif t_type == "Relative":
            dpg.configure_item("fitting_windows_y_threshold", max_value=0.1)
            if dpg.get_value("fitting_windows_y_threshold") > 0.1:
                dpg.set_value("fitting_windows_y_threshold", 0.1)

        self.refresh_fitting_windows()

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
                    dpg.add_progress_bar(tag="progress_bar", width=-1, height=19)
                    with dpg.tooltip("progress_bar", delay=TOOLTIP_DELAY_SEC):
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
                        height=400,
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
                            dpg.add_text("Normalize in range:".rjust(LABEL_PAD))
                            with dpg.group(horizontal=False):
                                with dpg.group(horizontal=True):
                                    dpg.add_text("from".rjust(4))
                                    dpg.add_input_float(
                                        tag="normalize_from",
                                        width=-1,
                                        max_value=10000,
                                        min_value=0,
                                        step_fast=20,
                                        format="%.1f",
                                        min_clamped=True,
                                        max_clamped=True,
                                        default_value=1,
                                        callback=lambda _s, _d: self.show_libs_plots(),
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
                                    dpg.add_input_float(
                                        tag="normalize_to",
                                        width=-1,
                                        format="%.1f",
                                        step_fast=20,
                                        max_value=10000,
                                        min_value=-1,
                                        default_value=-1,
                                        min_clamped=True,
                                        max_clamped=True,
                                        on_enter=True,
                                        callback=lambda _s, _d: self.show_libs_plots(),
                                    )

                        with dpg.group(horizontal=True):
                            dpg.add_text("Always fit to axes".rjust(LABEL_PAD))
                            dpg.add_checkbox(
                                tag="fit_to_axes",
                                default_value=True,
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )

                        with dpg.group(horizontal=True):
                            dpg.add_text("Remove baseline".rjust(LABEL_PAD))

                            dpg.add_combo(
                                tag="libs_baseline_corrected",
                                default_value="SNIP",
                                items=["None", "SNIP", "Adaptive minmax", "Polynomial"],
                                width=-1,
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )

                        with dpg.group(horizontal=True):
                            dpg.add_text("Clip to zero".rjust(LABEL_PAD))
                            dpg.add_checkbox(
                                default_value=True,
                                tag="clip_baseline",
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )

                        with dpg.group(horizontal=True):
                            dpg.add_text("Max half window".rjust(LABEL_PAD))
                            dpg.add_slider_int(
                                default_value=40,
                                width=-1,
                                tag="max_half_window",
                                min_value=2,
                                max_value=80,
                                clamped=True,
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )

                        with dpg.group(horizontal=True):
                            dpg.add_text("Filter order".rjust(LABEL_PAD))
                            dpg.add_combo(
                                items=["2", "4", "6", "8"],
                                default_value="2",
                                width=-1,
                                tag="filter_order",
                                callback=lambda _s, _d: self.show_libs_plots(),
                            )

                    with dpg.child_window(
                        label="Series",
                        width=-1,
                        height=300,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="Series", enabled=False):
                                pass

                        with dpg.group(horizontal=True):
                            dpg.add_text("Mode".rjust(LABEL_PAD))
                            dpg.add_combo(
                                items=["All", "Select", "Single"],
                                default_value="All",
                                width=-1,
                                callback=lambda s, d: self.toggle_series_list(d),
                            )

                        with dpg.group(horizontal=False, tag="series_list_wrapper"):
                            pass

                    with dpg.child_window(
                        label="Fitting",
                        width=-1,
                        height=-1,
                        menubar=True,
                        no_scrollbar=True,
                    ):
                        with dpg.menu_bar():
                            with dpg.menu(label="Fitting", enabled=False):
                                pass

                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                "Selected region:".rjust(LABEL_PAD),
                            )
                            dpg.add_text("None", tag="plot_selected_region_text")

                        with dpg.group(horizontal=True):
                            dpg.add_button(
                                label="Fit region",
                                callback=lambda s, d: self.perform_fit(),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                "Show fitting windows".rjust(LABEL_PAD),
                            )
                            dpg.add_checkbox(
                                default_value=False,
                                tag="toggle_fitting_windows_checkbox",
                                callback=lambda s, d: self.toggle_fitting_windows(d),
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                "X Window threshold".rjust(LABEL_PAD),
                            )
                            dpg.add_slider_float(
                                default_value=20,
                                min_value=0,
                                max_value=40,
                                format="%.2f",
                                clamped=True,
                                tag="fitting_windows_x_threshold",
                                callback=lambda s, d: self.refresh_fitting_windows(),
                                width=-1,
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text(
                                "Y Window threshold".rjust(LABEL_PAD),
                            )
                            dpg.add_slider_float(
                                default_value=0.001,
                                min_value=0,
                                max_value=0.2,
                                format="%.3f",
                                clamped=True,
                                tag="fitting_windows_y_threshold",
                                callback=lambda s, d: self.refresh_fitting_windows(),
                                width=-1,
                            )
                        with dpg.group(horizontal=True):
                            dpg.add_text("".rjust(LABEL_PAD))
                            dpg.add_combo(
                                tag="fitting_windows_y_threshold_type",
                                items=["Absolute", "Relative"],
                                default_value="Relative",
                                width=-1,
                                callback=lambda s, d: self.change_fitting_windows_threshold_type(
                                    d
                                ),
                            )

                with dpg.child_window(border=False, width=-1, tag="data"):
                    with dpg.group(horizontal=True):
                        with dpg.plot(
                            tag="libs_plots",
                            crosshairs=True,
                            anti_aliased=True,
                            query=True,
                            query_button=dpg.mvMouseButton_Left,
                            query_mod=1,
                            callback=self.plot_query_callback,
                            height=800,
                            width=-1,
                        ):
                            dpg.add_plot_legend(location=4)
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

        with dpg.window(
            label="Fitting results",
            tag="fitting_results",
            width=400,
            height=800,
            menubar=True,
            show=False,
            no_scrollbar=True,
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
