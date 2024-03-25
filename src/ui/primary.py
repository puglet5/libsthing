import gc
import logging
import math
from pathlib import Path
from typing import Literal

import dearpygui.dearpygui as dpg
import numpy as np
import pandas as pd
from attrs import define, field
from natsort import index_natsorted, order_by_index

from settings import BaselineRemoval, Setting, Settings
from src.static.nist_processed.line_data import (
    element_plot_data,
    get_emission_data,
    select_wl_region,
)
from src.ui.loading_indicator import LoadingIndicator
from src.ui.peak_table import PeakTable
from src.ui.periodic_table import PeriodicTable
from src.ui.settings_window import SettingsWindow
from src.ui.simulation_window import SimulationWindow
from src.utils import (
    LABEL_PAD,
    SIDEBAR_WIDTH,
    TOOLTIP_DELAY_SEC,
    WINDOW_TAG,
    DPGItem,
    Project,
    Series,
    Window,
    flatten,
)

logger = logging.getLogger(__name__)


@define
class UI:
    project: Project = field(init=False)
    settings: Settings = field(init=False)
    periodic_table: PeriodicTable = field(init=False)
    simulation_window: SimulationWindow = field(init=False)
    settings_window: SettingsWindow = field(init=False)
    peak_table: PeakTable = field(init=False)
    loading_indicator: LoadingIndicator = field(init=False)
    global_theme: int = field(init=False, default=0)
    thumbnail_plot_theme: int = field(init=False, default=0)
    active_thumbnail_plot_theme: int = field(init=False, default=0)
    series_list_n_columns: int = field(init=False, default=5)

    def __attrs_post_init__(self):
        self.setup_settings()

        dpg.create_context()

        with dpg.font_registry():
            with dpg.font(
                Path("./src/fonts/basis33.ttf").absolute().as_posix(), 16
            ) as default_font:
                dpg.add_font_range(0x0370, 0x03FF)
                dpg.add_font_range(0x0400, 0x04FF)

        dpg.create_viewport(title="libsthing", width=1920, height=1080, vsync=True)
        dpg.configure_app(wait_for_input=False)

        self.setup_themes()
        self.bind_themes()
        self.setup_handler_registries()

        self.periodic_table = PeriodicTable()
        self.simulation_window = SimulationWindow()
        self.peak_table = PeakTable()
        self.settings_window = SettingsWindow()
        self.loading_indicator = LoadingIndicator()

        self.periodic_table.ui_parent = self

        self.setup_layout()
        self.bind_item_handlers()

        dpg.bind_font(default_font)

    def start(self, dev=False, debug=False):
        dpg.setup_dearpygui()
        dpg.show_viewport()
        dpg.set_viewport_vsync(False)
        dpg.set_primary_window(WINDOW_TAG, True)
        try:
            if dev:
                dpg.set_frame_callback(1, self.setup_dev)
                if debug:
                    dpg.configure_app(manual_callback_management=True)
                    while dpg.is_dearpygui_running():
                        jobs = dpg.get_callback_queue()  # retrieves and clears queue
                        dpg.run_callbacks(jobs)
                        dpg.render_dearpygui_frame()
                else:
                    dpg.start_dearpygui()
            else:
                dpg.start_dearpygui()
        except Exception as e:
            logger.fatal(e)
        finally:
            self.stop()

    def setup_settings(self):
        self.settings = Settings(
            project_common_x=Setting(
                "settings_project_common_x",
                True,
                None,
            ),
            sample_drop_first=Setting(
                "settings_sample_drop_first",
                10,
                self.setup_project,
            ),
            spectra_normalized=Setting(
                "settings_spectra_normalized",
                False,
                self.show_all_plots,
            ),
            spectra_fit_to_axes=Setting(
                "settings_spectra_fit_to_axes",
                False,
                self.show_libs_plots,
            ),
            normalized_from=Setting(
                "settings_normalized_from",
                0,
                self.show_libs_plots,
            ),
            normalized_to=Setting(
                "settings_normalized_to",
                -1,
                self.show_libs_plots,
            ),
            baseline_removal_method=Setting(
                "settings_baseline_removal_method",
                BaselineRemoval.SNIP,
                self.show_libs_plots,
            ),
            baseline_clipped_to_zero=Setting(
                "settings_baseline_clipped_to_zero",
                False,
                self.show_libs_plots,
            ),
            baseline_max_half_window=Setting(
                "settings_baseline_max_half_window",
                40,
                self.show_libs_plots,
            ),
            baseline_filter_order=Setting(
                "settings_baseline_filter_order",
                "2",
                self.show_libs_plots,
            ),
            min_peak_height=Setting(
                "settings_min_peak_height",
                0.01,
                self.refresh_all,
            ),
            min_peak_prominance=Setting(
                "settings_min_peak_prominance",
                0.01,
                self.refresh_all,
            ),
            peak_smoothing_sigma=Setting(
                "settings_peak_smoothing_sigma",
                1,
                self.refresh_all,
            ),
            selection_guides_shown=Setting(
                "settings_selection_guides_shown",
                False,
                self.toggle_selection_guides,
            ),
            region_subdivided=Setting(
                "settings_region_subdivided",
                True,
                self.refresh_fitting_windows,
            ),
            fitting_windows_shown=Setting(
                "settings_fitting_windows_shown",
                False,
                self.toggle_fitting_windows,
            ),
            peaks_shown=Setting(
                "settings_peaks_shown",
                False,
                self.toggle_peaks,
            ),
            fitting_x_threshold=Setting(
                "settings_fitting_x_threshold",
                8,
                self.refresh_fitting_windows,
            ),
            fitting_y_threshold=Setting(
                "settings_fitting_y_threshold",
                0.001,
                self.refresh_fitting_windows,
            ),
            fitting_y_threshold_type=Setting(
                "settings_fitting_y_threshold_type",
                "Relative",
                self.change_fitting_windows_threshold_type,
            ),
            fitting_max_iterations=Setting(
                "settings_fitting_max_iterations",
                2000,
                None,
            ),
            fitting_fit_display_mode=Setting(
                "settings_fitting_fit_display_mode",
                "Sum",
                self.show_fit_plots,
            ),
        )

    def reset_settings(self):
        self.settings.reset()
        self.show_libs_plots()
        self.refresh_all()

    def stop(self):
        dpg.destroy_context()
        dpg.stop_dearpygui()

    def setup_dev(self):
        dpg.set_value(
            "project_directory",
            "/home/puglet5/Sync/ITMO/libs/GR",
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

        with dpg.theme() as self.thumbnail_plot_theme:
            with dpg.theme_component(dpg.mvPlot):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_PlotPadding, 0, 0, category=dpg.mvThemeCat_Plots
                )

        with dpg.theme() as self.active_thumbnail_plot_theme:
            with dpg.theme_component(dpg.mvPlot):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_PlotPadding, 0, 0, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvThemeCol_Border, [0, 255, 0, 200], category=dpg.mvAll
                )

    def bind_themes(self):
        dpg.bind_theme(self.global_theme)

    def setup_handler_registries(self):
        with dpg.handler_registry():
            dpg.add_key_down_handler(dpg.mvKey_Control, callback=self.on_key_ctrl_down)
            dpg.add_key_release_handler(
                dpg.mvKey_Control, callback=self.on_key_ctrl_release
            )
            dpg.add_mouse_click_handler(
                dpg.mvMouseButton_Left,
                callback=lambda s, d: self.series_left_click(),
            )
            dpg.add_mouse_click_handler(
                dpg.mvMouseButton_Right,
                callback=lambda s, d: self.series_right_click(),
            )
            dpg.add_key_press_handler(
                dpg.mvKey_F11, callback=lambda: dpg.toggle_viewport_fullscreen()
            )

        with dpg.item_handler_registry(tag="window_resize_handler"):
            dpg.add_item_resize_handler(callback=self.window_resize_callback)

        with dpg.item_handler_registry(tag="fit_lmb_handler"):
            dpg.add_item_clicked_handler(
                dpg.mvMouseButton_Left,
                callback=lambda s, d: self.fits_lmb_callback(s, d),
            )

            dpg.add_item_clicked_handler(
                dpg.mvMouseButton_Right,
                callback=lambda s, d: self.fits_rmb_callback(s, d),
            )

    def on_key_ctrl_release(self):
        dpg.configure_item("libs_plots", pan_button=dpg.mvMouseButton_Left)

    def on_key_ctrl_down(self):
        dpg.configure_item("libs_plots", pan_button=dpg.mvMouseButton_Middle)
        if dpg.is_key_pressed(dpg.mvKey_Q):
            self.stop()
        if dpg.is_key_pressed(dpg.mvKey_P):
            self.periodic_table.toggle()
        if dpg.is_key_pressed(dpg.mvKey_Comma):
            self.settings_window.toggle()
        if dpg.is_key_pressed(dpg.mvKey_B):
            self.toggle_sidebar()
        if dpg.is_key_pressed(dpg.mvKey_N):
            self.toggle_botton_bar()

        if dpg.is_key_down(dpg.mvKey_Alt):
            if dpg.is_key_down(dpg.mvKey_Shift):
                if dpg.is_key_pressed(dpg.mvKey_M):
                    dpg.show_tool(dpg.mvTool_Metrics)
            elif dpg.is_key_pressed(dpg.mvKey_M):
                menubar_visible = dpg.get_item_configuration(WINDOW_TAG)["menubar"]
                dpg.configure_item(WINDOW_TAG, menubar=(not menubar_visible))

    def series_left_click(self):
        if not dpg.is_item_clicked("series_list_wrapper"):
            return

        series_rows = dpg.get_item_children("series_list_wrapper", slot=1)
        if series_rows is None:
            return

        # plot thumbnail clicked = active
        # verify that only one is clicked
        active_groups = 0
        for row in series_rows:
            series_groups = dpg.get_item_children(row, slot=1)
            if series_groups is None:
                continue

            for group in series_groups:
                if dpg.is_item_active(group):
                    active_groups += 1

        if active_groups != 1:
            return

        for row in series_rows:
            series_groups = dpg.get_item_children(row, slot=1)
            if series_groups is None:
                continue

            for group in series_groups:
                sid = dpg.get_item_user_data(group)
                if sid is None:
                    continue

                series = self.project.series[sid]
                group_children = dpg.get_item_children(group, slot=1)
                if group_children is None:
                    continue
                thumbnail = group_children[0]

                if dpg.is_key_down(dpg.mvKey_Shift):
                    if dpg.is_item_active(group):
                        if not series.selected:
                            dpg.bind_item_theme(
                                thumbnail, self.active_thumbnail_plot_theme
                            )
                            series.selected = True
                        else:
                            dpg.bind_item_theme(thumbnail, self.thumbnail_plot_theme)
                            series.selected = False
                else:
                    if dpg.is_item_active(group):
                        dpg.bind_item_theme(thumbnail, self.active_thumbnail_plot_theme)
                        series.selected = True
                    else:
                        dpg.bind_item_theme(thumbnail, self.thumbnail_plot_theme)
                        series.selected = False

        self.show_libs_plots()
        self.refresh_fit_results()
        self.show_fit_plots()

    def series_right_click(self):
        series_rows = dpg.get_item_children("series_list_wrapper", slot=1)
        if series_rows is None:
            return

        for row in series_rows:
            series_groups = dpg.get_item_children(row, slot=1)
            if series_groups is None:
                return

            for group in series_groups:
                sid = dpg.get_item_user_data(group)
                if sid is None:
                    return

                if dpg.is_item_hovered(group):
                    series = self.project.series[sid]
                    group_children = dpg.get_item_children(group, slot=1)
                    if group_children is None:
                        return

                    if not dpg.does_item_exist(f"{series.id}_rmb_window"):
                        with dpg.window(
                            pos=dpg.get_mouse_pos(),
                            no_title_bar=True,
                            no_move=True,
                            no_open_over_existing_popup=True,
                            popup=True,
                            menubar=True,
                            tag=f"{series.id}_rmb_window",
                        ):
                            with dpg.menu_bar():
                                dpg.add_menu(label=f"{series.name} info", enabled=False)

                            dpg.add_text(f"Directory: {series.directory}")
                            dpg.add_text(
                                f"{series.samples_total} samples with {series.spectra_total} spectra total"
                            )

                            dpg.add_separator()
                            dpg.add_text("Drop first n spectra in samples:")
                            dpg.add_slider_int(
                                min_value=0,
                                default_value=series.sample_drop_first,
                                max_value=len(series.samples[0].spectra) - 1,
                                clamped=True,
                                callback=lambda s, d: self.set_series_drop_first_n(
                                    series, d
                                ),
                            )

                            with dpg.group(horizontal=True):
                                dpg.add_button(
                                    label="I", tag=f"{series.id}_save_plot_png"
                                )
                                with dpg.tooltip(dpg.last_item()):
                                    dpg.add_text("Save as .png image")

                                dpg.add_button(
                                    label="E", tag=f"{series.id}_export_spectrum_csv"
                                )
                                with dpg.tooltip(dpg.last_item()):
                                    dpg.add_text("Export averaged spectrum")
                    else:
                        dpg.show_item(f"{series.id}_rmb_window")

    def set_series_drop_first_n(self, series: Series, drop_first_n: int):
        with dpg.mutex():
            series._averaged = None
            series.sample_drop_first = drop_first_n
            self.show_libs_plots()

    def bind_item_handlers(self):
        dpg.bind_item_handler_registry(WINDOW_TAG, "window_resize_handler")

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
            common_x = self.settings.project_common_x.value
            sample_drop_first = self.settings.sample_drop_first.value
            self.project = Project(
                project_dir, common_x=common_x, sample_drop_first=sample_drop_first
            )
        except ValueError:
            dpg.show_item("project_directory_error_message")
            return

        with dpg.mutex():
            for i, (sid, series) in enumerate(self.project.series.items()):
                if series.color is None:
                    series.color = (
                        (
                            np.array(
                                dpg.sample_colormap(
                                    dpg.mvPlotColormap_Spectral,
                                    i / (len(self.project.series)),
                                )
                            )
                            * [255, 255, 255, 255]
                        )
                        .astype(int)
                        .tolist()
                    )

        self.show_libs_plots()
        self.populate_series_list(initial=True)
        dpg.fit_axis_data("libs_x_axis")
        dpg.fit_axis_data("libs_y_axis")
        series_window_height = (
            math.ceil(((len(self.project.series) / self.series_list_n_columns))) * 100
        )
        dpg.configure_item(
            "sidebar_series",
            height=series_window_height,
        )
        self.refresh_fit_results()

    def directory_picker_callback(self, _sender: DPGItem, data):
        dpg.set_value("project_directory", data["file_path_name"])
        self.setup_project()

    def set_series_color(self, sender: DPGItem, color: list[int]):
        series_id = dpg.get_item_user_data(sender)
        assert series_id
        series = self.project.series[series_id]
        series.color = (np.array(color) * [255, 255, 255, 200]).astype(int).tolist()
        with dpg.mutex():
            self.populate_series_list(skip_plot_update=True)
            self.show_libs_plots()
            self.refresh_fit_results()
            self.show_fit_plots()

    def show_libs_plots(self):
        range_from = self.settings.normalized_from.value
        range_to = self.settings.normalized_to.value
        if range_to < range_from and range_to != -1:
            self.settings.normalized_to.set(range_from)
            range_to = range_from

        normalized = self.settings.spectra_normalized.value
        baseline_removal = self.settings.baseline_removal_method.value
        normalization_range = (
            self.settings.normalized_from.value,
            self.settings.normalized_to.value,
        )
        shift: float = dpg.get_value("libs_x_shift")

        with dpg.mutex():
            for i, series in enumerate(self.project.selected_series):
                spectrum = series.averaged
                assert spectrum.raw_spectral_data is not None
                assert series.color
                spectrum.process_spectral_data(
                    normalized=normalized,
                    shift=shift,
                    normalization_range=normalization_range,
                    baseline_removal=baseline_removal,
                    baseline_clip=self.settings.baseline_clipped_to_zero.value,
                    baseline_params={
                        "max_half_window": self.settings.baseline_max_half_window.value,
                        "filter_order": int(self.settings.baseline_filter_order.value),
                    },
                )
                x, y = spectrum.xy.tolist()

                with dpg.theme() as plot_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(
                            dpg.mvPlotCol_Line,
                            series.color,
                            category=dpg.mvThemeCat_Plots,
                        )

                if dpg.does_item_exist(f"series_plot_{series.id}"):
                    dpg.configure_item(f"series_plot_{series.id}", x=x, y=y)
                    dpg.bind_item_theme(f"series_plot_{series.id}", plot_theme)
                else:
                    dpg.add_line_series(
                        x,
                        y,
                        parent="libs_y_axis",
                        label=f"{series.name} avg.",
                        tag=f"series_plot_{series.id}",
                    )
                    dpg.bind_item_theme(f"series_plot_{series.id}", plot_theme)
                    dpg.add_color_edit(
                        default_value=series.color,
                        parent=f"series_plot_{series.id}",
                        user_data=series.id,
                        callback=self.set_series_color,
                        alpha_preview=dpg.mvColorEdit_AlphaPreviewHalf,
                        display_mode=dpg.mvColorEdit_rgb,
                        input_mode=dpg.mvColorEdit_input_rgb,
                    )

                self.project.plotted_series_ids.add(series.id)

            if self.settings.spectra_fit_to_axes.value:
                dpg.fit_axis_data("libs_x_axis")
                dpg.fit_axis_data("libs_y_axis")

            for series in self.project.series.values():
                if series.id in self.project.plotted_series_ids and not series.id in [
                    s.id for s in self.project.selected_series
                ]:
                    self.project.plotted_series_ids.discard(series.id)
                    dpg.delete_item(f"series_plot_{series.id}")

        line_series = dpg.get_item_children("libs_y_axis", slot=1)
        assert isinstance(line_series, list)
        line_series_labels = [dpg.get_item_label(s) for s in line_series]
        index = index_natsorted(line_series_labels)
        sorted_line_series: list[int] = order_by_index(line_series, index)  # type: ignore
        dpg.reorder_items("libs_y_axis", slot=1, new_order=sorted_line_series)

        self.refresh_all()

    def populate_series_list(self, skip_plot_update=False, initial=False):
        dpg.delete_item("series_list_wrapper", children_only=True)

        for i in range(len(self.project.series) // self.series_list_n_columns + 1):
            dpg.add_group(
                tag=f"series_row_{i}", horizontal=True, parent="series_list_wrapper"
            )

        for i, (s_id, series) in enumerate(self.project.series.items()):
            if initial:
                if i == 0:
                    series.selected = True
            with dpg.group(
                horizontal=False,
                parent=f"series_row_{i//self.series_list_n_columns}",
                user_data=s_id,
            ):
                assert series.color
                with dpg.theme() as plot_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(
                            dpg.mvPlotCol_Line,
                            series.color,
                            category=dpg.mvThemeCat_Plots,
                        )

                with dpg.plot(
                    width=60,
                    height=60,
                    no_box_select=True,
                    no_mouse_pos=True,
                    no_menus=True,
                    pan_button=-1,
                    no_title=True,
                    tag=f"{s_id}_thumbnail_plot",
                ):
                    dpg.add_plot_axis(
                        dpg.mvXAxis,
                        no_gridlines=True,
                        no_tick_labels=True,
                        no_tick_marks=True,
                        tag=f"{s_id}_thumbnail_plot_x_axis",
                    )
                    dpg.add_plot_axis(
                        dpg.mvYAxis,
                        no_gridlines=True,
                        no_tick_labels=True,
                        no_tick_marks=True,
                        tag=f"{s_id}_thumbnail_plot_y_axis",
                    )
                    dpg.add_line_series(
                        *series.averaged.xy.tolist(),
                        parent=dpg.last_item(),
                        tag=f"{s_id}_thumbnail_plot_series",
                    )
                    dpg.bind_item_theme(dpg.last_item(), plot_theme)

                    dpg.set_axis_limits(
                        f"{s_id}_thumbnail_plot_y_axis",
                        series.averaged.y.min(),
                        series.averaged.y.max(),
                    )
                    dpg.set_axis_limits(
                        f"{s_id}_thumbnail_plot_x_axis",
                        series.averaged.x.min(),
                        series.averaged.x.max(),
                    )

                dpg.add_text(f"{series.name}"[:8].center(8))
                with dpg.tooltip(delay=TOOLTIP_DELAY_SEC, parent=dpg.last_item()):
                    dpg.add_text(f"{series.name}")

                if series.selected:
                    dpg.bind_item_theme(
                        f"{s_id}_thumbnail_plot",
                        self.active_thumbnail_plot_theme,
                    )
                else:
                    dpg.bind_item_theme(
                        f"{s_id}_thumbnail_plot", self.thumbnail_plot_theme
                    )

        if not skip_plot_update:
            self.show_libs_plots()

    def perform_fit(self):
        for series in self.project.selected_series:
            spectrum = series.averaged
            x_threshold = self.settings.fitting_x_threshold.value
            y_threshold = self.settings.fitting_y_threshold.value
            threshold_type = self.settings.fitting_y_threshold_type.value
            subdivide = self.settings.region_subdivided.value
            if None not in self.project.selected_region:
                region: Window = self.project.selected_region  # type: ignore
            else:
                region = spectrum.x_limits

            if subdivide:
                spectrum.generate_fitting_windows(
                    region,
                    x_threshold=x_threshold,
                    y_threshold=y_threshold,
                    threshold_type=threshold_type,
                )
            else:
                spectrum.fitting_windows = [region]

            max_iterations = self.settings.fitting_max_iterations.value
            spectrum.fit_windows_parallel(
                spectrum.fitting_windows, max_iterations=max_iterations
            )

        self.refresh_fit_results()
        self.show_fit_plots()

    def window_resize_callback(self, _sender=None, _data=None):
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        if dpg.does_item_exist("loading_indicator"):
            dpg.configure_item("loading_indicator", pos=(w // 2 - 100, h // 2 - 100))

        dpg.configure_item("settings_window", pos=[w // 2 - 350, h // 2 - 200])

    def toggle_fitting_windows(self):
        state = self.settings.fitting_windows_shown.value

        with dpg.mutex():
            if not state:
                lines = dpg.get_item_children("libs_plots", slot=0)
                if lines is None:
                    return
                for peak in lines:
                    if dpg.get_item_type(peak) == "mvAppItemType::mvDragLine":
                        if dpg.get_item_alias(peak) not in (
                            "region_guide_start",
                            "region_guide_end",
                        ):
                            dpg.delete_item(peak)

            else:
                for series_id, series in enumerate(self.project.selected_series):
                    spectrum = series.averaged
                    if None not in self.project.selected_region:
                        region: Window = self.project.selected_region  # type: ignore
                    else:
                        region = spectrum.x_limits

                    x_threshold = self.settings.fitting_x_threshold.value
                    y_threshold = self.settings.fitting_y_threshold.value
                    threshold_type = self.settings.fitting_y_threshold_type.value
                    subdivide = self.settings.region_subdivided.value

                    if subdivide:
                        spectrum.generate_fitting_windows(
                            region,
                            x_threshold=x_threshold,
                            y_threshold=y_threshold,
                            threshold_type=threshold_type,
                        )
                    else:
                        spectrum.fitting_windows = [region]

                    if len(spectrum.fitting_windows) >= 1:
                        win_starts = [w[0] for w in spectrum.fitting_windows]
                        assert series.color

                        for i, e in enumerate(win_starts[1:]):
                            if dpg.does_item_exist(f"{series.id}_fitting_window_{i+1}"):
                                dpg.delete_item(f"{series.id}_fitting_window_{i+1}")

                            dpg.add_drag_line(
                                color=series.color,
                                default_value=e,
                                user_data=e,
                                callback=lambda s, d: self.lock_drag_item(s, d),
                                parent="libs_plots",
                                label=f"{series.name}_fitting_window_{i+1}",
                                tag=f"{series.id}_fitting_window_{i+1}",
                            )

    def toggle_peaks(self):
        state = self.settings.peaks_shown.value
        with dpg.mutex():
            if not state:
                peaks = dpg.get_item_children("libs_plots", slot=0)
                if peaks is None:
                    return
                for peak in peaks:
                    if dpg.get_item_type(peak) == "mvAppItemType::mvDragPoint":
                        dpg.delete_item(peak)

            else:
                for id, s in enumerate(self.project.selected_series):
                    spectrum = s.averaged
                    if None not in self.project.selected_region:
                        region: Window = self.project.selected_region  # type: ignore
                    else:
                        region = spectrum.x_limits

                    height = self.settings.min_peak_height.value
                    prominance = self.settings.min_peak_prominance.value
                    sigma = self.settings.peak_smoothing_sigma.value

                    spectrum.find_peaks(
                        region,
                        height=height,
                        prominance=prominance,
                        smoothing_sigma=sigma,
                    )
                    assert isinstance(spectrum.peaks, np.ndarray)
                    peaks = spectrum.peaks
                    assert s.color
                    for i, peak in enumerate(peaks):
                        dpg.add_drag_point(
                            color=s.color,
                            default_value=(peak[0], peak[1]),
                            parent="libs_plots",
                            label=f"{s.name} peak {i}",
                            tag=f"{s.id}_peak_{i}",
                            user_data=(peak[0], peak[1]),
                            callback=lambda s, d: self.lock_drag_item(s, d),
                        )

            self.refresh_peak_table()

    def lock_drag_item(self, sender: DPGItem, data):
        dpg.set_value(sender, (dpg.get_item_user_data(sender)))

    def refresh_fitting_windows(self):
        if self.settings.fitting_windows_shown.value:
            self.settings.fitting_windows_shown.set(False)
            self.settings.fitting_windows_shown.set(True)

    def refresh_peaks(self):
        if self.settings.peaks_shown.value:
            self.settings.peaks_shown.set(False)
            self.settings.peaks_shown.set(True)

    def change_fitting_windows_threshold_type(self):
        threshold_type = self.settings.fitting_y_threshold_type.value
        if threshold_type == "Absolute":
            dpg.configure_item(self.settings.fitting_y_threshold.tag, max_value=500)
        elif threshold_type == "Relative":
            dpg.configure_item(self.settings.fitting_y_threshold.tag, max_value=0.1)
            if self.settings.fitting_y_threshold.value > 0.1:
                self.settings.fitting_y_threshold.set(0.1)

        self.refresh_fitting_windows()

    def refresh_all(self):
        with dpg.mutex():
            self.refresh_peaks()
            self.refresh_fitting_windows()

    def refresh_selection_guides(self):
        if self.settings.selection_guides_shown.value:
            self.settings.selection_guides_shown.set(False)
            self.settings.selection_guides_shown.set(True)

    def refresh_peak_table(self):
        if not self.project.selected_series:
            return
        self.peak_table.series = self.project.selected_series
        self.peak_table.populate_table_regular()

    def handle_region_guide(self, guide, start_or_end: Literal[0, 1]):
        start_value = dpg.get_value("region_guide_start")
        end_value = dpg.get_value("region_guide_end")

        if start_value > end_value or end_value < start_value:
            if start_or_end == 0:
                dpg.set_value("region_guide_start", dpg.get_value("region_guide_end"))
            else:
                dpg.set_value("region_guide_end", dpg.get_value("region_guide_start"))

        self.project.selected_region[start_or_end] = dpg.get_value(guide)

        region = f"{(start_value):.2f}..{(end_value):.2f} nm"

        dpg.set_value("plot_selected_region_text", region)

        self.refresh_all()

    def toggle_selection_guides(self):
        state = self.settings.selection_guides_shown.value

        if not state:
            if dpg.does_item_exist("region_guide_start"):
                dpg.delete_item("region_guide_start")
            if dpg.does_item_exist("region_guide_end"):
                dpg.delete_item("region_guide_end")
            dpg.set_value("plot_selected_region_text", "None")
            if not dpg.is_plot_queried("libs_plots"):
                self.project.selected_region = [None, None]
            self.refresh_all()
            return

        if None in self.project.selected_region:
            region_start, region_end = self.project.selected_region
            if len(self.project.selected_series) != 0:
                limit_start, limit_end = self.project.selected_series[
                    0
                ].averaged.x_limits
            else:
                limit_start, limit_end = None, None

            start, end = region_start or limit_start, region_end or limit_end

            if start is None or end is None:
                return

            self.project.selected_region = [start, end]

        else:
            start, end = self.project.selected_region

        dpg.add_drag_line(
            color=[0, 255, 0, 255],
            thickness=2,
            default_value=start,
            parent="libs_plots",
            tag="region_guide_start",
            callback=lambda s: self.handle_region_guide(s, 0),
            label="Selection start",
        )

        dpg.add_drag_line(
            color=[0, 255, 0, 255],
            thickness=2,
            default_value=end,
            parent="libs_plots",
            callback=lambda s: self.handle_region_guide(s, 1),
            tag="region_guide_end",
            label="Selection end",
        )

        region = f"{(start):.2f}..{(end):.2f} nm"

        dpg.set_value("plot_selected_region_text", region)

        self.refresh_all()

    def plot_query_callback(self, sender: DPGItem, data):
        with dpg.mutex():
            region = list(data[0:2])
            if not region == self.project.selected_region:
                self.project.selected_region = region
                if not self.settings.selection_guides_shown.value:
                    self.settings.selection_guides_shown.set(True)

                self.refresh_selection_guides()

    def setup_layout(self):
        with dpg.window(
            label="libsthing",
            tag=WINDOW_TAG,
            horizontal_scrollbar=False,
            no_scrollbar=True,
            min_size=[160, 90],
        ):
            with dpg.menu_bar(tag="menu_bar"):
                with dpg.menu(label="File"):
                    dpg.add_menu_item(label="Open new project", shortcut="(Ctrl+O)")
                    dpg.add_menu_item(label="Save", shortcut="(Ctrl+S)")
                    dpg.add_menu_item(label="Save As", shortcut="(Ctrl+Shift+S)")
                    dpg.add_menu_item(
                        label="Quit", shortcut="(Ctrl+Q)", callback=self.stop
                    )

                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(
                        label="Reset all inputs", callback=lambda: self.reset_settings()
                    )
                    dpg.add_menu_item(
                        label="Preferences",
                        shortcut="(Ctrl+,)",
                        callback=self.settings_window.show,
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
                        shortcut="(F11)",
                        callback=lambda _s, _d: dpg.toggle_viewport_fullscreen(),
                    )
                    dpg.add_menu_item(
                        label="Docking",
                        check=True,
                        callback=lambda _s, d: dpg.configure_app(
                            docking=d, docking_space=False
                        ),
                    )
                    dpg.add_menu_item(
                        label="Toggle Sidebar",
                        shortcut="(Ctrl+B)",
                        callback=lambda _s, _d: self.toggle_sidebar(),
                    )

                    dpg.add_menu_item(
                        label="Toggle Bottom Bar",
                        shortcut="(Ctrl+N)",
                        callback=lambda _s, _d: self.toggle_botton_bar(),
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
                    dpg.add_menu_item(
                        label="Periodic table",
                        shortcut="(Ctrl+P)",
                        callback=self.periodic_table.show,
                    )

                    dpg.add_menu_item(
                        label="Browse NIST data",
                        callback=self.simulation_window.show,
                    )

            with dpg.group(horizontal=True):
                with dpg.child_window(border=False, width=SIDEBAR_WIDTH, tag="sidebar"):
                    with dpg.collapsing_header(label="Project", default_open=True):
                        with dpg.child_window(
                            width=-1,
                            height=100,
                            no_scrollbar=True,
                        ):
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

                            with dpg.group(horizontal=True):
                                dpg.add_text("Assume common x values".rjust(LABEL_PAD))
                                dpg.add_checkbox(
                                    **self.settings.project_common_x.as_dict
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Drop first n in samples".rjust(LABEL_PAD))
                                dpg.add_input_int(
                                    width=-1,
                                    on_enter=True,
                                    min_value=0,
                                    max_value=40,
                                    min_clamped=True,
                                    max_clamped=True,
                                    **self.settings.sample_drop_first.as_dict,
                                )

                    with dpg.collapsing_header(label="Plots", default_open=True):
                        with dpg.child_window(
                            width=-1,
                            height=350,
                            no_scrollbar=True,
                        ):
                            with dpg.group(horizontal=True):
                                dpg.add_text("Shift".rjust(LABEL_PAD))
                                dpg.add_slider_float(
                                    min_value=-2,
                                    max_value=2,
                                    clamped=True,
                                    width=-1,
                                    default_value=0.0,
                                    callback=self.show_libs_plots,
                                    tag="libs_x_shift",
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Normalize".rjust(LABEL_PAD))
                                dpg.add_checkbox(
                                    **self.settings.spectra_normalized.as_dict
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Normalize in range:".rjust(LABEL_PAD))
                                with dpg.group(horizontal=False):
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("from".rjust(4))
                                        dpg.add_input_float(
                                            width=-1,
                                            max_value=10000,
                                            min_value=0,
                                            step_fast=20,
                                            format="%.1f",
                                            min_clamped=True,
                                            max_clamped=True,
                                            on_enter=True,
                                            **self.settings.normalized_from.as_dict,
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
                                            width=-1,
                                            format="%.1f",
                                            step_fast=20,
                                            max_value=10000,
                                            min_value=-1,
                                            min_clamped=True,
                                            max_clamped=True,
                                            on_enter=True,
                                            **self.settings.normalized_to.as_dict,
                                        )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Always fit to axes".rjust(LABEL_PAD))
                                dpg.add_checkbox(
                                    **self.settings.spectra_fit_to_axes.as_dict
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Remove baseline".rjust(LABEL_PAD))

                                dpg.add_combo(
                                    items=[
                                        "None",
                                        "SNIP",
                                        "Adaptive minmax",
                                        "Polynomial",
                                    ],
                                    width=-1,
                                    **self.settings.baseline_removal_method.as_dict,
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Clip to zero".rjust(LABEL_PAD))
                                dpg.add_checkbox(
                                    **self.settings.baseline_clipped_to_zero.as_dict
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Max half window".rjust(LABEL_PAD))
                                dpg.add_slider_int(
                                    width=-1,
                                    min_value=2,
                                    max_value=80,
                                    clamped=True,
                                    **self.settings.baseline_max_half_window.as_dict,
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Filter order".rjust(LABEL_PAD))
                                dpg.add_combo(
                                    items=["2", "4", "6", "8"],
                                    width=-1,
                                    **self.settings.baseline_filter_order.as_dict,
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Min peak height".rjust(LABEL_PAD))
                                dpg.add_slider_float(
                                    min_value=0,
                                    max_value=0.2,
                                    format="%.3f",
                                    clamped=True,
                                    width=-1,
                                    **self.settings.min_peak_height.as_dict,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Min peak prominance".rjust(LABEL_PAD))
                                dpg.add_slider_float(
                                    min_value=0,
                                    max_value=0.2,
                                    format="%.3f",
                                    clamped=True,
                                    width=-1,
                                    **self.settings.min_peak_prominance.as_dict,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("Peak smoothing sigma".rjust(LABEL_PAD))
                                dpg.add_slider_float(
                                    min_value=0.0,
                                    max_value=2,
                                    format="%.3f",
                                    clamped=True,
                                    width=-1,
                                    **self.settings.peak_smoothing_sigma.as_dict,
                                )

                    with dpg.collapsing_header(label="Series", default_open=True):
                        with dpg.child_window(
                            width=-1,
                            height=100,
                            no_scrollbar=True,
                            tag="sidebar_series",
                        ):
                            with dpg.group(horizontal=False, tag="series_list_wrapper"):
                                pass

                    with dpg.collapsing_header(label="Fitting", default_open=True):
                        with dpg.child_window(
                            width=-1,
                            height=-1,
                            no_scrollbar=True,
                        ):
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    "Selected region:".rjust(LABEL_PAD),
                                )
                                dpg.add_checkbox(
                                    **self.settings.selection_guides_shown.as_dict
                                )
                                with dpg.tooltip(parent=dpg.last_item()):
                                    dpg.add_text("Show region guides")
                                dpg.add_text("None", tag="plot_selected_region_text")

                            with dpg.group(horizontal=True):
                                dpg.add_text("Subdivide region".rjust(LABEL_PAD))
                                dpg.add_checkbox(
                                    **self.settings.region_subdivided.as_dict
                                )
                                dpg.add_button(
                                    label="Fit",
                                    width=-1,
                                    callback=lambda s, d: self.perform_fit(),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    "Show fitting windows".rjust(LABEL_PAD),
                                )
                                dpg.add_checkbox(
                                    **self.settings.fitting_windows_shown.as_dict
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    "Show peaks".rjust(LABEL_PAD),
                                )
                                dpg.add_checkbox(**self.settings.peaks_shown.as_dict)
                                dpg.add_button(
                                    label="Peak table",
                                    width=-1,
                                    callback=lambda s, d: self.peak_table.show(),
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    "X Window threshold".rjust(LABEL_PAD),
                                )
                                dpg.add_slider_float(
                                    min_value=0,
                                    max_value=40,
                                    format="%.2f",
                                    clamped=True,
                                    width=-1,
                                    **self.settings.fitting_x_threshold.as_dict,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text(
                                    "Y Window threshold".rjust(LABEL_PAD),
                                )
                                dpg.add_slider_float(
                                    min_value=0,
                                    max_value=0.2,
                                    format="%.3f",
                                    clamped=True,
                                    width=-1,
                                    **self.settings.fitting_y_threshold.as_dict,
                                )
                            with dpg.group(horizontal=True):
                                dpg.add_text("".rjust(LABEL_PAD))
                                dpg.add_combo(
                                    items=["Absolute", "Relative"],
                                    width=-1,
                                    **self.settings.fitting_y_threshold_type.as_dict,
                                )

                            with dpg.group(horizontal=True):
                                dpg.add_text("Max fit iterations".rjust(LABEL_PAD))
                                dpg.add_input_int(
                                    min_value=-1,
                                    width=-1,
                                    min_clamped=True,
                                    **self.settings.fitting_max_iterations.as_dict,
                                )

                with dpg.group(horizontal=False):
                    with dpg.child_window(
                        border=False, width=-1, height=-SIDEBAR_WIDTH, tag="data"
                    ):
                        with dpg.group(horizontal=True):
                            with dpg.plot(
                                tag="libs_plots",
                                crosshairs=True,
                                anti_aliased=True,
                                query=True,
                                query_button=dpg.mvMouseButton_Left,
                                query_mod=1,
                                callback=self.plot_query_callback,
                                height=-1,
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

                    with dpg.tab_bar(tag="bottom_bar"):
                        with dpg.tab(label="Fit Results"):
                            with dpg.group(horizontal=True):
                                with dpg.child_window(
                                    width=SIDEBAR_WIDTH,
                                    height=-1,
                                    no_scrollbar=True,
                                    tag="fit_results_controls_window",
                                ):
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("  ".rjust(LABEL_PAD))
                                        dpg.add_button(label="All", width=75)
                                        with dpg.tooltip(
                                            parent=dpg.last_item(),
                                            delay=TOOLTIP_DELAY_SEC,
                                        ):
                                            dpg.add_text("Select all fits")
                                        dpg.add_button(label="None", width=-1)
                                        with dpg.tooltip(
                                            parent=dpg.last_item(),
                                            delay=TOOLTIP_DELAY_SEC,
                                        ):
                                            dpg.add_text("Deselect all fits")

                                    with dpg.group(horizontal=True):
                                        dpg.add_text(
                                            "Fit display mode".rjust(LABEL_PAD)
                                        )
                                        dpg.add_combo(
                                            items=["Sum", "Components", "Both"],
                                            width=-1,
                                            **self.settings.fitting_fit_display_mode.as_dict,
                                        )
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("Fit info".rjust(LABEL_PAD))
                                        dpg.add_combo(
                                            items=["Minimal", "Full"],
                                            default_value="Minimal",
                                            width=-1,
                                        )

                                    with dpg.group(horizontal=True):
                                        dpg.add_text(
                                            "Auto select new fits".rjust(LABEL_PAD)
                                        )
                                        dpg.add_checkbox(default_value=True)

                                    with dpg.group(horizontal=True):
                                        dpg.add_text(
                                            "Include in legend".rjust(LABEL_PAD)
                                        )
                                        dpg.add_checkbox(default_value=False)

                                    with dpg.group(horizontal=True):
                                        dpg.add_text(
                                            "Default fit color".rjust(LABEL_PAD)
                                        )
                                        dpg.add_combo(
                                            items=["Series", "Negative", "Custom"],
                                            default_value="Series",
                                            width=-1,
                                        )
                                    with dpg.group(horizontal=True):
                                        dpg.add_text("Fill fit curve".rjust(LABEL_PAD))
                                        dpg.add_checkbox(default_value=True)

                                with dpg.child_window(
                                    width=-1,
                                    height=-1,
                                    no_scrollbar=True,
                                    tag="fit_results_window",
                                ):
                                    ...

                        with dpg.tab(label="Calibration Plots"):
                            with dpg.child_window(
                                width=-1,
                                height=-1,
                                no_scrollbar=True,
                                tag="calibration_plots_window",
                            ):
                                dpg.add_plot(height=-1, width=-1)

        with dpg.file_dialog(
            modal=True,
            show=False,
            directory_selector=True,
            callback=self.directory_picker_callback,
            width=800,
            height=600,
            tag="project_directory_picker",
        ):
            ...

    def toggle_sidebar(self):
        if not dpg.is_item_shown("sidebar"):
            dpg.show_item("sidebar")
        else:
            dpg.hide_item("sidebar")

    def toggle_botton_bar(self):
        if not dpg.is_item_shown("bottom_bar"):
            dpg.configure_item("data", height=-SIDEBAR_WIDTH)
            dpg.show_item("bottom_bar")
        else:
            dpg.configure_item("data", height=-1)
            dpg.hide_item("bottom_bar")

        self.window_resize_callback()

    def refresh_fit_results(self):
        dpg.delete_item("fit_results_window", children_only=True)

        for series in self.project.selected_series:
            with dpg.group(
                horizontal=False, parent="fit_results_window", tag=f"{series.id}_fits"
            ):
                dpg.add_text(f"{series.name} fits")
                dpg.add_separator()
                if len(series.fits) == 0:
                    dpg.add_text("No fits performed.")

                fit_line_color = [255, 255, 255, 255] - np.array(series.color)
                fit_line_color[-1] = 255

                assert series.color
                with dpg.theme() as plot_theme:
                    with dpg.theme_component(dpg.mvLineSeries):
                        dpg.add_theme_color(
                            dpg.mvPlotCol_Line,
                            fit_line_color.tolist(),
                            category=dpg.mvThemeCat_Plots,
                        )

                with dpg.group(horizontal=True):
                    for fit_id, fit in series.fits.items():
                        fit_data = fit.data
                        with dpg.group(
                            horizontal=False,
                            tag=f"{fit_id}_thumbnail_plot_wrapper",
                            user_data={"fit": fit_id, "series": series.id},
                        ):
                            with dpg.plot(
                                width=60,
                                height=60,
                                no_box_select=True,
                                no_mouse_pos=True,
                                no_menus=True,
                                pan_button=-1,
                                no_title=True,
                                tag=f"{fit_id}_thumbnail_plot",
                            ):
                                dpg.add_plot_axis(
                                    dpg.mvXAxis,
                                    no_gridlines=True,
                                    no_tick_labels=True,
                                    no_tick_marks=True,
                                    tag=f"{fit_id}_thumbnail_plot_x_axis",
                                )
                                dpg.add_plot_axis(
                                    dpg.mvYAxis,
                                    no_gridlines=True,
                                    no_tick_labels=True,
                                    no_tick_marks=True,
                                    tag=f"{fit_id}_thumbnail_plot_y_axis",
                                )
                                dpg.add_line_series(
                                    *fit_data.xy.tolist(),
                                    parent=dpg.last_item(),
                                    tag=f"{fit_id}_thumbnail_plot_series",
                                )
                                dpg.bind_item_theme(dpg.last_item(), plot_theme)

                                dpg.set_axis_limits(
                                    f"{fit_id}_thumbnail_plot_y_axis",
                                    fit_data.y.min(),
                                    fit_data.y.max(),
                                )
                                dpg.set_axis_limits(
                                    f"{fit_id}_thumbnail_plot_x_axis",
                                    fit_data.x.min(),
                                    fit_data.x.max(),
                                )

                            x_bounds = f"{fit.x_bounds[0]:.0f}..{fit.x_bounds[1]:.0f}"
                            dpg.add_text(x_bounds.center(8), indent=2)

                            if fit.selected:
                                dpg.bind_item_theme(
                                    f"{fit_id}_thumbnail_plot",
                                    self.active_thumbnail_plot_theme,
                                )
                            else:
                                dpg.bind_item_theme(
                                    f"{fit_id}_thumbnail_plot",
                                    self.thumbnail_plot_theme,
                                )
                        dpg.bind_item_handler_registry(
                            f"{fit_id}_thumbnail_plot_wrapper", "fit_lmb_handler"
                        )

        self.peak_table.populate_table_fitted()

    def show_fit_plots_sum(self):
        with dpg.mutex():
            for s_id, series in self.project.series.items():
                fits = series.fits
                assert series.color

                if series.selected:
                    for fit_i, fit in fits.items():
                        if fit.selected:
                            x, y = fit.data.xy.tolist()
                            x_area, y_area = fit.data.area_fill_xy.tolist()

                            fit_line_color = [255, 255, 255, 255] - np.array(
                                series.color
                            )
                            fit_line_color[-1] = 255

                            fit_fill_color = [255, 255, 255, 255] - np.array(
                                series.color
                            )

                            fit_fill_color[-1] = 100

                            with dpg.theme() as plot_theme:
                                with dpg.theme_component(dpg.mvLineSeries):
                                    dpg.add_theme_color(
                                        dpg.mvPlotCol_Line,
                                        fit_line_color.tolist(),
                                        category=dpg.mvThemeCat_Plots,
                                    )
                            with dpg.theme() as area_theme:
                                with dpg.theme_component(dpg.mvAreaSeries):
                                    dpg.add_theme_color(
                                        dpg.mvPlotCol_Line,
                                        (255, 255, 255, 0),
                                        category=dpg.mvThemeCat_Plots,
                                    )

                            if dpg.does_item_exist(f"fit_plot_{fit.id}"):
                                dpg.configure_item(f"fit_plot_{fit.id}", x=x, y=y)
                                dpg.bind_item_theme(f"fit_plot_{fit.id}", plot_theme)
                            else:
                                x_bounds = (
                                    f"{fit.x_bounds[0]:.0f}..{fit.x_bounds[1]:.0f}"
                                )
                                dpg.add_line_series(
                                    x,
                                    y,
                                    parent="libs_y_axis",
                                    label=f"{series.name} fit {x_bounds}",
                                    tag=f"fit_plot_{fit.id}",
                                    user_data={"fit": fit.id},
                                )
                                dpg.add_area_series(
                                    x_area,
                                    y_area,
                                    parent="libs_y_axis",
                                    fill=fit_fill_color.tolist(),
                                    tag=f"fit_plot_area_{fit.id}",
                                    user_data={"fit": fit.id},
                                )
                                dpg.bind_item_theme(f"fit_plot_{fit.id}", plot_theme)
                                dpg.bind_item_theme(
                                    f"fit_plot_area_{fit.id}", area_theme
                                )
                        else:
                            if dpg.does_item_exist(f"fit_plot_{fit.id}"):
                                dpg.delete_item(f"fit_plot_{fit.id}")
                            if dpg.does_item_exist(f"fit_plot_area_{fit.id}"):
                                dpg.delete_item(f"fit_plot_area_{fit.id}")
                else:
                    for fit_i, fit in fits.items():
                        if dpg.does_item_exist(f"fit_plot_{fit.id}"):
                            dpg.delete_item(f"fit_plot_{fit.id}")
                        if dpg.does_item_exist(f"fit_plot_area_{fit.id}"):
                            dpg.delete_item(f"fit_plot_area_{fit.id}")

            # check for stale plots
            all_fit_ids = flatten(
                [list(s.fits.keys()) for s in self.project.series.values()]
            )
            y_axis_children = dpg.get_item_children("libs_y_axis", slot=1)
            if y_axis_children is None:
                return
            assert isinstance(y_axis_children, list)

            for plot in y_axis_children:
                if not dpg.does_item_exist(plot):
                    continue
                plot_user_data = dpg.get_item_user_data(plot)
                if not isinstance(plot_user_data, dict):
                    continue

                fit_id = plot_user_data.get("fit", None)
                if fit_id is None:
                    continue

                if fit_id not in all_fit_ids:
                    if dpg.does_item_exist(f"fit_plot_{fit_id}"):
                        dpg.delete_item(f"fit_plot_{fit_id}")
                    if dpg.does_item_exist(f"fit_plot_area_{fit_id}"):
                        dpg.delete_item(f"fit_plot_area_{fit_id}")

    def show_fit_plots_components(self):
        with dpg.mutex():
            for s_id, series in self.project.series.items():
                fits = series.fits
                assert series.color

                if series.selected:
                    for fit_i, (fit_id, fit) in enumerate(fits.items()):
                        if fit.selected:
                            for peak_i, (peak_id, peak) in enumerate(
                                fit.components.items()
                            ):
                                x, y = peak.fitted.xy.tolist()
                                x_area, y_area = peak.fitted.area_fill_xy.tolist()

                                fit_line_color = np.array(
                                    dpg.sample_colormap(
                                        dpg.mvPlotColormap_Viridis,
                                        peak_i / len(fit.components),
                                    )
                                ) * [255, 255, 255, 255]
                                fit_line_color[-1] = 255

                                fit_fill_color = fit_line_color

                                fit_fill_color[-1] = 100

                                with dpg.theme() as plot_theme:
                                    with dpg.theme_component(dpg.mvLineSeries):
                                        dpg.add_theme_color(
                                            dpg.mvPlotCol_Line,
                                            fit_line_color.tolist(),
                                            category=dpg.mvThemeCat_Plots,
                                        )
                                with dpg.theme() as area_theme:
                                    with dpg.theme_component(dpg.mvAreaSeries):
                                        dpg.add_theme_color(
                                            dpg.mvPlotCol_Line,
                                            (255, 255, 255, 0),
                                            category=dpg.mvThemeCat_Plots,
                                        )

                                if dpg.does_item_exist(f"fit_plot_{peak_id}"):
                                    dpg.configure_item(f"fit_plot_{peak_id}", x=x, y=y)
                                    dpg.bind_item_theme(
                                        f"fit_plot_{peak_id}", plot_theme
                                    )
                                else:
                                    x_bounds = (
                                        f"{fit.x_bounds[0]:.0f}..{fit.x_bounds[1]:.0f}"
                                    )
                                    dpg.add_line_series(
                                        x,
                                        y,
                                        parent="libs_y_axis",
                                        label=f"{series.name} fit {x_bounds} peak {peak_i}",
                                        tag=f"fit_plot_{peak_id}",
                                        user_data={"fit": peak_id},
                                    )
                                    dpg.add_area_series(
                                        x_area,
                                        y_area,
                                        parent="libs_y_axis",
                                        fill=fit_fill_color.tolist(),
                                        tag=f"fit_plot_area_{peak_id}",
                                        user_data={"fit": peak_id},
                                    )
                                    dpg.bind_item_theme(
                                        f"fit_plot_{peak_id}", plot_theme
                                    )
                                    dpg.bind_item_theme(
                                        f"fit_plot_area_{peak_id}", area_theme
                                    )

        # cleanup
        for s_id, series in self.project.series.items():
            fits = series.fits
            if not series.selected:
                for fit_id, fit in fits.items():
                    if dpg.does_item_exist(f"fit_plot_{fit_id}"):
                        dpg.delete_item(f"fit_plot_{fit_id}")
                    if dpg.does_item_exist(f"fit_plot_area_{fit_id}"):
                        dpg.delete_item(f"fit_plot_area_{fit_id}")
                    for peak_id, peak in fit.components.items():
                        if dpg.does_item_exist(f"fit_plot_{peak_id}"):
                            dpg.delete_item(f"fit_plot_{peak_id}")
                        if dpg.does_item_exist(f"fit_plot_area_{peak_id}"):
                            dpg.delete_item(f"fit_plot_area_{peak_id}")
            else:
                for fit_id, fit in fits.items():
                    if dpg.does_item_exist(f"fit_plot_{fit_id}"):
                        dpg.delete_item(f"fit_plot_{fit_id}")
                    if dpg.does_item_exist(f"fit_plot_area_{fit_id}"):
                        dpg.delete_item(f"fit_plot_area_{fit_id}")
                    if not fit.selected:
                        for peak_id, peak in fit.components.items():
                            if dpg.does_item_exist(f"fit_plot_{peak_id}"):
                                dpg.delete_item(f"fit_plot_{peak_id}")
                            if dpg.does_item_exist(f"fit_plot_area_{peak_id}"):
                                dpg.delete_item(f"fit_plot_area_{peak_id}")

        peak_ids = []
        for series in self.project.series.values():
            for fit in series.fits.values():
                for peak_id, peak in fit.components.items():
                    peak_ids.append(peak_id)

        y_axis_children = dpg.get_item_children("libs_y_axis", slot=1)
        if y_axis_children is None:
            return
        assert isinstance(y_axis_children, list)

        for plot in y_axis_children:
            if not dpg.does_item_exist(plot):
                continue
            plot_user_data = dpg.get_item_user_data(plot)
            if plot_user_data is None:
                continue

            peak_id = plot_user_data.get("fit", None)
            if peak_id is None:
                continue

            if peak_id not in peak_ids:
                if dpg.does_item_exist(f"fit_plot_{peak_id}"):
                    dpg.delete_item(f"fit_plot_{peak_id}")
                if dpg.does_item_exist(f"fit_plot_area_{peak_id}"):
                    dpg.delete_item(f"fit_plot_area_{peak_id}")

    def show_fit_plots(self):
        if self.settings.fitting_fit_display_mode.value == "Sum":
            self.show_fit_plots_sum()
        elif self.settings.fitting_fit_display_mode.value == "Components":
            self.show_fit_plots_components()

    def fits_lmb_callback(self, sender, data):
        wrapper_id: int = data[1]
        user_data = dpg.get_item_user_data(wrapper_id)
        if user_data is None:
            return

        series = self.project.series[user_data["series"]]
        fit = series.fits[user_data["fit"]]

        fit.selected = not fit.selected

        self.refresh_fit_results()
        self.show_fit_plots()

    def fits_rmb_callback(self, sender, data):
        wrapper_id: int = data[1]
        user_data = dpg.get_item_user_data(wrapper_id)
        if user_data is None:
            return

        series = self.project.series[user_data["series"]]
        fit = series.fits[user_data["fit"]]

        if not dpg.does_item_exist(f"{fit.id}_rmb_window"):
            with dpg.window(
                no_title_bar=True,
                no_move=True,
                no_open_over_existing_popup=True,
                popup=True,
                menubar=True,
                tag=f"{fit.id}_rmb_window",
            ):
                with dpg.menu_bar():
                    dpg.add_menu(
                        label=f"{series.name} fit {fit.printable_x_bounds} info",
                        enabled=False,
                    )
                with dpg.table(
                    borders_innerH=True,
                    borders_innerV=True,
                    borders_outerH=True,
                    borders_outerV=True,
                    freeze_rows=1,
                    policy=dpg.mvTable_SizingFixedFit,
                    width=-1,
                ):
                    dpg.add_table_column(label="Parameter")
                    dpg.add_table_column(label="Value")
                    if fit.windows_total > 1:
                        with dpg.table_row():
                            dpg.add_text("Mean R²")
                            dpg.add_text(f"{fit.r_squared_mean:.4f}")
                        with dpg.table_row():
                            dpg.add_text("Min R²")
                            dpg.add_text(f"{fit.r_squared_min:.4f}")
                        with dpg.table_row():
                            dpg.add_text("R² std. dev.")
                            dpg.add_text(f"{fit.r_squared_st_dev:.4f}")
                        with dpg.table_row():
                            dpg.add_text("Windows")
                            dpg.add_text(f"{fit.windows_total}")
                        with dpg.table_row():
                            dpg.add_text("Peaks")
                            dpg.add_text(f"{fit.n_peaks}")
                    else:
                        with dpg.table_row():
                            dpg.add_text("R²")
                            dpg.add_text(f"{fit.r_squared_mean:.4f}")
                        with dpg.table_row():
                            dpg.add_text("Peaks")
                            dpg.add_text(f"{fit.n_peaks}")

                dpg.add_button(
                    label="Delete fit",
                    width=-1,
                    callback=lambda s, d: self.delete_fit(series, fit.id),
                )
        else:
            dpg.show_item(f"{fit.id}_rmb_window")

    def delete_fit(self, series: Series, fit_id: str):
        dpg.delete_item(f"{fit_id}_rmb_window")
        del series.fits[fit_id]
        self.refresh_fit_results()
        self.show_fit_plots()

    def show_all_plots(self):
        self.show_libs_plots()
        self.show_fit_plots()

    def show_emission_plots(self):
        emission_plots = dpg.get_item_children("libs_y_axis", slot=1)
        assert isinstance(emission_plots, list)

        if emission_plots:
            for plot in emission_plots:
                if dpg.get_item_user_data(plot) == "emission_plot":
                    dpg.delete_item(plot)

        emission_data_wl_region = self.project.selected_series[0].averaged.x_limits
        emission_data = [
            (
                symbol,
                element_plot_data(
                    select_wl_region(
                        get_emission_data(symbol), *emission_data_wl_region
                    )
                ),
            )
            for symbol in self.periodic_table.element_symbols_selected
            if symbol is not None
        ]

        with dpg.theme() as plot_theme:
            with dpg.theme_component(dpg.mvLineSeries):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 1, category=dpg.mvThemeCat_Plots
                )

        for element_symbol, data in emission_data:
            y_multiplier = np.max(
                [np.max(s.averaged.y) for s in self.project.selected_series]
            )
            x, y = data.T
            y = y / np.nanmax(y) * y_multiplier
            y = y.tolist()
            x = x.tolist()
            dpg.add_line_series(
                x,
                y,
                parent="libs_y_axis",
                label=element_symbol,
                user_data="emission_plot",
            )
            dpg.bind_item_theme(dpg.last_item(), plot_theme)
