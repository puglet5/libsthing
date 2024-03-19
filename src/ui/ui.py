import gc
import logging
import math
from pathlib import Path
from typing import Literal

import dearpygui.dearpygui as dpg
import numpy as np
from attrs import define, field
from natsort import index_natsorted, order_by_index

from settings import BaselineRemoval, Setting, Settings
from src.utils import (
    LABEL_PAD,
    SIDEBAR_WIDTH,
    TOOLTIP_DELAY_SEC,
    WINDOW_TAG,
    Project,
    Series,
    Window,
)
from ui.peak_table import PeakTable
from ui.periodic_table import PeriodicTable

logger = logging.getLogger(__name__)


@define
class UI:
    project: Project = field(init=False)
    settings: Settings = field(init=False)
    periodic_table: PeriodicTable = field(init=False)
    peak_table: PeakTable = field(init=False)
    global_theme: int = field(init=False, default=0)
    button_theme: int = field(init=False, default=0)
    thumbnail_plot_theme: int = field(init=False, default=0)
    active_thumbnail_plot_theme: int = field(init=False, default=0)
    series_list_n_columns: int = field(init=False, default=5)

    def __attrs_post_init__(self):
        self.setup_settings()

        dpg.create_context()

        with dpg.font_registry():
            with dpg.font(Path("./src/fonts/cozette.ttf").absolute().as_posix(), 13):
                dpg.add_font_range(0x0370, 0x03FF)

        dpg.create_viewport(title="libsthing", width=1920, height=1080, vsync=True)
        dpg.configure_app(wait_for_input=False)

        self.setup_themes()
        self.bind_themes()
        self.setup_handler_registries()
        self.setup_layout()
        self.bind_item_handlers()
        self.periodic_table = PeriodicTable()
        self.peak_table = PeakTable()

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
                self.show_libs_plots,
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
                -1,
                None,
            ),
        )

    def reset_settings(self):
        self.settings.reset()
        self.show_libs_plots()
        self.refresh_all()

    def stop(self):
        dpg.stop_dearpygui()
        dpg.destroy_context()

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
            # with dpg.theme_component(dpg.mvAll):
            #     dpg.add_theme_style(
            #         dpg.mvStyleVar_SelectableTextAlign,
            #         0.5,
            #         category=dpg.mvThemeCat_Core,
            #     )

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

    def on_key_ctrl_release(self):
        dpg.configure_item("libs_plots", pan_button=dpg.mvMouseButton_Left)

    def on_key_ctrl_down(self):
        dpg.configure_item("libs_plots", pan_button=dpg.mvMouseButton_Middle)
        if dpg.is_key_pressed(dpg.mvKey_Q):
            self.stop()
        if dpg.is_key_pressed(dpg.mvKey_P):
            self.periodic_table.toggle()

    def series_left_click(self):
        if not dpg.is_item_clicked("series_list_wrapper"):
            return

        series_rows = dpg.get_item_children("series_list_wrapper", slot=1)
        if series_rows is None:
            return

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

                    with dpg.window(
                        pos=dpg.get_mouse_pos(),
                        no_title_bar=True,
                        no_move=True,
                        no_open_over_existing_popup=True,
                        popup=True,
                        menubar=True,
                    ):
                        with dpg.menu_bar():
                            dpg.add_menu(label=f"{series.name} info", enabled=False)

                        dpg.add_text(f"Directory: {series.directory}")
                        dpg.add_text(
                            f"{series.samples_total} samples with {series.spectra_total} spectra total"
                        )

                        dpg.add_text("")
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

    def directory_picker_callback(self, _sender, data):
        dpg.set_value("project_directory", data["file_path_name"])
        self.setup_project()

    def set_series_color(self, sender: int | str, color: list[int]):
        series_id = dpg.get_item_user_data(sender)
        assert series_id
        series = self.project.series[series_id]
        series.color = (np.array(color) * [255, 255, 255, 200]).astype(int).tolist()
        with dpg.mutex():
            self.populate_series_list(skip_plot_update=True)
            self.show_libs_plots()

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

        with dpg.mutex():
            for i, series in enumerate(self.project.selected_series):
                spectrum = series.averaged
                assert spectrum.raw_spectral_data is not None
                assert series.color
                spectrum.process_spectral_data(
                    normalized,
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
                        label=f"{series.name}",
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
        spectrum = self.project.selected_series[0].averaged
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
        fitted = spectrum.fit_windows_parallel(
            spectrum.fitting_windows, max_iterations=max_iterations
        )
        if fitted is not None:
            x, y = fitted.data.xy
            dpg.add_line_series(
                x,
                y,
                parent="libs_y_axis",
                label=f"Fitted",
            )

    def window_resize_callback(self, _sender=None, _data=None):
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        if dpg.does_item_exist("loading_indicator"):
            dpg.configure_item("loading_indicator", pos=(w // 2 - 100, h // 2 - 100))

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
                            dpg.add_drag_line(
                                color=series.color,
                                default_value=e,
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
                            label=f"{s.name}_peak_{i}",
                            tag=f"{s.id}_peak_{i}",
                        )

        self.refresh_peak_table()

    def refresh_fitting_windows(self):
        if self.settings.fitting_windows_shown.value:
            self.settings.fitting_windows_shown.set(False)
            self.settings.fitting_windows_shown.set(True)

    def refresh_peaks(self):
        if self.settings.peaks_shown.value:
            self.settings.peaks_shown.set(False)
            self.settings.peaks_shown.set(True)

        self.refresh_peak_table()

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
        self.peak_table.peaks = self.project.selected_series[0].averaged.peaks

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

    def plot_query_callback(self, sender, data):
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
                    dpg.add_menu_item(label="Quit", shortcut="(Ctrl+Q)")

                with dpg.menu(label="Edit"):
                    dpg.add_menu_item(
                        label="Reset all inputs", callback=lambda: self.reset_settings()
                    )
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
                        shortcut="(F11)",
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
                                dpg.add_text("Drop first in samples".rjust(LABEL_PAD))
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
                            height=400,
                            no_scrollbar=True,
                        ):
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
            width=800,
            height=600,
            menubar=True,
            show=False,
            no_scrollbar=True,
            autosize=True,
        ):
            dpg.add_text(tag="fitting_results_text")

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
