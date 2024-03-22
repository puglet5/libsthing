import logging

import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
from attrs import define, field

from utils import DPGItem, Series

logger = logging.getLogger(__name__)


@define
class PeakTable:
    series: list[Series] = field(init=False, factory=list)
    selected_rows_regular: set[int] = field(init=False, factory=set)
    hovered_row_regular: int = field(init=False, default=0)
    last_hovered_row_fitted: int = field(init=False, default=0)

    def __attrs_post_init__(self):
        with dpg.item_handler_registry(tag="peak_table_row_hover_handler"):
            dpg.add_item_hover_handler(
                callback=lambda s, d: self.highlight_drag_point(s, d)
            )

        with dpg.window(
            label=f"Peak table",
            show=False,
            no_resize=True,
            autosize=True,
            width=800,
            height=600,
            max_size=(800, 600),
            min_size=(400, 200),
            tag="peak_table_window",
            no_scrollbar=True,
            on_close=self.on_close,
        ):
            with dpg.tab_bar():
                with dpg.tab(label="All", tag="peak_table_regular_tab"):
                    with dpg.tab_bar(tag="peak_table_regular_tabbar", show=False):
                        ...
                with dpg.tab(label="Fitted"):
                    with dpg.tab_bar(tag="peak_table_fitted_tabbar", show=False):
                        ...

    def is_shown(self):
        return dpg.is_item_shown("peak_table_window")

    def show(self):
        dpg.show_item("peak_table_window")

    def hide(self):
        self.on_close()
        dpg.hide_item("peak_table_window")

    def toggle(self):
        if dpg.is_item_shown("peak_table_window"):
            dpg.hide_item("peak_table_window")
        else:
            dpg.show_item("peak_table_window")

    def populate_table_regular(self):
        with dpg.mutex():
            dpg.hide_item("peak_table_regular_tabbar")
            self.clear_selections()
            dpg.delete_item("peak_table_regular_tabbar", children_only=True, slot=1)

        for series in self.series:
            if series.averaged is None:
                continue

            if series.averaged.peaks is None:
                continue

            if len(series.averaged.peaks) == 0:
                continue

            dpg.show_item("peak_table_regular_tabbar")

            with dpg.tab(
                label=f"{series.name}",
                parent="peak_table_regular_tabbar",
            ):
                with dpg.table(
                    hideable=True,
                    borders_innerH=True,
                    borders_innerV=True,
                    borders_outerH=True,
                    borders_outerV=True,
                    scrollY=True,
                    freeze_rows=1,
                    freeze_columns=1,
                    # policy=dpg.mvTable_SizingFixedFit,
                    height=-1,
                    width=-1,
                    tag=f"peak_table_{series.id}",
                ):
                    dpg.add_table_column(label="Peak no.")
                    dpg.add_table_column(label="Pos., nm")
                    dpg.add_table_column(label="Cnts.")
                    dpg.add_table_column(label="Cnts., rel.")

            max_peak_y = np.max(series.averaged.peaks[:, 1])

            for peak_i, peak in enumerate(series.averaged.peaks):
                with dpg.table_row(parent=f"peak_table_{series.id}"):
                    dpg.add_selectable(
                        label=str(peak_i),
                        span_columns=True,
                        user_data={"series": series, "peak": peak_i},
                        callback=lambda s, d: self.row_clicked(s, d),
                    )
                    dpg.bind_item_handler_registry(
                        dpg.last_item(), "peak_table_row_hover_handler"
                    )

                    dpg.add_selectable(label=f"{peak[0]:.3f}")
                    dpg.add_selectable(label=f"{peak[1]:.0f}")
                    dpg.add_selectable(label=f"{(peak[1] / max_peak_y):.2f}")

    def populate_table_fitted(self):
        with dpg.mutex():
            dpg.hide_item("peak_table_fitted_tabbar")
            self.clear_selections()
            dpg.delete_item("peak_table_fitted_tabbar", children_only=True, slot=1)

            for series in self.series:
                if series.fits is None:
                    continue

                if len(series.fits) == 0:
                    continue

                dpg.show_item("peak_table_fitted_tabbar")

                with dpg.tab(
                    label=f"{series.name}",
                    parent="peak_table_fitted_tabbar",
                ):
                    with dpg.table(
                        hideable=True,
                        borders_innerH=True,
                        borders_innerV=True,
                        borders_outerH=True,
                        borders_outerV=True,
                        scrollY=True,
                        freeze_rows=1,
                        freeze_columns=1,
                        # policy=dpg.mvTable_SizingFixedFit,
                        height=-1,
                        tag=f"peak_table_fitted_{series.id}",
                    ):
                        dpg.add_table_column(label="Pos., nm")
                        dpg.add_table_column(label="Cnts.")
                        dpg.add_table_column(label="Area, cnts*nm")
                        dpg.add_table_column(label="FWHM, nm")
                        dpg.add_table_column(label="σ, nm")
                        dpg.add_table_column(label="α")

                for fit_id, fit in series.fits.items():
                    if not fit.selected:
                        continue
                    for peak in fit.components.values():
                        with dpg.table_row(parent=f"peak_table_fitted_{series.id}"):
                            dpg.add_selectable(
                                label=f"{peak.center_fitted:.3f}", span_columns=True
                            )
                            dpg.add_selectable(label=f"{np.max(peak.fitted.y):.3f}")
                            dpg.add_selectable(label=f"{peak.amplitude_fitted:.3f}")
                            dpg.add_selectable(label=f"{peak.fwhm_fitted:.3f}")
                            dpg.add_selectable(label=f"{peak.sigma_fitted:.3f}")
                            dpg.add_selectable(label=f"{peak.fraction_fitted:.2f}")

    def highlight_table(self): ...

    def clear_selections(self):
        self.selected_rows_regular = set()

    def on_close(self):
        if self.hovered_row_regular != 0:
            row_user_data = dpg.get_item_user_data(self.hovered_row_regular)
            if row_user_data is None:
                return
            row_i = row_user_data["peak"]
            series = row_user_data["series"]
            dpg.configure_item(f"{series.id}_peak_{row_i}", color=series.color)

    def highlight_drag_point(self, _, row):
        if self.hovered_row_regular == row:
            return

        self.last_hovered_row_fitted = self.hovered_row_regular
        self.hovered_row_regular = row

        row_user_data = dpg.get_item_user_data(row)
        if row_user_data is None:
            return
        row_i = row_user_data["peak"]
        series_current = row_user_data["series"]
        if self.last_hovered_row_fitted != 0:
            if dpg.does_item_exist(self.last_hovered_row_fitted):
                last_hovered_row_user_data = dpg.get_item_user_data(self.last_hovered_row_fitted)  # type: ignore
                if last_hovered_row_user_data is None:
                    return
                series_last = last_hovered_row_user_data["series"]
                peak_i = last_hovered_row_user_data["peak"]
                dpg.configure_item(
                    f"{series_last.id}_peak_{peak_i}",
                    color=series_last.color,
                )

        dpg.configure_item(f"{series_current.id}_peak_{row_i}", color=[0, 255, 0, 255])

    def row_clicked(self, row: DPGItem, state: bool):
        if not dpg.does_item_exist(row):
            return
        row_i: int = dpg.get_item_user_data(row)["peak"]  # type: ignore
        if state:
            self.selected_rows_regular.add(row_i)
        else:
            self.selected_rows_regular.remove(row_i)
