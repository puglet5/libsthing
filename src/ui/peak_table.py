import logging

import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
from attrs import define, field

from utils import DPGItem, Series

logger = logging.getLogger(__name__)


@define
class PeakTable:
    _peaks: npt.NDArray[np.float_] | None = field(init=False)
    series: Series = field(init=False)
    selected_rows: set[int] = field(init=False, factory=set)
    hovered_row: int = field(init=False, default=0)
    last_hovered_row: int = field(init=False, default=0)

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
            tag="peak_table_window",
            no_scrollbar=True,
            on_close=self.on_close
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
                policy=dpg.mvTable_SizingFixedFit,
                tag="peak_table",
                height=-1,
            ):
                dpg.add_table_column(label="Peak no.")
                dpg.add_table_column(label="Pos., nm")
                dpg.add_table_column(label="Cnts.")
                dpg.add_table_column(label="Cnts., rel.")
                dpg.add_table_column(label="Area, cnts*nm")
                dpg.add_table_column(label="FWHM, nm")

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

    @property
    def peaks(self):
        return self._peaks

    @peaks.setter
    def peaks(self, new_peaks):
        self._peaks = new_peaks
        dpg.configure_item(
            "peak_table_window", label=f"Peak table ({self.series.name})"
        )
        self.populate_table()
        self.highlight_table()

    def populate_table(self):
        self.clear_selections()
        dpg.delete_item("peak_table", children_only=True, slot=1)
        if self.peaks is None or (len(self.peaks) == 0):
            return

        max_peak_y = np.max(self.peaks[:, 1])

        for i, peak in enumerate(self.peaks):
            with dpg.table_row(parent="peak_table"):
                dpg.add_selectable(
                    label=str(i),
                    span_columns=True,
                    user_data=i,
                    callback=lambda s, d: self.row_clicked(s, d),
                )
                dpg.bind_item_handler_registry(
                    dpg.last_item(), "peak_table_row_hover_handler"
                )

                dpg.add_selectable(label=f"{peak[0]:.3f}")
                dpg.add_selectable(label=f"{peak[1]:.0f}")
                dpg.add_selectable(label=f"{(peak[1] / max_peak_y):.2f}")

    def highlight_table(self): ...

    def clear_selections(self):
        self.selected_rows = set()

    def on_close(self):
        row_i: int = dpg.get_item_user_data(self.hovered_row)  # type: ignore
        dpg.configure_item(f"{self.series.id}_peak_{row_i}", color=self.series.color)

    def highlight_drag_point(self, _, row):
        if self.hovered_row == row:
            return

        self.last_hovered_row = self.hovered_row
        self.hovered_row = row

        row_i: int = dpg.get_item_user_data(row)  # type: ignore
        if self.last_hovered_row != 0:
            if dpg.does_item_exist(self.last_hovered_row):
                last_hovered_row_i: int = dpg.get_item_user_data(self.last_hovered_row)  # type: ignore
                dpg.configure_item(
                    f"{self.series.id}_peak_{last_hovered_row_i}",
                    color=self.series.color,
                )

        dpg.configure_item(f"{self.series.id}_peak_{row_i}", color=[0, 255, 0, 255])

    def row_clicked(self, row: DPGItem, state: bool):
        row_i: int = dpg.get_item_user_data(row)  # type: ignore
        if state:
            self.selected_rows.add(row_i)
        else:
            self.selected_rows.remove(row_i)
