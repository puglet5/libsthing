import logging

import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
from attrs import define, field

logger = logging.getLogger(__name__)


@define
class PeakTable:
    _peaks: npt.NDArray[np.float_] | None = field(init=False)

    def __attrs_post_init__(self):
        with dpg.window(
            label=f"Peak table",
            show=False,
            no_resize=True,
            autosize=True,
            max_size=(400, 600),
            tag="peak_table_window",
        ):
            with dpg.table(
                hideable=True,
                borders_innerH=True,
                borders_innerV=True,
                borders_outerH=True,
                borders_outerV=True,
                policy=dpg.mvTable_SizingFixedFit,
                tag="peak_table",
            ):
                dpg.add_table_column(label="Peak no.")
                dpg.add_table_column(label="Pos., nm")
                dpg.add_table_column(label="Cnts.")
                dpg.add_table_column(label="Cnts., rel.")
                dpg.add_table_column(label="Element")

    def is_shown(self):
        return dpg.is_item_shown("peak_table_window")

    def show(self):
        dpg.show_item("peak_table_window")

    def hide(self):
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
        self.populate_table()
        self.highlight_table()

    def populate_table(self):
        dpg.delete_item("peak_table", children_only=True, slot=1)
        if self.peaks is None or (len(self.peaks) == 0):
            return

        max_peak_y = np.max(self.peaks[:, 1])

        for i, peak in enumerate(self.peaks):
            with dpg.table_row(parent="peak_table"):
                dpg.add_selectable(label=str(i), span_columns=True)
                dpg.add_selectable(label=f"{peak[0]:.2f}")
                dpg.add_selectable(label=f"{peak[1]:.2f}")
                dpg.add_selectable(label=f"{(peak[1] / max_peak_y):.2f}")

    def highlight_table(self): ...

    def clear_selections(self):
        self.elements_z_selected = set()
