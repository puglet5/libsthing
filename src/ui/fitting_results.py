import logging

import dearpygui.dearpygui as dpg
from attrs import define

logger = logging.getLogger(__name__)


@define
class FittingResults:

    def __attrs_post_init__(self):
        with dpg.window(
            label="Fitting results",
            tag="fitting_results_window",
            width=800,
            height=600,
            menubar=True,
            show=False,
            no_scrollbar=True,
            autosize=True,
        ):
            dpg.add_text(tag="fitting_results_text")

    def is_shown(self):
        return dpg.is_item_shown("fitting_results_window")

    def show(self):
        dpg.show_item("fitting_results_window")

    def hide(self):
        dpg.hide_item("fitting_results_window")

    def toggle(self):
        if dpg.is_item_shown("fitting_results_window"):
            dpg.hide_item("fitting_results_window")
        else:
            dpg.show_item("fitting_results_window")
