import logging

import dearpygui.dearpygui as dpg
from attrs import define, field

logger = logging.getLogger(__name__)


@define
class SimulationWindow:

    def __attrs_post_init__(self):
        with dpg.window(
            label=f"Reference & Simulation",
            show=False,
            no_resize=True,
            autosize=True,
            max_size=(1600, 900),
            tag="simulation_window",
        ):
            with dpg.plot(
                height=600,
                width=800,
            ):
                dpg.add_plot_axis(dpg.mvXAxis)
                dpg.add_plot_axis(dpg.mvYAxis)

    def is_shown(self):
        return dpg.is_item_shown("simulation_window")

    def show(self):
        dpg.show_item("simulation_window")

    def hide(self):
        dpg.hide_item("simulation_window")

    def toggle(self):
        if dpg.is_item_shown("simulation_window"):
            dpg.hide_item("simulation_window")
        else:
            dpg.show_item("simulation_window")
