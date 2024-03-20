import logging

import dearpygui.dearpygui as dpg
from attrs import define, field

logger = logging.getLogger(__name__)


@define
class SettingsWindow:

    def __attrs_post_init__(self):
        with dpg.window(
            label=f"Settings",
            tag="settings_window",
            show=False,
            no_move=True,
            no_collapse=True,
            modal=True,
            width=700,
            height=400,
            no_resize=True,
        ):
            ...

        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()

        dpg.configure_item("settings_window", pos=[w // 2 - 350, h // 2 - 200])

    def is_shown(self):
        return dpg.is_item_shown("settings_window")

    def show(self):
        w, h = dpg.get_viewport_width(), dpg.get_viewport_height()
        dpg.configure_item("settings_window", pos=[w // 2 - 350, h // 2 - 200])
        dpg.show_item("settings_window")

    def hide(self):
        dpg.hide_item("settings_window")

    def toggle(self):
        if dpg.is_item_shown("settings_window"):
            dpg.hide_item("settings_window")
        else:
            dpg.show_item("settings_window")
