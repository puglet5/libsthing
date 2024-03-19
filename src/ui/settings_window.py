import logging

import dearpygui.dearpygui as dpg
from attrs import define, field

logger = logging.getLogger(__name__)


@define
class SettingsWindow:

    def __attrs_post_init__(self):
        with dpg.window(
            label=f"Settings",
            show=False,
            no_resize=True,
            max_size=(800, 600),
            tag="settings_window",
        ):
            ...

    def is_shown(self):
        return dpg.is_item_shown("settings_window")

    def show(self):
        dpg.show_item("settings_window")

    def hide(self):
        dpg.hide_item("settings_window")

    def toggle(self):
        if dpg.is_item_shown("settings_window"):
            dpg.hide_item("settings_window")
        else:
            dpg.show_item("settings_window")
