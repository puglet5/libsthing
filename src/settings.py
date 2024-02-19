import inspect
from functools import wraps
from typing import Any, Callable, Literal

import dearpygui.dearpygui as dpg
from attrs import define, field

from utils import Window


@define
class Setting[T]:
    tag: str | int
    default_value: T = field()
    value: T = field(init=False)
    callback: Callable | None

    def __attrs_post_init__(self):
        self.value = self.default_value

    def update(self):
        self.value = dpg.get_value(self.tag)
        if self.callback is not None:
            self.callback()

    def set(self, value: T):
        dpg.set_value(self.tag, value)
        self.value = value
        if self.callback is not None:
            self.callback()

    @property
    def as_dict(self):
        return {
            "tag": self.tag,
            "default_value": self.default_value,
            "callback": self.update,
        }

    def disable(self):
        dpg.disable_item(self.tag)

    def enable(self):
        dpg.enable_item(self.tag)


@define
class Settings:
    spectra_normalized: Setting[bool]
    spectra_fit_to_axes: Setting[bool]
    normzlized_from: Setting[float]
    normzlized_to: Setting[float]
    baseline_removal_method: Setting[
        Literal["None", "SNIP", "Adaptive minmax", "Polynomial"]
    ]
    baseline_clipped_to_zero: Setting[bool]
    baseline_max_half_window: Setting[int]
    baseline_filter_order: Setting[Literal["2", "4", "6", "8"]]
    min_peak_height: Setting[float]
    min_peak_prominance: Setting[float]
    peak_smoothing_sigma: Setting[float]
    selection_guides_shown: Setting[bool]
    region_subdivided: Setting[bool]
    fitting_windows_shown: Setting[bool]
    peaks_shown: Setting[bool]
    fitting_x_threshold: Setting[float]
    fitting_y_threshold: Setting[float]
    fitting_y_threshold_type: Setting[Literal["Absolute", "Relative"]]
    fitting_max_iterations: Setting[Literal[-1] | int]
