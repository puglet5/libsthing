from enum import StrEnum
from typing import Callable, Generator, Literal, TypedDict

import dearpygui.dearpygui as dpg
from attrs import define, field, fields

from src.utils import DPGItem


class BaselineRemoval(StrEnum):
    SNIP = "SNIP"
    NONE = "None"
    POLY = "Polynomial"
    AMM = "Adaptive minmax"


class DPGWidgetArgs[T](TypedDict):
    tag: DPGItem
    default_value: T
    callback: Callable


@define
class Setting[T]:
    tag: DPGItem
    default_value: T = field()
    value: T = field(init=False)
    callback: Callable | None

    def __attrs_post_init__(self):
        self.value = self.default_value

    def _update(self):
        self.value = dpg.get_value(self.tag)
        if self.callback is not None:
            self.callback()

    def set(self, value: T, skip_callback=False):
        dpg.set_value(self.tag, value)
        self.value = value
        if self.callback is not None and not skip_callback:
            self.callback()

    def set_default(self):
        self.set(self.default_value)

    @property
    def as_dict(self):
        args: DPGWidgetArgs[T] = {
            "tag": self.tag,
            "default_value": self.default_value,
            "callback": self._update,
        }

        return args

    def disable(self):
        dpg.disable_item(self.tag)

    def enable(self):
        dpg.enable_item(self.tag)


@define
class Settings:
    project_common_x: Setting[bool]
    sample_drop_first: Setting[int]
    spectra_normalized: Setting[bool]
    spectra_fit_to_axes: Setting[bool]
    spectra_normalization_method: Setting[Literal["Area", "Max. intensity", "Norm"]]
    spectra_shift: Setting[float]
    normalized_from: Setting[float]
    normalized_to: Setting[float]
    baseline_removal_method: Setting[BaselineRemoval]
    baseline_clipped_to_zero: Setting[bool]
    baseline_polynomial_degree: Setting[int]
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
    fitting_fit_display_mode: Setting[Literal["Sum", "Components", "Both"]]
    fitting_fit_info: Setting[Literal["Minimal", "All"]]
    fitting_autoselect: Setting[bool]
    fitting_inclide_in_legend: Setting[bool]
    fitting_default_color: Setting[Literal["Series", "Negative", "Custom"]]
    fitting_fill: Setting[bool]
    fitting_default_color_colorpicker: Setting[list[int]]
    emission_lines_max_ionization_level: Setting[int]
    emission_lines_follow_selected_region: Setting[bool]
    emission_lines_intensity_threshold: Setting[float]
    emission_lines_fit_intensity: Setting[bool]

    def __iter__(self) -> Generator[Setting, None, None]:
        return (getattr(self, field.name) for field in fields(Settings))

    def reset(self):
        for setting in self:
            setting.set(setting.default_value, skip_callback=True)
