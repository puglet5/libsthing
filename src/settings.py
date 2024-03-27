from enum import StrEnum
from typing import Callable, Generator, Literal, TypedDict

import dearpygui.dearpygui as dpg
from attrs import define, field, fields

from src.history import HISTORY, OperationCallback, undoable
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


@define(repr=False)
class Setting[T]:
    tag: DPGItem
    default_value: T = field()
    value: T = field(init=False)
    callback: Callable | None
    _last_value: T = field(init=False)

    def __attrs_post_init__(self):
        self.value = self.default_value
        self._last_value = self.default_value

    def _update(self):
        self.value = dpg.get_value(self.tag)
        if self.callback is not None:
            self.callback()

    @undoable
    def set(self, value: T, /, skip_callback=False, skip_undo=True):
        dpg.set_value(self.tag, value)
        self.value = value
        if self.callback is not None and not skip_callback:
            self.callback()

        if not skip_undo:
            self._set_last_value()

            operation_direct: OperationCallback = (
                self.set,
                (self.value,),
                {"skip_callback": skip_callback},
            )

            operation_inverse: OperationCallback = (
                self.set,
                (self._last_value,),
                {"skip_callback": skip_callback},
            )

            yield operation_direct
            yield operation_inverse
        else:
            yield None

    def bind_last_value_handler(self):
        with dpg.item_handler_registry(tag=f"{self.tag}_undo_handler"):
            dpg.add_item_activated_handler(callback=self._set_last_value)
            dpg.add_item_deactivated_handler(callback=self._set_current)

        dpg.bind_item_handler_registry(self.tag, f"{self.tag}_undo_handler")

    def _set_last_value(self):
        # ran before _update

        # prevent setting duplicate values
        if self.value == dpg.get_value(self.tag):
            return
        self._last_value = self.value

    def _set_current(self):
        self.set(dpg.get_value(self.tag), skip_callback=False, skip_undo=False)

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
    normalized_from: Setting[float]
    normalized_to: Setting[float]
    spectra_shift: Setting[float]
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
    peaks_shown: Setting[bool]
    fitting_windows_shown: Setting[bool]
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
