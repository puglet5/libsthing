import logging
import os
import re
import threading
import time
import uuid
from functools import partial, wraps
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    ParamSpec,
    Sequence,
    Sized,
    TypeAlias,
    TypedDict,
    TypeVar,
    assert_never,
)

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
from attr import define, field
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from lmfit.models import PseudoVoigtModel
from natsort import natsorted
from pathos.multiprocessing import ProcessingPool as Pool
from pyarrow import csv
from pybaselines import Baseline
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

DPGItem = str | int

from src.settings import BaselineRemoval

PA_READ_OPTIONS = csv.ReadOptions(skip_rows=2, autogenerate_column_names=True)
PA_PARSE_OPTIONS = csv.ParseOptions(delimiter="\t", quote_char=False)
PA_CONVERT_OPTIONS = csv.ConvertOptions(
    decimal_point=",",
    check_utf8=False,
    quoted_strings_can_be_null=False,
    column_types={"f0": pa.float32(), "f1": pa.float32()},
)
PA_CONVERT_OPTIONS_SKIP_X = csv.ConvertOptions(
    decimal_point=",",
    check_utf8=False,
    quoted_strings_can_be_null=False,
    column_types={"f0": pa.float32(), "f1": pa.float32()},
    include_columns=["f1"],
)

LOADING_INDICATOR_FETCH_DELAY_S = 0.1
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=Path(ROOT_DIR, "log/main.log"), filemode="a")
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

CPU_COUNT = cpu_count()

Window = tuple[float, float]
Windows = list[Window]

TOOLTIP_DELAY_SEC = 0.1
LABEL_PAD = 23
WINDOW_TAG = "primary"
SIDEBAR_WIDTH = 350


def log_exec_time[T, **P](f: Callable[P, T]) -> Callable[P, T]:
    @wraps(f)
    def _wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = f(*args, **kwargs)
        logger.debug(f"{f.__name__}: {time.perf_counter() - start_time} s.")
        return result

    return _wrapper  # type:ignore


def show_loading_indicator():
    dpg.show_item("loading_indicator")


def hide_loading_indicator():
    if dpg.is_item_shown("loading_indicator"):
        dpg.hide_item("loading_indicator")


def loading_indicator[
    T, **P
](f: Callable[P, Generator[float | int | str, Any, T]], message: str) -> Callable[P, T]:
    @wraps(f)
    def _wrapper(*args, **kwargs):
        dpg.configure_item("loading_indicator_message", label=message.center(30))
        threading.Timer(0.1, show_loading_indicator).start()
        progress_generator = f(*args, **kwargs)
        progress = 0

        try:
            while True:
                progress = next(progress_generator)
                if progress == "aborted":
                    dpg.configure_item(
                        "loading_indicator_message",
                        label=f"{message}: aborted!".center(30),
                    )
                else:
                    dpg.configure_item(
                        "loading_indicator_message",
                        label=f"{message}: {progress:.0f}%".center(30),
                    )
        except StopIteration as result:
            return result.value
        except TypeError:
            return None
        except Exception as e:
            raise ValueError from e
        finally:
            if progress == "aborted":
                dpg.configure_item(
                    "loading_indicator_message",
                    label=f"{message}: aborted!".center(30),
                )
            else:
                dpg.configure_item(
                    "loading_indicator_message",
                    label=f"{message}: 100%".center(30),
                )

            threading.Timer(0.5, hide_loading_indicator).start()

    return _wrapper  # type:ignore


def flatten(l: list):
    if not isinstance(l[0], Iterable):
        return l
    return list(chain.from_iterable(l))


def multi_sub(sub_pairs: list[tuple[str, str]], string: str):
    def repl_func(m):
        return next(
            repl for (_, repl), group in zip(sub_pairs, m.groups()) if group is not None
        )

    pattern = "|".join("({})".format(patt) for patt, _ in sub_pairs)
    return re.sub(pattern, repl_func, string, flags=re.U)


def partition(list_, indices: list[int]):
    indices = [0] + indices + [len(list_)]
    return [list_[v : indices[k + 1]] for k, v in enumerate(indices[:-1])]


def nearest_ciel(arr: np.ndarray, val: float):
    i = np.searchsorted(arr, val)
    return arr[i]


def nearest_floor(arr: np.ndarray, val: float):
    i = np.searchsorted(arr, val)
    return arr[i - 1]


def nearest(arr: np.ndarray, val: float):
    return arr.flat[np.abs(arr - val).argmin()]


def np_delete_row(arr: np.ndarray, num: int):
    mask = np.zeros(arr.shape[0], dtype=np.int64) == 0
    mask[np.where(arr == num)[0]] = False
    return arr[mask]


@define(repr=False)
class Spectrum:
    file: Path | None = field(default=None)
    raw_spectral_data: npt.NDArray | None = field(default=None, repr=False)
    processed_spectral_data: npt.NDArray | None = field(
        default=None, init=False, repr=False
    )
    common_x: npt.NDArray | None = field(default=None, repr=False)
    peaks: npt.NDArray | None = field(init=False, default=None)
    fitting_windows: Windows = field(init=False, factory=list)
    series: "Series | None" = field(init=False, default=None)

    def __attrs_post_init__(self):
        if self.raw_spectral_data is None and self.file is not None:
            self.raw_spectral_data = self.spec_to_numpy()

        if self.raw_spectral_data is None and self.file is None:
            raise ValueError

        self.processed_spectral_data = self.raw_spectral_data

    @classmethod
    def from_file(cls, file: Path, common_x: npt.NDArray | None = None):
        return cls(file=file, common_x=common_x)

    @classmethod
    def from_data(cls, data: npt.NDArray):
        return cls(raw_spectral_data=data)

    def spec_to_numpy(self):
        if not isinstance(self.common_x, np.ndarray):
            return np.array(
                csv.read_csv(
                    self.file,
                    read_options=PA_READ_OPTIONS,
                    parse_options=PA_PARSE_OPTIONS,
                    convert_options=PA_CONVERT_OPTIONS,
                ),
                dtype=np.float32,
            )

        x = self.common_x
        y = np.array(
            csv.read_csv(
                self.file,
                read_options=PA_READ_OPTIONS,
                parse_options=PA_PARSE_OPTIONS,
                convert_options=PA_CONVERT_OPTIONS_SKIP_X,
            ),
            dtype=np.float32,
        )

        return np.c_[x, y]

    @property
    def x(self):
        if self.raw_spectral_data is None:
            raise ValueError
        if self.processed_spectral_data is None:
            return self.raw_spectral_data[:, 0]

        return self.processed_spectral_data[:, 0]

    @property
    def y(self):
        if self.processed_spectral_data is None:
            raise ValueError
        return self.processed_spectral_data[:, 1]

    @property
    def step(self) -> float:
        return self.x[1] - self.x[0]

    @property
    def xy(self):
        if self.processed_spectral_data is None:
            raise ValueError
        return self.processed_spectral_data.T

    @property
    def area_fill_xy(self):
        if self.processed_spectral_data is None:
            raise ValueError

        area_coords = np.append(
            self.processed_spectral_data,
            [
                [self.processed_spectral_data[-1][0], 0],
                [self.processed_spectral_data[0][0], 0],
                self.processed_spectral_data[0],
            ],
            axis=0,
        )

        return area_coords.T

    @property
    def x_limits(self) -> Window:
        return (self.x.min(), self.x.max())

    def baseline(
        self,
        y: np.ndarray,
        method: BaselineRemoval,
        method_params: dict,
    ) -> np.ndarray:
        baseline_fitter = Baseline(x_data=self.x)
        if method == BaselineRemoval.SNIP:
            bkg = baseline_fitter.snip(y, **method_params)[0]
        elif method == BaselineRemoval.AMM:
            bkg = baseline_fitter.adaptive_minmax(y)[0]
        elif method == BaselineRemoval.POLY:
            bkg = baseline_fitter.penalized_poly(y, **method_params)[0]
        elif method == BaselineRemoval.NONE:
            return np.zeros_like(y)
        else:
            assert_never(method)

        return bkg

    def process_spectral_data(
        self,
        shift: float,
        normalized=False,
        normalization_range: Window = (1, -1),
        normalization_method: Literal["Area", "Max. intensity", "Norm"] = "Area",
        baseline_clip=True,
        baseline_removal: BaselineRemoval = BaselineRemoval.SNIP,
        baseline_params: dict[str, int] | None = None,
    ):
        if (data := self.raw_spectral_data) is None:
            raise ValueError

        if baseline_params is None:
            baseline_params = {}

        x, y = data.T

        if baseline_removal != BaselineRemoval.NONE:
            bkg = self.baseline(
                y, method=baseline_removal, method_params=baseline_params
            )

            if baseline_clip:
                y = np.clip(y - bkg, 0, None)
            else:
                y = y - bkg

        if shift != 0:
            x = x + shift

        if normalized:
            norm_min = max(normalization_range[0], self.x.min())
            if normalization_range[1] == -1:
                norm_max = self.x.max()
            else:
                norm_max = max(norm_min, min(normalization_range[1], self.x.max()))

            if (
                norm_min > self.x.max()
                or norm_max < self.x.min()
                or norm_min == norm_max
            ):
                pass
            else:
                normalization_range_mask = np.logical_and(
                    self.x > norm_min, self.x < norm_max
                )
                if normalization_method == "Max. intensity":
                    y = y / y[normalization_range_mask].max()
                elif normalization_method == "Area":
                    y = y / np.trapz(
                        y[normalization_range_mask],
                        x[normalization_range_mask],
                    )
                elif normalization_method == "Norm":
                    y = y / np.linalg.norm(y[normalization_range_mask])

        self.processed_spectral_data = np.array([x, y]).T

    def find_peaks(
        self,
        region: Window | None = None,
        height=0.005,
        prominance=0.001,
        smoothing_sigma=1.0,
        y: npt.NDArray | None = None,
    ):
        if not isinstance(y, np.ndarray):
            y = self.y
        assert y is not None

        if region is None:
            region = self.x_limits

        x = self.x[np.logical_and(self.x >= region[0], self.x <= region[1])]
        y = y[np.logical_and(self.x >= region[0], self.x <= region[1])]

        if len(x) <= 4:
            self.peaks = np.array([])
            return

        y_spl_der = -UnivariateSpline(x, y, s=0, k=2).derivative(2)(x)  # type:ignore
        if smoothing_sigma != 0:
            y_spl_der = np.clip(gaussian_filter1d(y_spl_der, smoothing_sigma), 0, None)

        height = height * y_spl_der.max()
        prominance = prominance * y_spl_der.max()

        peaks = find_peaks(
            y_spl_der,
            height=height,
            prominence=prominance,
        )[0]

        self.peaks = np.array([x[peaks], y[peaks]]).T

    def _expand_windows(self, windows: np.ndarray, region: Window):
        if len(windows) == 1:
            windows[0][0] = region[0]
            windows[0][1] = region[1]
            return windows

        assert isinstance(self.peaks, np.ndarray)

        for i, window in enumerate(windows):
            if i == 0:
                window[0] = region[0]
                mid_x = window[1] + (windows[i + 1][0] - window[1]) / 2

                nearest_peak_x_up = nearest_ciel(self.peaks[:, 0], mid_x)
                nearest_peak_x_down = nearest_floor(self.peaks[:, 0], mid_x)

                y_window = self.y[
                    np.logical_and(
                        self.x >= nearest_peak_x_down - 0.5 * self.step,
                        self.x <= nearest_peak_x_up,
                    )
                ]
                x_window = self.x[
                    np.logical_and(
                        self.x >= nearest_peak_x_down,
                        self.x <= nearest_peak_x_up,
                    )
                ]

                x_valley = x_window[y_window.argmin()]

                window[1] = nearest_floor(self.x, x_valley)
            elif i == windows.shape[0] - 1:
                window[0] = windows[i - 1][1] + self.step
                window[1] = region[1]
            else:
                window[0] = windows[i - 1][1] + self.step
                mid_x = window[1] + (windows[i + 1][0] - window[1]) / 2

                nearest_peak_x_up = nearest_ciel(self.peaks[:, 0], mid_x)
                nearest_peak_x_down = nearest_floor(self.peaks[:, 0], mid_x)

                y_window = self.y[
                    np.logical_and(
                        self.x >= nearest_peak_x_down - 0.5 * self.step,
                        self.x <= nearest_peak_x_up,
                    )
                ]
                x_window = self.x[
                    np.logical_and(
                        self.x >= nearest_peak_x_down - 0.5 * self.step,
                        self.x <= nearest_peak_x_up,
                    )
                ]

                x_valley = x_window[y_window.argmin()]

                window[1] = nearest_floor(self.x, x_valley)

        return windows

    def _fit_window(self, window: Window, max_iterations: int | None = None):
        if not isinstance(self.peaks, np.ndarray):
            self.find_peaks(window)
        if not isinstance(self.peaks, np.ndarray):
            raise ValueError("No peaks found in selected region")

        if max_iterations == -1:
            max_iterations = None

        peaks_xy = self.peaks

        x_min, x_max = window[0] - self.step * 0.5, window[1]

        selected_peaks_xy = peaks_xy[
            np.logical_and(
                peaks_xy[:, 0] >= x_min,
                peaks_xy[:, 0] <= x_max,
            )
        ]

        if len(selected_peaks_xy) == 0:
            return None

        win_x, win_y = (
            self.x[np.logical_and(self.x >= x_min, self.x <= x_max)],
            self.y[np.logical_and(self.x >= x_min, self.x <= x_max)],
        )

        peaks = [
            Peak(
                x,
                y,
                Spectrum.from_data(np.array([win_x, win_y]).T),
            )
            for x, y in selected_peaks_xy
        ]

        params = Parameters()
        for peak in peaks:
            peak.estimate_params()
            params.add(
                f"{peak.prefix}sigma", min=0.1, max=2, value=peak.sigma_estimated
            )
            params.add(f"{peak.prefix}fwhm", value=peak.fwhm_estimated)
            params.add(f"{peak.prefix}amplitude", min=0, value=peak.amplitude_estimated)
            params.add(f"{peak.prefix}fraction", min=0, max=1, value=0.5)
            params.add(
                f"{peak.prefix}center",
                min=peak.x - 0.75 * self.step,
                max=peak.x + 0.75 * self.step,
                value=peak.x,
            )

        model: Model = np.sum([peak.model for peak in peaks])

        try:
            result = model.fit(win_y, params=params, x=win_x, max_nfev=max_iterations)
        except TypeError as e:
            logger.error(f"Wrong fit parameters: {e}")
            return None

        components = list(result.eval_components().values())
        parameters = result.params
        for peak_i, peak in enumerate(peaks):
            peak.fitted = Spectrum.from_data(
                np.array([peak.data.x, components[peak_i]]).T
            )
            peak.sigma_fitted = parameters[f"peak_{peak.id}sigma"].value
            peak.amplitude_fitted = parameters[f"peak_{peak.id}amplitude"].value
            peak.fraction_fitted = parameters[f"peak_{peak.id}fraction"].value
            peak.center_fitted = parameters[f"peak_{peak.id}center"].value
            peak.fwhm_fitted = parameters[f"peak_{peak.id}fwhm"].value

        fit_result: FitResult = {"model_result": result, "peaks": peaks}

        return fit_result

    def _generate_split_ids(
        self,
        peak_ids: set[int],
        peaks: npt.NDArray,
        threshold: float,
        threshold_type: Literal["Relative", "Absolute"],
    ):
        while True:
            peak_ids_init = peak_ids.copy()
            peak_split_ids = sorted(peak_ids)
            for i, j in zip(peak_split_ids, peak_split_ids[1:]):
                peak_win_x = peaks[i:j][:, 0]
                if peak_win_x.size > 1:
                    x_min, x_max = peak_win_x.min(), peak_win_x.max()
                    peak_mask = np.logical_and(self.x >= x_min, self.x <= x_max)
                    win_x, win_y = self.x[peak_mask], self.y[peak_mask]

                    if threshold_type == "Absolute":
                        mask = win_y <= threshold
                    else:
                        mask = win_y / win_y.max() <= threshold

                    for x in win_x[mask]:
                        split_peak_i = np.searchsorted(peaks[:, 0], [x], side="left")[0]
                        if peaks[split_peak_i][0] >= x_min:
                            peak_ids.add(split_peak_i)

            if len(peak_ids) == len(peak_ids_init):
                break

    def _merge_small_windows(
        self,
        windows: npt.NDArray,
        x_threshold: float,
    ):
        for _ in range(100):
            small_ids = set(
                [
                    i
                    for i, w in enumerate(windows)
                    if self.x[np.logical_and(self.x >= w[0], self.x <= w[1])].size
                    <= max(x_threshold, 8)
                ]
            )
            small_ids_init = small_ids.copy()

            for i in small_ids_init:
                if i in small_ids:
                    if len(windows) <= i + 1:
                        continue
                    windows[i, 1] = windows[i + 1, 1]
                    windows = np.delete(windows, i + 1, axis=0)
                    small_ids.remove(i)

            if len(small_ids_init) == len(small_ids):
                break

        return windows

    def generate_fitting_windows(
        self,
        region: Window,
        x_threshold=8.000,
        y_threshold=0.001,
        threshold_type: Literal["Relative", "Absolute"] = "Relative",
    ):
        if self.peaks is None:
            self.find_peaks()
            assert isinstance(self.peaks, np.ndarray)

        if len(self.peaks) == 0:
            self.fitting_windows = [region]
            return

        region_min, region_max = region

        peak_xs = self.peaks[:, 0]
        selected_peaks = self.peaks[
            np.logical_and(peak_xs >= region_min, peak_xs <= region_max)
        ]

        peak_x_diffs = np.diff(selected_peaks[:, 0])
        x_split_ids = [
            np.where(np.isclose(peak_x_diffs, v))[0][0] + 1
            for v in peak_x_diffs[peak_x_diffs >= x_threshold]
        ]

        peak_ids = set([0, *x_split_ids, len(selected_peaks)])

        self._generate_split_ids(peak_ids, selected_peaks, y_threshold, threshold_type)

        final_split_ids = sorted(peak_ids)[1:-1]

        partitioned_peaks = partition(selected_peaks, final_split_ids)

        if len(partitioned_peaks) == 0:
            self.fitting_windows = []
            return

        if len(partitioned_peaks[0]) == 0:
            self.fitting_windows = []
            return

        windows = np.array(
            [[np.min(i[:, 0]), np.max(i[:, 0])] for i in partitioned_peaks]
        )

        windows = self._expand_windows(windows, region)

        if len(windows) == 1:
            self.fitting_windows = windows.tolist()
            return

        windows = self._merge_small_windows(windows, x_threshold)
        self.fitting_windows = windows.tolist()

    @log_exec_time
    @partial(loading_indicator, message="Fitting series")
    def fit_windows_parallel(self, windows: Windows, max_iterations: int | None = None):
        if len(windows) == 0:
            return

        with Pool(processes=CPU_COUNT) as pool:
            pool.restart()
            try:
                fitting_func = partial(self._fit_window, max_iterations=max_iterations)
                result = pool.amap(fitting_func, windows)
                while not result.ready():
                    yield (1 - result._number_left / len(windows)) * 100
                    time.sleep(LOADING_INDICATOR_FETCH_DELAY_S)

                    if dpg.is_key_down(dpg.mvKey_Escape):
                        yield "aborted"
                        pool.terminate()
                        pool.join()
                        return None

                fit_results: list[FitResult | None] = result.get()
            except Exception as e:
                logger.error(f"Error while fitting: {e}")
                pool.terminate()
                pool.join()

                return None

            finally:
                pool.close()
                pool.join()

        try:
            for r in fit_results:
                if r is None:
                    raise ValueError(
                        "Fitting failed in some windows. \
                            Possibly too many fit parameters for a given window size"
                    )
        except ValueError as e:
            logger.error(f"Fitting result error: {e}")
            return None

        assert isinstance(fit_results, list)

        fit = Fit.from_fit_results(self, windows, fit_results)

        if not self.series:
            return

        self.series.fits[fit.id] = fit


@define(repr=False)
class Peak:
    x: float
    y: float
    data: Spectrum
    fitted: Spectrum = field(init=False)
    model: Any = field(init=False, default=None)
    id: int = field(init=False, default=0)
    prefix: str = field(init=False, default="")
    fwhm_estimated: float | None = field(init=False, default=None)
    amplitude_estimated: float | None = field(init=False, default=None)
    sigma_estimated: float | None = field(init=False, default=None)
    fwhm_fitted: float | None = field(init=False, default=None)
    amplitude_fitted: float | None = field(init=False, default=None)
    sigma_fitted: float | None = field(init=False, default=None)
    fraction_fitted: float | None = field(init=False, default=None)
    center_fitted: float | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.id = uuid.uuid4().int & (1 << 64) - 1
        self.prefix = f"peak_{self.id}"
        self.model = PseudoVoigtModel(prefix=self.prefix)

    def estimate_params(self):
        win_x = self.data.x
        win_y = self.data.y

        hwhm = 0.0
        start_x_index: int = np.where(np.isclose(win_x, self.x))[0][0]
        curr_x_index: int = start_x_index

        # go right
        while curr_x_index < len(win_x) - 1:
            if win_y[curr_x_index] > win_y[start_x_index]:
                break
            if win_y[curr_x_index] <= 0.5 * win_y[start_x_index]:
                break
            curr_x_index += 1

        hwhm = win_x[curr_x_index] - win_x[start_x_index]
        integration_x = win_x[
            np.logical_and(win_x >= self.x, win_x <= win_x[curr_x_index])
        ]
        integration_y = win_y[
            np.logical_and(win_x >= self.x, win_x <= win_x[curr_x_index])
        ]
        area = np.trapz(integration_y, integration_x)

        # go left
        if curr_x_index - start_x_index == 1:
            curr_x_index = start_x_index
            while curr_x_index >= 0:
                if win_y[curr_x_index] > win_y[start_x_index]:
                    break
                if win_y[curr_x_index] <= 0.5 * win_y[start_x_index]:
                    break
                curr_x_index -= 1

                hwhm: float = -win_x[curr_x_index] + win_x[start_x_index]

                integration_x = win_x[
                    np.logical_and(
                        win_x <= self.x,
                        win_x >= win_x[curr_x_index],
                    )
                ]
                integration_y = win_y[
                    np.logical_and(
                        win_x <= self.x,
                        win_x >= win_x[curr_x_index],
                    )
                ]

                area = np.trapz(integration_y, integration_x)

        if hwhm == 0.0:
            self.fwhm_estimated = 0.0
            self.sigma_estimated = 0.3
            self.amplitude_estimated = self.y
            return

        self.fwhm_estimated = hwhm * 2
        self.sigma_estimated = self.fwhm_estimated / 3.6
        self.amplitude_estimated = area / self.sigma_estimated * 0.55


class FitResult(TypedDict):
    model_result: ModelResult
    peaks: list[Peak]


@define(repr=False)
class Sample:
    directory: Path
    name: str = field(init=False)
    spectra: list[Spectrum] = field(init=False, factory=list)
    common_x: bool = field(default=True)

    def __attrs_post_init__(self):
        specra_files = natsorted(self.directory.rglob("*.spec"))

        x = Spectrum.from_file(specra_files[0]).x if self.common_x else None

        self.spectra = [Spectrum.from_file(f, common_x=x) for f in specra_files]
        self.name = self.directory.name

    def averaged(self, drop_first: int):
        data = np.array([s.raw_spectral_data for s in self.spectra[drop_first:]])
        return Spectrum.from_data(data.mean(axis=0))


@define(repr=False)
class Fit:
    id: str = field(init=False, default=0)
    data: Spectrum
    r_squared_mean: np.float_ = field(init=False)
    r_squared_min: np.float_ = field(init=False)
    r_squared_st_dev: np.float_ = field(init=False)
    windows: Windows
    fit_results: list[FitResult]
    components: Mapping[str, Peak]
    selected: bool = field(init=False, default=True)
    x_bounds: tuple[float, float] = field(init=False, default=(0, 0))

    def __attrs_post_init__(self):
        self.id = uuid.uuid4().urn

        r_squared = np.array([r["model_result"].rsquared for r in self.fit_results])
        self.r_squared_mean = np.mean(r_squared)
        self.r_squared_min = np.min(r_squared)
        self.r_squared_st_dev = np.std(r_squared)

        self.x_bounds = (self.data.x[0], self.data.x[-1])

    @classmethod
    def from_fit_results(
        cls,
        parent_spectrum: Spectrum,
        windows: Windows,
        fit_results: list[FitResult | None],
    ):
        for r in fit_results:
            if r is None:
                raise ValueError

        x = parent_spectrum.x[
            np.logical_and(
                parent_spectrum.x >= windows[0][0] - 0.5 * parent_spectrum.step,
                parent_spectrum.x <= windows[-1][1],
            )
        ]
        y = np.concatenate(
            [r["model_result"].best_fit for r in fit_results]  # type:ignore
        )

        components = {}
        for result in fit_results:
            assert result
            for peak in result["peaks"]:
                components[peak.id] = peak

        data = Spectrum.from_data(np.array([x, y]).T)

        return cls(
            data=data,
            fit_results=fit_results,  # type:ignore
            windows=windows,
            components=components,
        )

    @property
    def printable_x_bounds(self):
        return f"{self.x_bounds[0]:.0f}..{self.x_bounds[1]:.0f}"

    @property
    def n_peaks(self):
        return len(self.components)

    @property
    def windows_total(self):
        return len(self.windows)


@define(repr=False)
class Series:
    directory: Path
    name: str | None = field(default=None)
    samples: list[Sample] = field(init=False, factory=list)
    id: str = field(init=False, default=0)
    selected: bool = field(init=False, default=False)
    color: list[int] | None = field(init=False, default=None)
    common_x: bool = field(default=True)
    sample_drop_first: int = field(default=0)
    spectra_total: int = field(init=False)
    samples_total: int = field(init=False)
    _averaged: Spectrum | None = field(init=False, default=None)
    fits: dict[str, Fit] = field(init=False, factory=dict)

    def __attrs_post_init__(self):
        self.id = uuid.uuid4().urn

        child_dirs = natsorted(d for d in self.directory.iterdir() if d.is_dir())
        if not child_dirs:
            self.samples = [Sample(self.directory, self.common_x)]
        else:
            self.samples = [Sample(d, self.common_x) for d in child_dirs]

        if self.name is None:
            self.name = "_".join([self.directory.parent.name, self.directory.name])

        self.spectra_total = np.sum([len(s.spectra) for s in self.samples])
        self.samples_total = len(self.samples)

    @property
    def averaged(self):
        if self._averaged is None:
            spectra: list[Spectrum] = flatten(
                [s.averaged(self.sample_drop_first) for s in self.samples]
            )
            data = np.array([s.raw_spectral_data for s in spectra])
            spectrum = Spectrum.from_data(data.mean(axis=0))
            self._averaged = spectrum
            self._averaged.series = self
            return spectrum

        return self._averaged


@define
class Project:
    directory: Path
    series_dirs: list[Path] = field(init=False, factory=list)
    series: Mapping[str, Series] = field(init=False, factory=dict)
    plotted_series_ids: set[str] = field(init=False, default=set())
    plotted_fits_ids: set[str] = field(init=False, default=set())
    common_x: bool = field(default=False)
    sample_drop_first: int = field(default=0)

    def __attrs_post_init__(self):
        self.validate_directory()
        self.load_series(
            common_x=self.common_x, sample_drop_first=self.sample_drop_first
        )

    def validate_directory(self):
        if not self.directory.exists():
            raise ValueError

        possible_series_dirs = natsorted(
            f for f in self.directory.iterdir() if f.is_dir()
        )

        if not any(possible_series_dirs):
            raise ValueError

        self.series_dirs = possible_series_dirs

    @property
    def selected_series(self):
        selected_series = [s for s in self.series.values() if s.selected]
        selected_series = natsorted(selected_series, key=lambda s: s.name)
        return selected_series

    @log_exec_time
    @partial(loading_indicator, message="Loading series")
    def load_series(self, common_x, sample_drop_first):
        with Pool(processes=CPU_COUNT) as pool:
            pool.restart()
            try:
                result = pool.amap(
                    partial(
                        Series, common_x=common_x, sample_drop_first=sample_drop_first
                    ),
                    self.series_dirs,
                    chunksize=1,
                )
                while not result.ready():
                    yield (1 - result._number_left / len(self.series_dirs)) * 100
                    time.sleep(LOADING_INDICATOR_FETCH_DELAY_S)
                self.series = {s.id: s for s in result.get()}
            except IndexError as e:
                raise ValueError from e
            except Exception as e:
                logger.error(e)
                raise ValueError from e
            finally:
                pool.close()
                pool.join()
