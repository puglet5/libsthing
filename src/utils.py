import gc
import logging
import os
import re
import threading
import time
import uuid
from functools import cached_property, partial, wraps
from itertools import chain
from multiprocessing import cpu_count
from pathlib import Path
from types import NoneType
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Literal,
    Mapping,
    ParamSpec,
    Tuple,
    TypeVar,
)

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attr import define, field
from lmfit import Model, Parameters
from lmfit.model import ModelResult
from lmfit.models import PseudoVoigtModel, VoigtModel
from natsort import natsorted
from pathos.multiprocessing import ProcessingPool as Pool
from pyarrow import csv
from pybaselines import Baseline
from scipy.interpolate import UnivariateSpline
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter

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

LOADING_INDICATOR_FETCH_DELAY_S = 0.01
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=Path(ROOT_DIR, "log/main.log"), filemode="a")
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

CPU_COUNT = cpu_count()

T = TypeVar("T")
P = ParamSpec("P")

Window = tuple[float, float]
Windows = list[Window]


def log_exec_time(f: Callable[P, T]) -> Callable[P, T]:  # type:ignore
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


def loading_indicator(
    f: Callable[P, Generator[float | int, Any, Any]], message: str
) -> Callable[P, Generator[float | int, Any, Any]]:  # type:ignore
    @wraps(f)
    def _wrapper(*args, **kwargs):
        dpg.configure_item("loading_indicator_message", label=message.center(30))
        threading.Timer(0.1, show_loading_indicator).start()
        progress_generator = f(*args, **kwargs)

        try:
            while True:
                progress = next(progress_generator)
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
            dpg.configure_item(
                "loading_indicator_message",
                label=f"{message}: 100%".center(30),
            )
            threading.Timer(0.5, hide_loading_indicator).start()

    return _wrapper  # type:ignore


def progress_bar(
    f: Callable[P, Generator[float, None, None]]
) -> Callable[P, T]:  # type:ignore
    @wraps(f)
    def _wrapper(*args, **kwargs):
        progress_generator = f(*args, **kwargs)
        try:
            while True:
                progress = next(progress_generator)
                dpg.set_value("progress_bar", progress)
                dpg.configure_item("progress_bar", overlay=f"{progress*100:.0f}%")
        except StopIteration as result:
            dpg.set_value("progress_bar", 0)
            dpg.configure_item("progress_bar", overlay="")
            return result.value
        except TypeError:
            dpg.set_value("progress_bar", 0)
            dpg.configure_item("progress_bar", overlay="")
            return None

    return _wrapper  # type:ignore


def flatten(iter: list):
    if not isinstance(iter[0], Iterable):
        return iter
    return list(chain.from_iterable(iter))


def multi_sub(sub_pairs: list[tuple[str, str]], string: str):
    def repl_func(m):
        return next(
            repl for (_, repl), group in zip(sub_pairs, m.groups()) if group is not None
        )

    pattern = "|".join("({})".format(patt) for patt, _ in sub_pairs)
    return re.sub(pattern, repl_func, string, flags=re.U)


def partition(alist, indices: list):
    return [alist[i:j] for i, j in zip([0] + indices, indices + [None])]


def nearest_ciel(arr: np.ndarray, val: float):
    i = np.searchsorted(arr, val)
    return arr[i]


def nearest_floor(arr: np.ndarray, val: float):
    i = np.searchsorted(arr, val)
    return arr[i - 1]


def nearest(arr: np.ndarray, val: float):
    return arr.flat[np.abs(arr - val).argmin()]


@define
class Peak:
    x: float
    y: float
    model: Any = field(init=False, default=None)
    id: int = field(init=False, default=0)
    prefix: str = field(init=False, default="")
    fwhm: float | None = field(init=False, default=None)
    amplitude: float | None = field(init=False, default=None)
    sigma: float | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.id = uuid.uuid4().int & (1 << 64) - 1
        self.prefix = f"peak{self.id}"
        self.model = PseudoVoigtModel(prefix=self.prefix)

    def estimate_params(self, win_x, win_y):
        hwhm = 0
        start_x_index = np.where(np.isclose(win_x, self.x))[0][0]
        curr_x_index = start_x_index

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

                hwhm = -win_x[curr_x_index] + win_x[start_x_index]

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

        if hwhm == 0:
            self.fwhm = 0
            self.sigma = 0.3
            self.amplitude = self.y
            return

        self.fwhm = hwhm * 2
        self.sigma = self.fwhm / 3.6
        self.amplitude = area / self.sigma * 0.55


@define
class Spectrum:
    file: Path | None = field(default=None)
    raw_spectral_data: npt.NDArray | None = field(default=None, repr=False)
    processed_spectral_data: npt.NDArray | None = field(
        default=None, init=False, repr=False
    )
    common_x: npt.NDArray | None = field(default=None, repr=False)
    fit_result: ModelResult | None = field(init=False, default=None)
    peaks: npt.NDArray | None = field(init=False, default=None)
    fitting_windows: Windows = field(init=False, factory=list)
    fitted: npt.NDArray | None = field(init=False, default=None)

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
            )
        else:
            x = self.common_x
            y = np.array(
                csv.read_csv(
                    self.file,
                    read_options=PA_READ_OPTIONS,
                    parse_options=PA_PARSE_OPTIONS,
                    convert_options=PA_CONVERT_OPTIONS_SKIP_X,
                ),
            )

            return np.c_[x, y]

    @property
    def x(self):
        if self.raw_spectral_data is None:
            raise ValueError
        return self.raw_spectral_data[:, 0]

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
    def x_limits(self) -> Window:
        return (self.x.min(), self.x.max())

    def baseline(
        self,
        y: np.ndarray,
        method: Literal["SNIP", "Adaptive minmax", "Polynomial"],
        method_params: dict,
    ) -> np.ndarray:
        baseline_fitter = Baseline(x_data=self.x)
        if method == "SNIP":
            bkg = baseline_fitter.snip(y, **method_params)[0]
        if method == "Adaptive minmax":
            bkg = baseline_fitter.adaptive_minmax(y)[0]
        if method == "Polynomial":
            bkg = baseline_fitter.penalized_poly(y, poly_order=8)[0]

        return bkg

    def process_spectral_data(
        self,
        normalized=False,
        normalization_range: Window = (1, -1),
        baseline_clip=True,
        baseline_removal: Literal[
            "None", "SNIP", "Adaptive minmax", "Polynomial"
        ] = "SNIP",
        baseline_params: dict = {},
    ):
        if (data := self.raw_spectral_data) is None:
            raise ValueError

        y = data.T[1]

        if baseline_removal != "None":
            bkg = self.baseline(
                y, method=baseline_removal, method_params=baseline_params
            )

            if baseline_clip:
                y = np.clip(y - bkg, 0, None)
            else:
                y = y - bkg

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
                y = y / y[np.logical_and(self.x > norm_min, self.x < norm_max)].max()

        self.processed_spectral_data = np.array([self.x, y]).T

    def find_peaks(
        self,
        region: Window | None = None,
        height=0.005,
        prominance=0.001,
        smoothing_sigma=1,
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

    def fit_window(self, window: Window, max_iterations: int | None = None):
        assert isinstance(self.peaks, np.ndarray)

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

        selected_peaks = [Peak(*xy) for xy in selected_peaks_xy]

        win_x, win_y = (
            self.x[np.logical_and(self.x >= x_min, self.x <= x_max)],
            self.y[np.logical_and(self.x >= x_min, self.x <= x_max)],
        )

        params = Parameters()
        for peak in selected_peaks:
            peak.estimate_params(win_x, win_y)
            params.add(f"{peak.prefix}sigma", min=0.1, max=2, value=peak.sigma)
            params.add(f"{peak.prefix}amplitude", min=0, value=peak.amplitude)
            params.add(f"{peak.prefix}fraction", min=0, max=1, value=0.5)
            params.add(
                f"{peak.prefix}center",
                min=peak.x - 0.75 * self.step,
                max=peak.x + 0.75 * self.step,
                value=peak.x,
            )

        model: Model = np.sum([peak.model for peak in selected_peaks])

        try:
            result = model.fit(win_y, params=params, x=win_x, max_nfev=max_iterations)
        except TypeError as e:
            logger.error(f"Wrong fit parameters: {e}")
            self.fit_result = None
            return self.fit_result

        self.fit_result = result

        return self.fit_result

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

        selected_peak_xs = selected_peaks[:, 0]
        peak_x_diffs = np.diff(selected_peak_xs)
        x_split_ids = [
            np.where(np.isclose(peak_x_diffs, v))[0][0] + 1
            for v in peak_x_diffs[peak_x_diffs >= x_threshold]
        ]

        peak_ids = {0, *x_split_ids, len(selected_peaks)}

        # subdivide peaks into windows where win_y / win_y.max() <= y_threshold
        while True:
            peak_ids_init = peak_ids.copy()
            peak_split_ids = sorted(peak_ids)
            for i, j in zip(peak_split_ids, peak_split_ids[1:]):
                peak_win_x = selected_peaks[i:j][:, 0]
                if peak_win_x.size > 1:
                    x_min, x_max = peak_win_x.min(), peak_win_x.max()
                    peak_mask = np.logical_and(self.x >= x_min, self.x <= x_max)
                    win_x, win_y = self.x[peak_mask], self.y[peak_mask]

                    if threshold_type == "Absolute":
                        mask = win_y <= y_threshold
                    else:
                        mask = win_y / win_y.max() <= y_threshold

                    for x in win_x[mask]:
                        split_peak_i = np.searchsorted(
                            selected_peak_xs, [x], side="left"
                        )[0]
                        if selected_peaks[split_peak_i][0] >= x_min:
                            peak_ids.add(split_peak_i)

            if len(peak_ids) == len(peak_ids_init):
                break

        final_split_ids = sorted(peak_ids)[1:-1]

        partitioned_peaks = partition(selected_peaks, final_split_ids)

        if len(partitioned_peaks) == 0:
            self.fitting_windows = []
            return

        if len(partitioned_peaks[0]) == 0:
            self.fitting_windows = []
            return

        partitioned_ranges = np.array(
            [[np.min(i[:, 0]), np.max(i[:, 0])] for i in partitioned_peaks]
        )

        windows = self._expand_windows(partitioned_ranges, region)

        if len(windows) == 1:
            self.fitting_windows = windows.tolist()
            return

        # ensure window size >= 8
        n = 0
        while n < 100:
            small_ids = {
                i
                for i, w in enumerate(windows)
                if self.x[np.logical_and(self.x >= w[0], self.x <= w[1])].size
                <= max(x_threshold, 8)
            }
            small_ids_init = small_ids.copy()

            for i in small_ids_init:
                if i in small_ids:
                    try:
                        windows[i, 1] = windows[i + 1, 1]
                        windows = np.delete(windows, i + 1, axis=0)
                        small_ids.discard(i)
                    except IndexError:
                        try:
                            if i < len(windows):
                                windows[i - 1, 1] = windows[i, 1]
                                windows = np.delete(windows, i, axis=0)
                                small_ids.discard(i)
                        except IndexError:
                            pass

            if len(small_ids_init) == len(small_ids):
                break

            n += 1

        self.fitting_windows = windows.tolist()

    @log_exec_time
    @partial(loading_indicator, message="Fitting series")
    def fit_windows_parallel(self, windows: Windows, max_iterations):
        if len(windows) == 0:
            return

        with Pool(processes=CPU_COUNT) as pool:
            pool.restart()
            try:
                fitting_func = partial(self.fit_window, max_iterations=max_iterations)
                result = pool.amap(fitting_func, windows)
                while not result.ready():
                    yield (1 - result._number_left / len(windows)) * 100
                    time.sleep(LOADING_INDICATOR_FETCH_DELAY_S)

                    if dpg.is_key_down(dpg.mvKey_Escape):
                        return

                fitted_results = result.get()
            except Exception as e:
                logger.error(f"Error while fitting: {e}")
                pool.terminate()
                pool.join()

                return

            finally:
                pool.close()
                pool.join()

        x = self.x[
            np.logical_and(
                self.x >= windows[0][0] - 0.5 * self.step, self.x <= windows[-1][1]
            )
        ]

        try:
            for r in fitted_results:
                if r is None:
                    raise ValueError(
                        "Fitting failed in some windows. Possibly too many fit parameters for a given window size"
                    )

            fitted_y = [r.best_fit for r in fitted_results]
            y = np.concatenate(fitted_y)
        except ValueError as e:
            logger.error(f"Fitting result error: {e}")
            return

        self.fitted = np.array([x, y]).T


@define
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

    def averaged(self, drop_first: int = 0):
        data = np.array([s.raw_spectral_data for s in self.spectra[drop_first:]])
        return Spectrum.from_data(data.mean(axis=0))


@define
class Series:
    directory: Path
    name: str | None = field(default=None)
    samples: list[Sample] = field(init=False, factory=list)
    id: str = field(init=False, default=0)
    selected: bool = field(init=False, default=True)
    color: list[int] | None = field(init=False, default=None)
    _averaged: Spectrum | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.id = str(uuid.uuid4())

        child_dirs = natsorted(d for d in self.directory.iterdir() if d.is_dir())
        if not child_dirs:
            self.samples = [Sample(self.directory)]
        else:
            self.samples = [Sample(d) for d in child_dirs]

        if self.name is None:
            self.name = "_".join([self.directory.parent.name, self.directory.name])

    @property
    def averaged(self):
        if self._averaged is None:
            spectra: list[Spectrum] = flatten([s.spectra for s in self.samples])
            data = np.array([s.raw_spectral_data for s in spectra])
            spectrum = Spectrum.from_data(data.mean(axis=0))
            self._averaged = spectrum
            return spectrum
        else:
            return self._averaged


@define
class Project:
    directory: Path
    series_dirs: list[Path] = field(init=False, factory=list)
    series: Mapping[str, Series] = field(init=False, factory=list)
    plotted_series_ids: set[str] = field(init=False, default=set())
    selected_region: list[float] | list[None] = field(init=False, default=[None, None])

    def __attrs_post_init__(self):
        self.validate_directory()
        self.load_series()

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
    def load_series(self):
        with Pool(processes=CPU_COUNT) as pool:
            pool.restart()
            try:
                result = pool.amap(Series, self.series_dirs, chunksize=1)
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
