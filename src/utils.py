import dis
import logging
import os
import threading
import time
from functools import cached_property, partial, wraps
from itertools import chain
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Generator, Iterable, ParamSpec, TypeVar
from xml.etree.ElementInclude import include

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
import pyarrow as pa
from attr import define, field
from cycler import V
from natsort import natsorted
from pyarrow import csv
from pybaselines import Baseline
from scipy.signal import find_peaks

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

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=Path(ROOT_DIR, "log/main.log"), filemode="a")
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

CPU_COUNT = cpu_count()

T = TypeVar("T")
P = ParamSpec("P")


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


def loading_indicator(f: Callable[P, T], message: str) -> Callable[P, T]:  # type:ignore
    @wraps(f)
    def _wrapper(*args, **kwargs):
        dpg.configure_item("loading_indicator_message", label=message.center(30))
        threading.Timer(0.1, show_loading_indicator).start()
        result = f(*args, **kwargs)
        threading.Timer(0.1, hide_loading_indicator).start()
        return result

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
                dpg.set_value("table_progress", progress)
                dpg.configure_item("table_progress", overlay=f"{progress*100:.0f}%")
        except StopIteration as result:
            dpg.set_value("table_progress", 0)
            dpg.configure_item("table_progress", overlay="")
            return result.value
        except TypeError:
            dpg.set_value("table_progress", 0)
            dpg.configure_item("table_progress", overlay="")
            return None

    return _wrapper  # type:ignore


def flatten(iter: list):
    if not isinstance(iter[0], Iterable):
        return iter
    return list(chain.from_iterable(iter))


@define
class Spectrum:
    file: Path | None = field(default=None)
    spectral_data: npt.NDArray | None = field(default=None, repr=False)
    common_x: npt.NDArray | None = field(default=None, repr=False)

    def __attrs_post_init__(self):
        if self.spectral_data is None and self.file is not None:
            self.spectral_data = self.spec_to_numpy()

        if self.spectral_data is None and self.file is None:
            raise ValueError

    @classmethod
    def from_file(cls, file: Path, common_x: npt.NDArray | None = None):
        return cls(file=file, common_x=common_x)

    @classmethod
    def from_data(cls, data: npt.NDArray):
        return cls(spectral_data=data)

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
        else:
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

    @cached_property
    def x(self):
        if self.spectral_data is None:
            raise ValueError
        return self.spectral_data[:, 0]

    @cached_property
    def y(self):
        if self.spectral_data is None:
            raise ValueError
        return self.spectral_data[:, 1]

    @cached_property
    def baseline(self):
        baseline_fitter = Baseline(x_data=self.x)
        bkg = baseline_fitter.snip(
            self.y, max_half_window=40, smooth_half_window=2, filter_order=2
        )[0]
        return bkg

    @cached_property
    def spectral_data_corrected(self):
        if self.spectral_data is None:
            raise ValueError

        y_clamped = np.clip(self.y - self.baseline, 0, None)

        return np.array([self.x, y_clamped]).T

    def find_peaks(
        self,
        width: tuple[int, int] = (0, 5),
        distance=1,
        threshold=0.1,
        height=10,
        y: npt.NDArray | None = None,
    ):
        if not isinstance(y, np.ndarray):
            y = self.spectral_data_corrected[:, 1]
        assert y is not None

        peaks = find_peaks(
            y,
            width=width,
            distance=distance,
            threshold=threshold,
            height=height,
        )[0]

        return np.array([self.x[peaks], y[peaks]]).T


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
        data = np.array([s.spectral_data for s in self.spectra[drop_first:]])
        return Spectrum.from_data(data.mean(axis=0))


@define
class Series:
    directory: Path
    name: str | None = field(default=None)
    samples: list[Sample] = field(init=False, factory=list)

    def __attrs_post_init__(self):
        child_dirs = natsorted(d for d in self.directory.iterdir() if d.is_dir())
        if not child_dirs:
            self.samples = [Sample(self.directory)]
        else:
            self.samples = [Sample(d) for d in child_dirs]

        if self.name is None:
            self.name = "_".join([self.directory.parent.name, self.directory.name])

    @cached_property
    def averaged(self):
        spectra: list[Spectrum] = flatten([s.spectra for s in self.samples])
        data = np.array([s.spectral_data for s in spectra])
        spectrum = Spectrum.from_data(data.mean(axis=0))
        return spectrum


@define
class Project:
    directory: Path
    series_dirs: list[Path] = field(init=False, factory=list)
    series: list[Series] = field(init=False, factory=list)

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

        for directory in possible_series_dirs:
            if not directory.name.isnumeric():
                raise ValueError

        self.series_dirs = possible_series_dirs

    @partial(loading_indicator, message=f"Loading series...")
    @log_exec_time
    def load_series(self):
        pool = Pool(processes=len(self.series_dirs))
        self.series = pool.map(Series, self.series_dirs)
        pool.terminate()
        pool.join()
