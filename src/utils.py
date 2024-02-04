import logging
import os
import re
import threading
import time
from functools import cached_property, lru_cache, partial, wraps
from itertools import chain
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Generator, Iterable, ParamSpec, TypeVar

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
from attr import define, field
from natsort import natsorted

from src.filetypes import spec_to_numpy

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
        print(f"{f.__name__}: {time.perf_counter() - start_time} s.")
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
    filepath: Path
    spectral_data: npt.NDArray[np.float_] | None = field(
        init=False, default=None, repr=False
    )

    def __attrs_post_init__(self):
        self.spectral_data = spec_to_numpy(self.filepath)


@define
class Sample:
    directory: Path
    name: str = field(init=False)
    spectra: list[Spectrum] = field(init=False, factory=list)

    def __attrs_post_init__(self):
        specra_files = natsorted(self.directory.rglob("*.spec"))

        self.spectra = [Spectrum(f) for f in specra_files]
        self.name = self.directory.name

    def averaged(self, drop_first: int = 0) -> npt.NDArray[np.float_]:
        data = np.array([s.spectral_data for s in self.spectra[drop_first:]])
        return data.mean(axis=0)


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
    def averaged(self) -> npt.NDArray[np.float_]:
        spectra: list[Spectrum] = flatten([s.spectra for s in self.samples])
        data = np.array([s.spectral_data for s in spectra])
        return data.mean(axis=0)


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

    @partial(loading_indicator, message="Loading series...")
    @log_exec_time
    def load_series(self):
        for d in self.series_dirs:
            self.series.append(Series(d))
