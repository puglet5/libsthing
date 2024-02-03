import logging
import os
import threading
import time
from functools import cached_property, wraps
from itertools import chain
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, Generator, Iterable, ParamSpec, TypeVar

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt
from attr import define, field

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


@define(repr=False, eq=False)
class Spectrum:
    filepath: Path
    spectral_data: npt.NDArray[np.float_] | None = field(init=False, default=None)

    def __attrs_post_init__(self):
        self.spectral_data = spec_to_numpy(self.filepath)


@define(repr=False, eq=False)
class LIBSSeries:
    name: str
    folder: Path
    spectra: list[Spectrum] | list[list[Spectrum]] = field(init=False, factory=list)

    def __attrs_post_init__(self):
        libs_files = [
            Path(os.path.join(dp, f))
            for dp, _, filenames in os.walk(self.folder)
            for f in filenames
            if os.path.splitext(f)[1] == ".spec"
        ]

        pool = Pool(processes=CPU_COUNT)
        self.spectra = pool.map(Spectrum, libs_files)

        pool.terminate()
        pool.join()

    @cached_property
    def averaged(self) -> npt.NDArray[np.float_]:
        data = np.array([s.spectral_data for s in flatten(self.spectra)])
        return data.mean(axis=0)

    def average_per_sample(self) -> npt.NDArray[np.float_]: ...
