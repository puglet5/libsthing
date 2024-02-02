import logging
import os
import threading
from dataclasses import dataclass, field
from functools import wraps
from io import BytesIO
from pathlib import Path
from time import time
from typing import Callable, Generator, ParamSpec, TypeVar

import coloredlogs
import dearpygui.dearpygui as dpg
import numpy as np
import numpy.typing as npt

from src.filetypes import convert_to_csv

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
logging.basicConfig(filename=Path(ROOT_DIR, "log/main.log"), filemode="a")
coloredlogs.install(level="DEBUG")
logger = logging.getLogger(__name__)

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


@dataclass(slots=True)
class Spectrum:
    filepath: Path
    spectral_data: npt.NDArray[np.float64] | None = field(init=False, default=None)

    def __post_init__(self):
        with open(self.filepath, "rb") as fh:
            buf = BytesIO(fh.read())
            csv = convert_to_csv(buf, self.filepath)
            if csv is None:
                raise ValueError

            self.spectral_data = np.loadtxt(csv, delimiter=",")
