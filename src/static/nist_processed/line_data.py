from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
from pyarrow import csv

from src.utils import Window

PA_READ_OPTIONS = csv.ReadOptions(
    skip_rows=1,
    column_names=[
        "element",
        "level",
        "wavelength",
        "intensity",
    ],
)
PA_PARSE_OPTIONS = csv.ParseOptions(quote_char=False)
PA_CONVERT_OPTIONS = csv.ConvertOptions(
    check_utf8=False,
    quoted_strings_can_be_null=False,
    column_types={
        "element": pa.string(),
        "level": pa.int8(),
        "wavelength": pa.float32(),
        "intensity": pa.float32(),
    },
)

EMISSION_LINE_DATA: pd.DataFrame = csv.read_csv(
    Path("./src/static/nist_processed/all_clean.csv").absolute(),
    read_options=PA_READ_OPTIONS,
    parse_options=PA_PARSE_OPTIONS,
    convert_options=PA_CONVERT_OPTIONS,
).to_pandas()


def get_emission_data(symbol: str, wl_window: Window, max_ionization_level: int = 3):
    emission_data = (
        EMISSION_LINE_DATA.groupby("element")
        .get_group(symbol)
        .drop(columns=["element"])
        .reset_index(drop=True)
    )

    emission_data = emission_data[emission_data["wavelength"].between(*wl_window)]
    emission_data = emission_data[emission_data["level"].le(max_ionization_level)]
    
    return emission_data


def element_plot_data(element_data: pd.DataFrame):
    data = element_data.drop(columns=["level"]).to_numpy()
    data_zeroes = data.copy()
    data_zeroes[:, 1] = 0
    nans = np.empty(data.shape)
    nans[:] = np.nan

    plot_data = np.empty(
        (data_zeroes.shape[0] + data.shape[0] + nans.shape[0], data.shape[1])
    )

    plot_data[::3, :] = data_zeroes
    plot_data[1::3, :] = data
    plot_data[2::3, :] = nans
    return plot_data[:-1]
