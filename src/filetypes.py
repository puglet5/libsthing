import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Pattern, Tuple, TypedDict

import chardet
import numpy as np
from pyarrow import csv

PA_READ_OPTIONS = csv.ReadOptions(skip_rows=2, autogenerate_column_names=True)
PA_PARSE_OPTIONS = csv.ParseOptions(delimiter="\t")
PA_CONVERT_OPTIONS = csv.ConvertOptions(decimal_point=",")
logger = logging.getLogger(__name__)


class Filetype(TypedDict):
    columns: tuple[int, ...]
    method: str
    field_delimiter: str
    radix_point: str
    split_indices: list
    line_matchers: list[Pattern[str]]


filetypes: dict[str, Filetype] = {
    "libs.spectable": {
        "columns": (0, 1),
        "method": "libs",
        "radix_point": "\\,",
        "field_delimiter": "\t",
        "split_indices": [2],
        "line_matchers": [
            re.compile("^Wavelenght[ \t]+Spectrum$"),
            re.compile("^Integration delay[ \t]+[+-]?([0-9]*[,])?[0-9]+$"),
            re.compile("^[+-]?([0-9]*[,])?[0-9]+\t[+-]?([0-9]*[,])?[0-9]+$"),
        ],
    },
    "libs.spec": {
        "columns": (0, 1),
        "method": "libs",
        "radix_point": "\\,",
        "field_delimiter": "\t",
        "split_indices": [2],
        "line_matchers": [
            re.compile("^[0-9]+$"),
            re.compile("^[0-9]+$"),
            re.compile("^[+-]?([0-9]*[,])?[0-9]+[ \t]+[+-]?([0-9]*[,])?[0-9]+$"),
        ],
    },
}


def detect_filetype(file: BytesIO):
    try:
        enc = detect_encoding(file)
        filetype = None
        for _, ft in filetypes.items():
            res_list = []
            for r in ft["line_matchers"]:
                line = file.readline().decode(enc)
                res = re.match(r, line.strip())
                res_list.append(res)
            file.seek(0)
            if None not in res_list:
                filetype = ft
                break

        return filetype
    except Exception as e:
        logger.error(f"Error detecting filetype: {e}")
        return None


def detect_encoding(file: BytesIO):
    enc = chardet.detect(file.read())["encoding"] or "utf-8"
    file.seek(0)
    return enc


def multi_sub(sub_pairs: list[Tuple[str, str]], string: str):
    def repl_func(m):
        return next(
            repl for (_, repl), group in zip(sub_pairs, m.groups()) if group is not None
        )

    pattern = "|".join("({})".format(patt) for patt, _ in sub_pairs)
    return re.sub(pattern, repl_func, string, flags=re.U)


def partition(alist: list, indices: list):
    return [alist[i:j] for i, j in zip([0] + indices, indices + [None])]


def spec_to_numpy(filepath: Path):
    return np.array(
        csv.read_csv(
            filepath,
            read_options=PA_READ_OPTIONS,
            parse_options=PA_PARSE_OPTIONS,
            convert_options=PA_CONVERT_OPTIONS,
        )
    )
