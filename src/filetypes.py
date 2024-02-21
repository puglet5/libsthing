import logging
import re
from io import BytesIO
from typing import Pattern, TypedDict

import chardet

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
