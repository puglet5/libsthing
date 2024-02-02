import csv
import logging
import re
from io import BytesIO, StringIO
from pathlib import Path
from typing import Pattern, Tuple, TypedDict

import chardet
import numpy as np

logger = logging.getLogger(__name__)


class Filetype(TypedDict):
    columns: Tuple[int, ...]
    method: str
    field_delimiter: str
    radix_point: str
    split_indices: Tuple[int, ...]
    line_matchers: list[Pattern[str]]


filetypes: dict[str, Filetype] = {
    "libs.spectable": {
        "columns": (0, 1),
        "method": "libs",
        "radix_point": "\\,",
        "field_delimiter": "\t",
        "split_indices": (2,),
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
        "split_indices": (2,),
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
        for ft in filetypes:
            res_list = []
            for r in filetypes[ft]["line_matchers"]:
                line = file.readline().decode(enc)
                res = re.match(r, line.strip())
                res_list.append(res)
            file.seek(0)
            if None not in res_list:
                filetype = filetypes[ft]
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


def convert_to_csv(file: BytesIO, filename: Path) -> BytesIO | None:
    try:
        with file as f:
            filetype = detect_filetype(f)
            encoding = detect_encoding(f)

            if filetype is None:
                logger.error("Error! Unsupported filetype")
                return None

            header, body, *footer = np.split(
                f.readlines(), np.asarray(filetype["split_indices"])
            )

            replacements = [
                (filetype["field_delimiter"], ","),
                (filetype["radix_point"], "."),
            ]

            sio = StringIO()
            csv_writer = csv.writer(sio, delimiter=",")

            for line in body:
                parsed_line = multi_sub(replacements, line.decode(encoding).strip())
                all_cols = [i.strip() for i in parsed_line.split(",")]
                csv_writer.writerow([all_cols[i] for i in filetype["columns"]])

            sio.seek(0)

            bio: BytesIO = BytesIO(sio.read().encode("utf8"))

            sio.close()

            bio.name = f'{filename.as_posix().rsplit(".", 1)[0]}.csv'
            bio.seek(0)

            return bio

    except Exception as e:
        logger.error(e)
        return None
