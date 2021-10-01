#!/usr/bin/python3
from __future__ import annotations

from BSV import *
from pprint import pprint
import sys
from _io import TextIOWrapper

"""
Quick testing harness drivers to load from real BSV files
"""


def test_reading(
    file_name: str = "../../test_BSV_files/test_table.bsv",
    i_d: bool = False,
    out_path: str | TextIO = "../../test_BSV_files/test_out.tmp",
):
    rows = []
    with open(file_name) as f:
        e = ErrorList()
        for row in read_file_into_rows(
            f,
            e,
            into_dicts=i_d,
            acceptable_errors=[
                InputError,
                # TooLongError,
                # TooShortError,
                # LengthError,
                # ValueTypeError,
            ],
        ):
            rows.append(row)
            pprint(row)
        pprint(e)

    with out_path if isinstance(out_path, TextIOWrapper) else open(out_path, "w") as g:
        if i_d:
            write_from_dicts(g, rows)
        else:
            write_to_file(g, rows)


if __name__ == "__main__":
    test_reading(i_d=False, out_path=sys.stdout)
