#!/usr/bin/python3

from BSV import *
from pprint import pprint
from collections import defaultdict

"""
Quick testing harness drivers to load from real BSV files
"""


def test_reading(
    file_name: str = "../../test_BSV_files/test_table.bsv", i_d: bool = False
):
    with open(file_name) as f:
        # r = FileReader(f)
        e = defaultdict(list)
        # for row in r.read_file(e, strict=False, into_dicts=i_d):
        for row in read_file_into_rows(f, e, strict=False, into_dicts=i_d):
            pprint(row)
        pprint(e)


if __name__ == "__main__":
    test_reading(i_d=False)
