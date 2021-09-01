#!/usr/bin/python3

from BSV import *
from pprint import pprint
from collections import defaultdict

"""
Quick testing harness drivers to load from real BSV files
"""


with open("../../test_BSV_files/test_table.bsv") as f:
    r = FileReader(f)
    e = defaultdict(list)
    for row in r.read_file(False, e):
        pprint(row)
    pprint(e)

if __name__ == "__main__":
    pass
