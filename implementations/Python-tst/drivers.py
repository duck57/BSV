#!/usr/bin/python3

from BSV import *

"""
Quick testing harness drivers to load from real BSV files
"""


with open("./IAT.bsv") as f:
    r = FileReader(f)
    for row in r.read_file():
        print(row)

if __name__ == "__main__":
    pass
