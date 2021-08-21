#!/usr/bin/python3
import io
from typing import NamedTuple, Generator

FILE: chr = "\x1C"
GROUP: chr = "\x1D"
RECORD: chr = "\x1E"
UNIT: chr = "\x1F"

LINE_BREAKS = [FILE, GROUP]

NAMES_DICT = {
    FILE: "File",
    GROUP: "Group",
    RECORD: "Record",
    UNIT: "Unit",
}


def generate_test_string() -> str:
    """
    :return: a test string with a bunch of non-printable characters
    """
    o = "File separator" + FILE
    o += "Group separator" + GROUP
    o += "Record separator" + RECORD
    o += "Unit separator" + UNIT
    o += "Newline\n"
    o += "Tab (horizontal)\t"
    o += "Tab (vertical)\v"
    o += "Bell\a\x07"
    o += "Carriage return\r"
    o += "End of file"
    return o


class RawLine(NamedTuple):
    """
    The raw line data for a BSV file.

    Ending meanings:
    '\x1D' = end of line/record/row
    '\x1C' = end of table, client should read next table
    '' = end of physical file
    other values may cause undefined behavior
    """

    content: str
    ending: chr


def read_file_into_lines(
    f, buffer: int = io.DEFAULT_BUFFER_SIZE, **_options
) -> Generator[RawLine, None, None]:
    """
    Reads a file and splits into "lines" terminated with either \x1C or \x1D.
    Essentially a very fancy str.splitlines(True)

    This implementation incrementally loads the file from disk into memory so that
    large files may be loaded without excessive RAM requirements.

    :param f: an opened File object in read mode
    :param buffer: read this many characters at a time
    :return: a RawLine
    """
    done: bool = False
    # r is the raw buffer in RAM off the disk
    # o is the output buffer
    o, r = "", ""  # initialize empty buffers
    while not done:
        r: str = f.read(buffer)
        if not r:  # nothing more in the file, process the current s buffer and exit
            done = True
        o += r  # prepend any residual output buffer
        lines = o.splitlines(True)
        o = ""  # the output buffer has been split for analysis; time to clear it
        for line in lines:  # is there something more efficient here?
            end = line[-1]
            o += line[:-1]
            if end in LINE_BREAKS:
                yield RawLine(o, end)
                o = ""  # reset the output string
                continue
            o += end  # it's not one of the line breaks we care about just yet

    """
    yield out the final line
    it's o[-1] to strip out the trailing newline
    r == "" at this point.
    """
    yield RawLine(o[:-1], r)


class Reader:
    pass


def test_reading_file_into_lines(f="../test_BSV_files/test_string.txt"):
    """
    Tests reading a file into lines with a very short reading buffer
    The short buffer exercises how the loader handles lines that span reads
    """
    with open(f) as opened_file:
        for z in read_file_into_lines(
            opened_file, 40  # comment out to debug with the default buffer
        ):
            print(z)


if __name__ == "__main__":
    test_reading_file_into_lines()
