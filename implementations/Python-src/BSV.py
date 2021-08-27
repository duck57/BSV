import dataclasses
from typing import Type, Tuple

from columns import *
import io

"""
"""

LINE_BREAKS: str = FILE + GROUP

NAMES_DICT: "Dict[chr, str]" = {
    FILE: "File",
    GROUP: "Group",
    RECORD: "Record",
    UNIT: "Unit",
}


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
    starts_after: int = -1

    @property
    def parts(self) -> List[str]:
        return self.content.split(RECORD)


def read_next_line_from_file(
    f, buffer: int = io.DEFAULT_BUFFER_SIZE, **_options
) -> RawLine:  # -> Generator[RawLine, None, None]:
    """
    Reads a file and splits into "lines" terminated with either \x1C or \x1D.
    Essentially a very fancy str.splitlines(True)

    This implementation incrementally loads the file from disk into memory so that
    large files may be loaded without excessive RAM requirements.

    :param f: an opened File object in read mode
    :param buffer: read this many characters at a time
    :return: a RawLine
    """
    start_after: int = 0  # for debugging purposes
    done: bool = False
    # r is the raw buffer in RAM off the disk
    # o is the output buffer
    o, r = "", ""  # initialize empty buffers
    while not done:
        start_after = f.tell()
        r: str = f.read(buffer)
        if not r:  # nothing more in the file, process the current s buffer and exit
            done = True
        o += r  # prepend any residual output buffer

        # take advantage of the fact that splitlines() includes the file and group separators
        # otherwise, we'd need regular expressions
        lines = o.splitlines(True)  # keep the terminal character for proper processing

        o = ""  # the output buffer has been split for analysis; time to clear it
        for line in lines:  # is there something more efficient here?
            if not o and line == "\n":
                continue  # strip convenience lines
            # print(repr(line))  # debugging
            end = line[-1]
            o += line[:-1]
            if end in LINE_BREAKS:
                yield RawLine(o, end, start_after)
                o = ""  # reset the output string
                continue
            o += end  # it's not one of the line breaks we care about just yet

    """
    yield out the final line
    it's o[-1] to strip out the trailing newline
    r == "" at this point.
    """
    yield RawLine(o[:-1], r, start_after)


class TableRow:
    def is_valid(self):
        # TODO: rewrite this to work with dataclasses
        pass

    @classmethod
    def new_from_BSV(cls, *fields):
        return cls.__init__(*fields)


def new_table_type(
    name: str,
    *columns: ColumnDefinition,
    allow_short: bool = False,
    allow_extras: bool = False,
) -> "RowType":
    return dataclasses.make_dataclass(
        f"{to_class_name(name)}_row",
        [(f"col_{to_class_name(c.name)}", list, c.dataclass_field) for c in columns]
        + [("extras", list, field(default_factory=list))]
        if allow_extras
        else [],
        bases=(TableRow,),
        namespace={"allow_short": allow_short, "allow_extras": allow_extras},
    )


RowType = TypeVar("RowType", bound=TableRow)


def new_table_from_BSV_header(
    title: str, column_row: List[str], *options, **meta
) -> "RowType":
    a_s: bool = True in ["S" in o.upper() for o in options]
    a_x: bool = True in ["X" in o.upper() for o in options]

    columns = []
    for c in column_row:
        c_name = c.split(UNIT)[0]
        columns.append(ColumnDefinition(c_name))

    return new_table_type(title, *columns, allow_short=a_s, allow_extras=a_x)


def raw_line2dataclass(row_type: Type[TableRow], *data):
    num_fields = len(row_type.__dataclass_fields__) - 1
    d2 = list(data)
    if len(data) > num_fields and row_type.allow_extras:  # -1 for the extras field
        x: List = d2[num_fields:]
        d2 = d2[0:num_fields]
        return row_type(*d2, x)
    return row_type(*data)


class FileReader:
    """"""

    def __init__(self, f):
        self.tables = {}
        self.rows = read_next_line_from_file(f)
        self.current_table = None
        self.current_row = None
        self.last_ending = ""
        self.row_index: int = 0

    def next_row(self):
        self.row_index += 1
        self.last_ending = self.current_row.ending if self.current_row else ""
        self.current_row = next(self.rows)
        return self.current_row

    def _read_table_header(self):
        """
        Reads a table header row
        :return: An existing TableReader object if one exists for the same name
        If no match is found, creates and returns the new reader
        """
        assert (
            self.current_row.ending == GROUP
        ), f"Incorrect ending in table header\n\t{repr(self.current_row)}"
        meta: Dict[str] = {
            "row_number": self.row_index,
            "bytes_after": self.current_row.starts_after,
        }
        parts = self.current_row.parts
        title = parts[0].lower().strip()
        options = parts[1:] if len(parts) > 1 else []
        t = self.tables.get(title)
        if not t:
            t = new_table_from_BSV_header(
                title, self.next_row().parts, *options, **meta
            )
            self.tables[title] = t
        self.current_table = t
        return t

    def read_file(self):
        while True:
            line = self.next_row()
            if not line.ending and not line.content:
                raise StopIteration  # file is all done
            if (
                self.last_ending in FILE
            ):  # also covers the empty string at the start of reading
                print("new table!")
                self._read_table_header()  # no need to pass a param
                print(self.current_table.__dict__)
                continue
            yield raw_line2dataclass(self.current_table, *line.parts)

    pass
