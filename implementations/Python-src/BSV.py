import dataclasses
from pprint import pprint
from typing import Type, Tuple
from collections import defaultdict
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


class TableRow(abc.ABC):
    def is_valid(self):
        # TODO: rewrite this to work with dataclasses
        pass

    @classmethod
    def new_from_BSV(cls, *fields):
        return cls.__init__(*fields)

    @property
    def num_fields(self) -> int:
        return len(self.__dataclass_fields__) - (1 if self.allow_extras else 0)

    def __post_init__(self, *fields):
        for col_num, (value, (c_name, c_type)) in enumerate(
            zip(fields, self.columns.items())
        ):
            if value is not None:
                value = value.split(c_type.sep)
            elif not self.allow_short:
                raise RowError(
                    "Line too short",
                    RECORD.join([str(f) for f in fields]),
                    f"Only {col_num} column(s).",
                )
            self.__setattr__(
                c_name, None if value is None else [c_type.type(v) for v in value]
            )
        if self.allow_extras:
            self.extras = (
                [] if fields[-1] is None else [x.split(UNIT) for x in fields[-1]]
            )
        else:
            self.extras = None

        # pprint(self.__dict__)
        pass

    @property
    def values(self):
        return [getattr(self, x) for x in self.__dataclass_fields__]


def _column_repr__(self):
    return f"{self.__dir__()[0]}({self.__dict__})"


def new_table_type(
    name: str,
    *columns: ColumnDefinition,
    allow_short: bool = False,
    allow_extras: bool = False,
) -> "RowType":
    rows: List[Tuple] = []
    ns: Dict[str] = {
        "allow_short": allow_short,
        "allow_extras": allow_extras,
        "columns": {},
        "__repr__": _column_repr__,
    }
    for c in columns:
        n = "col_" + to_class_name(c.name)
        rows.append((n, dataclasses.InitVar))
        ns["columns"][n] = c
    if allow_extras:
        rows.append(("extras", dataclasses.InitVar))
        ns["extras"] = True
    return dataclasses.make_dataclass(
        f"{to_class_name(name)}_row", rows, bases=(TableRow,), namespace=ns,
    )


RowType = TypeVar("RowType", bound=TableRow)


def new_table_from_BSV_header(
    title: str, column_row: List[str], *options, **meta
) -> "RowType":
    a_s: bool = True in ["S" in o.upper() for o in options]
    a_x: bool = True in ["X" in o.upper() for o in options]

    columns = []
    for c in column_row:
        c_name, *c_attrs = c.split(UNIT)
        if not c_attrs:
            ...  # assume unlimited string
        columns.append(ColumnDefinition(c_name))

    return new_table_type(title, *columns, allow_short=a_s, allow_extras=a_x)


def raw_line2dataclass(row_type: Type[RowType], data: RawLine, row_index: int = -1):
    num_fields = len(row_type.__dataclass_fields__) - (
        1 if row_type.allow_extras else 0
    )
    d2 = data.parts
    if len(d2) > num_fields:
        if row_type.allow_extras:
            x: List = d2[num_fields:]
            d2 = d2[0:num_fields]
            o = row_type(*d2, x)
            o.__setattr__("row_index", row_index)
        raise RowError(
            "Too many fields!",
            data,
            f"{len(d2)} fields were provided for a table with {num_fields} in row {row_index} of the input file.",
        )

    if len(d2) < num_fields:
        # +1 for extra fields
        d2 += [None] * ((1 if row_type.allow_extras else 0) + num_fields - len(d2))
    o = row_type(*d2)
    o.__setattr__("row_index", row_index)
    return o


class InputError(ValueError):
    def __init__(
        self,
        description: Optional[str] = "Incorrectly-formatted file",
        row: Optional = None,
        details: Optional[str] = "",
        line_num: Optional[int] = None,
        *_args,
        **_kwargs,
    ):
        self.description = description
        self.row = row
        self.file_offset = row.starts_after if isinstance(row, RawLine) else None
        self.details = details
        self.line_num = line_num

        super().__init__(str(self))

    def __str__(self):
        return f"{self.description}!\n{self.details}\n{self.row}"

    def __repr__(self):
        return f"{type(self).__name__}: {self.description} on line {self.line_num}.  {self.details}"


class RowError(InputError):
    pass


class ColumnError(InputError):
    pass


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

    def read_file(self, strict: bool = True, errors: defaultdict = None):
        if errors is None:
            errors = defaultdict(list)
        while True:
            line = self.next_row()
            if not line.ending and not line.content:
                # raise StopIteration  # file is all done
                return errors
            if (
                self.last_ending in FILE
            ):  # also covers the empty string at the start of reading
                print("new table!")
                self._read_table_header()  # no need to pass a param
                pprint(self.current_table.__dict__)
                continue
            try:
                yield raw_line2dataclass(self.current_table, line, self.row_index)
            except InputError as e:
                errors[self.row_index].append((e, line,))
                if strict:
                    raise e
                else:
                    continue

    pass
