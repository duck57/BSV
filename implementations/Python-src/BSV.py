import dataclasses
from itertools import zip_longest
from pprint import pprint
from typing import Type, Tuple, Iterable, Iterator
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
    """The raw line data for a BSV file.

    Ending meanings:

    - GROUP SEPARATOR = end of line/record/row

    - FILE SEPARATOR = end of table, client should read next table

    - '' = end of physical file

    other values may cause undefined behavior
    """

    content: str
    ending: chr
    row_index: int = -1
    starts_after: int = -1

    @property
    def parts(self) -> List[str]:
        return self.content.split(RECORD)

    def extract_table_name(self) -> str:
        return to_class_name(self.content.split(RECORD)[0])


def _split_file_into_rows(
    f, buffer: int = io.DEFAULT_BUFFER_SIZE, **_options
) -> "Generator[RawLine]":
    """
    Reads a file and splits into "lines" terminated with either \x1C or \x1D.
    Essentially a very fancy str.splitlines(True)

    This implementation incrementally loads the file from disk into memory so that
    large files may be loaded without excessive RAM requirements.

    :param f: an opened File object in read mode
    :param buffer: read this many characters at a time
    :return: a RawLine
    """
    # setup
    start_after: int = 0  # for debugging purposes
    done: bool = False
    row_index = 1

    # r is the raw buffer in RAM off the disk
    # o is the output buffer
    o, r = "", ""  # initialize empty buffers

    # read the file until it's all done
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
                yield RawLine(o, end, row_index, start_after)
                o = ""  # reset the output string
                row_index += 1  # increment for the next row
                continue
            o += end  # it's not one of the line breaks we care about just yet

    """
    yield out the final line
    it's o[-1] to strip out the trailing newline
    r == "" at this point.
    """
    yield RawLine(o[:-1], r, row_index, start_after)


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
    return f"{self.__name__}_row({self.__dict__})"


def _column_names(cls) -> List[str]:
    return [c.name for c in cls.columns.values()]


def new_table_type(
    name: str,
    *columns: ColumnDefinition,
    allow_short: bool = False,
    allow_extras: bool = False,
    **ns,
) -> "Type[RowType]":
    """Creates a new Class to contain rows from a table

    :param name: The name of the table.  Gets changed to "Name_row" in the output
    :param columns: Any number of ColumnDefinitions in order of the table's columns
    :param allow_short: Consider short rows to be valid
    :param allow_extras: Consider rows with extra values to be valid
    :param ns: other attributes to be put into the class' namespace
    :return: A new Class
    """
    # prep work
    columns_as_fields: List[Tuple] = []
    ns = {
        "__name__": name,
        **ns,
        "allow_short": allow_short,
        "allow_extras": allow_extras,
        "columns": {},
        "__repr__": _column_repr__,
        "column_names": _column_names,
    }
    # turn the columns into Dataclass fields
    for c in columns:
        n = "col_" + to_class_name(c.name)
        columns_as_fields.append((n, dataclasses.InitVar))
        ns["columns"][n] = c
    if allow_extras:
        columns_as_fields.append(("extras", dataclasses.InitVar))
        ns["extras"] = True
    # create the new class
    return dataclasses.make_dataclass(
        f"{name}_row", columns_as_fields, bases=(TableRow,), namespace=ns,
    )


RowType = TypeVar("RowType", bound=TableRow)


def new_table_from_raw_lines(
    table_definition: RawLine, column_heads: RawLine
) -> "Type[RowType]":
    """

    :param table_definition:
    :param column_heads:
    :return:
    """
    # There clearly has to be a better way to split a string into
    # a list with meaningful order when not all components are present
    tad = {
        "original_name": "",  # 0
        "options": "",  # 1
        "comment": None,  # 2
        "client": None,  # 3
        "extras": [],  # 4
    }
    table_attrs: List[str] = table_definition.content.split(RECORD)
    for attr, value in zip(tad.keys(), table_attrs):
        if attr == "extras":
            break  # handle these a line later
        tad[attr] = value
    tad["extras"] = table_attrs[4:]
    tad["allow_short"] = "S" in tad["options"].upper()
    tad["allow_extras"] = "X" in tad["options"].upper()

    columns = []
    for raw_c in column_heads.content.split(RECORD):
        c_attrs = {
            "name": "",  # 0
            "data_hint": "",  # 1
            "range": "",  # 2
            "comment": "",  # 3
            "client": None,  # 4
            "extras": [],  # 5
        }
        col_attrs: List[str] = raw_c.split(UNIT)
        for attr, value in zip(c_attrs.keys(), col_attrs):
            if attr == "extras":
                break
            c_attrs[attr] = value
        c_attrs["extras"] = col_attrs[5:]

        columns.append(ColumnDefinition(**c_attrs))

    return new_table_type(table_definition.extract_table_name(), *columns, **tad)


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


def raw_line2dict(row_type: Type[RowType], data: RawLine, row_index: int = -1):
    """
    Closer to csv.DictReader than raw_line2dataclass in spite sharing a function
    signature with the later.

    This function does zero validation of the input rows and puts everything
    (both column names and values) into strings or lists thereof.

    :param row_type: A class of type of row
    :param data: the RawLine to process
    :param row_index: for creating the metadata
    :return: A dict where each column name is a (string) key.
    Any non-string keys in this dict are either metadata o[151374]
    or are for spillover values when there are more values than columns o[None].

    - o[None] == None indicates that there are no spillover values in the row.

    - Likewise, o[column] == None indicates that the row ran out of values before reaching this column.

    - A column with an empty value has a value of '' in the dict.

    Each cell is turned into a list of strings that are split according to the
    column definition.
    """
    meta = {"table_name": row_type.original_name, "row_index": row_index}
    o = {None: None}
    for column_type, value in zip_longest(
        row_type.columns.values(), data.content.split(RECORD), fillvalue=None
    ):
        if column_type is None:  # extra fields
            if o[None] is None:
                o[None] = []
            o[column_type].append(value.split(UNIT))
            continue

        # the usual route
        column_name = column_type.name
        o[column_name] = value if value is None else value.split(TAB if False else UNIT)
    o[151374] = meta  # 151 = IVI (looks like m), 374 = ETA
    return o


class ErrorList(list):
    @property
    def by_line(self):
        ...

    @property
    def by_severity(self):
        ...

    @property
    def by_type(self):
        ...


def read_file_into_rows(
    f,
    errors: defaultdict = None,
    *,
    strict: bool = True,
    into_dicts: bool = False,
    direct_iterator: Optional[Iterable[RawLine]] = None,
):
    """Where all the magic occurs.  Reads the file into dataclasses or dicts.

    :param direct_iterator: alternate to f
    :param f: the BSV file-like object to read
    :param errors: writable
    :param strict: stop processing and raise an exception at any error?
    :param into_dicts: Read the file into dicts?  If not, each table generates a
    subtype of dataclass in which the results are stored.
    :return: yields either a dict or a dataclass for each row of the input file
    """

    # setup
    last_ending: chr = ""
    rows: Iterator[RawLine] = iter(
        direct_iterator
    ) if direct_iterator else _split_file_into_rows(f)
    if errors is None:
        errors = defaultdict(list)
    conversion_function = raw_line2dict if into_dicts else raw_line2dataclass
    current_table: Type[RowType] = type(
        RowType
    )  # this assignment is to make the linter shut up later
    tables: Dict[str, Type[RowType]] = {}

    # the actual processing loop
    for line in rows:
        if not line.ending and not line.content:
            continue  # file is all done

        # read table header
        # also covers the empty string at the start of reading
        if last_ending in FILE:
            print("next table!")
            table_name = line.extract_table_name()
            if table_name not in tables.keys():  # define a new table
                """
                next(rows) is called here and not in new_table() so
                that last_ending may be properly set
                """
                column_headers: RawLine = next(rows)
                tables[table_name] = new_table_from_raw_lines(line, column_headers)
                last_ending = column_headers.ending
            else:
                last_ending = line.ending
            current_table = tables[table_name]
            pprint(current_table)
            continue

        # read the row into an object (or collect an error)
        try:
            yield conversion_function(current_table, line, line.row_index)
        except InputError as e:
            errors[line.row_index].append((e, line,))
            if strict:
                raise e
            else:
                pass

        # save the ending for reading the next line
        last_ending = line.ending

    return errors
