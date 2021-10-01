from __future__ import annotations

import io
from copy import deepcopy
from itertools import zip_longest
from pprint import pprint

from columns import *
from columns import _make_header_str

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

    content: List[str]
    ending: chr
    row_index: int = -1
    starts_after: int = -1


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

    # r: raw buffer in RAM off the disk
    # o: output list (split by RECORD SEPARATOR)
    # s: string buffer to add to output
    o, r, s = [], "", ""  # initialize empty buffers

    # read the file until it's all done
    while not done:
        start_after = f.tell()
        r: str = f.read(buffer)
        if not r:  # nothing more in the file, process the current s buffer and exit
            done = True
        s += r  # prepend any residual output buffer

        # take advantage of the fact that splitlines() includes the file and group separators
        # otherwise, we'd need regular expressions
        lines = s.splitlines(True)  # keep the terminal character for proper processing

        s = ""  # the output buffer has been split for analysis; time to clear it
        for line in lines:  # is there something more efficient here?
            if not s and line == "\n":
                continue  # strip convenience lines
            # print(repr(line))  # debugging
            end = line[-1]
            s += line[:-1]
            if end in LINE_BREAKS:
                yield RawLine(o + [s], end, row_index, start_after)
                o, s = [], ""  # reset the output string
                row_index += 1  # increment for the next row
                continue
            if end == RECORD:
                o.append(s)
                s = ""
                continue
            s += end  # it's not one of the line breaks we care about just yet

    """
    yield out the final line
    it's s[-1] to strip out the trailing newline
    r == "" at this point.
    """
    yield RawLine(o + [s[:-1]], r, row_index, start_after)


class TableRow(abc.ABC):
    """
    Mostly-empty class to serve as a base class for all rows read from a BSV file.

    These methods should be defined when creating the ns dict in new_table_type()
    """

    columns = {}
    allow_short = True
    allow_extras = True

    def __post_init__(self, *fields):
        self.errors = ErrorList()
        *fields, raw_extras, self.row_index = fields
        self.extras = [x.split(UNIT) for x in raw_extras if x is not None]
        for col_num, (value, (c_name, c_type)) in enumerate(
            zip(fields, self.columns.items())
        ):
            if value is None and not self.allow_short and not self.errors:
                self.errors.append(
                    RowTooShortError(
                        "Line too short",
                        fields,
                        f"Only {col_num} column(s).",
                        self.row_index,
                        severity=ErrorSeverity.ROW_MALFORMATION,
                    )
                )
            self.__setattr__(c_name, c_type(value, self.errors, self.row_index))
        if self.extras and not self.allow_extras:
            self.errors.append(
                RowTooLongError(
                    "Too many fields!",
                    fields,
                    f"{len(self.extras)} extra fields were provided for a table with {len(self.columns)}.",
                    self.row_index,
                    severity=ErrorSeverity.ROW_MALFORMATION,
                )
            )

    @classmethod
    def table_str(cls, use_n: bool = True) -> str:
        """The opposite of new_table_from_raw_lines

        :return: a string that will generate the RowType
        """
        # pprint([c.definition_string for c in cls.columns.values()])  # debugging
        return (
            _make_header_str(
                [
                    getattr(cls, "original_name", cls.__name__),  # name
                    ("X" if cls.allow_extras else "")
                    + ("S" if cls.allow_short else ""),  # options
                    getattr(cls, "comment", None),  # comment
                    getattr(cls, "client", None),  # client
                ]
                + getattr(cls, "extra_headers", []),
                RECORD,
            )
            + GROUP
            + ("\n" if use_n else "")
            + RECORD.join([c.definition_string for c in cls.columns.values()])
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.column_names = [c for c in cls.columns.keys()]
        cls.num_columns = len(cls.columns)

    def data_dict(self) -> Dict[str, List]:
        d = {k: getattr(self, k) for k in self.column_names}
        d["extras"] = self.extras
        return d

    def as_bsv_str(
        self, fill: bool = True, trim_columns: bool = True, trim_values: bool = False
    ) -> str:
        o: List[str] = []
        for c_name, c_def in self.columns.items():
            v = getattr(self, c_name, None)
            if v is None:
                if not fill:
                    break  # there's no more columns after the missing one
                v = ""
            o.append(c_def.to_str(v, trim_values))
        if not trim_columns and self.extras:
            o.append(RECORD.join(UNIT.join(s for s in z) for z in self.extras))
        return RECORD.join(o)


def _repr4table_row(self):
    """This should be __repr__ for TableRow,
    but it doesn't work properly when defined within the class directly"""
    return f"{self.__name__}_row({self.data_dict()})"


def new_table_type(
    name: str,
    *columns: ColumnDefinition,
    allow_short: bool = False,
    allow_extras: bool = False,
    error_collector: Optional[ErrorList] = None,
    **ns,
) -> "Type[RowType]":
    """Creates a new Class to contain rows from a table

    :param error_collector: a place to collect errors
    :param name: The name of the table.  Gets changed to "Name_row" in the output
    :param columns: Any number of ColumnDefinitions in order of the table's columns
    :param allow_short: Consider short rows to be valid
    :param allow_extras: Consider rows with extra values to be valid
    :param ns: other attributes to be put into the class' namespace
    :return: A new Class
    """
    # prep work
    columns_as_fields: List[Tuple] = []
    original_name = name
    name = to_class_name(name)

    ns = {  # class attributes
        "__name__": name,
        **ns,
        "original_name": original_name,
        "allow_short": allow_short,
        "allow_extras": allow_extras,
        "columns": {},
        "__repr__": _repr4table_row,
    }
    # turn the columns into Dataclass fields
    for c in columns:
        if isinstance(c, ColumnTemplate):
            c: ColumnDefinition = c(c.name)
            add_error2el(
                ColumnError("Attempted use of template column"), error_collector
            )
        n = "col_" + to_class_name(c.name)
        columns_as_fields.append((n, dataclasses.InitVar))
        ns["columns"][n] = c
    columns_as_fields.append(("extras", dataclasses.InitVar[list]))
    columns_as_fields.append(
        ("row_index", dataclasses.InitVar[int], dataclasses.field(default=-1))
    )

    # create the new class
    return dataclasses.make_dataclass(
        f"{name}_row", columns_as_fields, bases=(TableRow,), namespace=ns,
    )


RowType = TypeVar("RowType", bound=TableRow)


def new_table_from_raw_lines(
    table_definition: RawLine, column_heads: RawLine, el: Optional[ErrorList] = None
) -> "Type[RowType]":
    """

    :param el: optional collector for errors
    :param table_definition: RawLine with the basic table information
    :param column_heads: RawLine containing the column setup information
    :return: a new subclass of TableRow for this new table
    """
    # There clearly has to be a better way to split a string into
    # a list with meaningful order when not all components are present
    tad = {
        "original_name": "",
        "options": "",  # 1
        "comment": None,  # 2
        "client": None,  # 3
        "extras": [],  # 4
    }
    for attr, value in zip(tad.keys(), table_definition.content):
        if attr == "extras":
            break  # handle these a line later
        tad[attr] = value
    tad["extra_headers"] = table_definition.content[4:]
    tad["allow_short"] = "S" in tad["options"].upper()
    tad["allow_extras"] = "X" in tad["options"].upper()

    columns = []
    for raw_c in column_heads.content:
        c_attrs = {
            "name": "",  # 0
            "data_type": "",  # 1
            "range_str": "",  # 2
            "comment": None,  # 3
            "client": None,  # 4
            "extra_headers": [],  # 5
        }
        col_attrs: List[str] = raw_c.split(UNIT)
        for attr, value in zip(c_attrs.keys(), col_attrs):
            if attr == "extras":
                break
            c_attrs[attr] = value
        c_attrs["extras"] = col_attrs[5:]
        c_attrs["el"] = el

        columns.append(ColumnDefinition(**c_attrs))

    return new_table_type(tad["original_name"], *columns, **tad)


def raw_line2dataclass(row_type: Type[RowType], data: RawLine, row_index: int = -1):
    num_fields = row_type.num_columns
    d2 = deepcopy(data.content)
    x: List[str] = d2[num_fields:]
    d2 = d2[:num_fields] + ([None] * (num_fields - len(d2)))
    o = row_type(*d2, x, row_index)
    return o


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
    o = {None: []}
    for column_type, value in zip_longest(
        row_type.columns.values(), data.content, fillvalue=None
    ):
        if column_type is None:  # extra fields
            o[column_type].append(value.split(UNIT))
            continue

        # the usual route
        column_name = column_type.name
        o[column_name] = value if value is None else value.split(TAB if False else UNIT)
    o[151374] = meta  # 151 = IVI (looks like m), 374 = ETA
    return o


def read_file_into_rows(
    f,
    errors: ErrorList = None,
    *,
    strictness: int = 23,
    into_dicts: bool = False,
    direct_iterator: Optional[Iterable[RawLine]] = None,
    acceptable_errors: List[Type[InputError]] = None,
    buffer: int = io.DEFAULT_BUFFER_SIZE,
):
    """Where all the magic occurs.  Reads the file into dataclasses or dicts.

    :param acceptable_errors: a list of types of acceptable error rather than a numeric value guess
    :param buffer: controls how much of the file to read at a times.
        No need to adjust this under normal circumstances.
    :param direct_iterator: alternate to f
    :param f: the BSV file-like object to read
    :param errors: writable
    :param strictness: stop processing and raise an exception at any error?
    :param into_dicts: Read the file into dicts?  If not, each table generates a
        subtype of dataclass in which the results are stored.
    :return: yields either a dict or a dataclass for each row of the input file
    """

    # setup
    last_ending: chr = ""
    rows: Iterator[RawLine] = iter(
        direct_iterator
    ) if direct_iterator else _split_file_into_rows(f, buffer)
    if errors is None:
        errors = ErrorList()
    if acceptable_errors is None:
        acceptable_errors = []
    conversion_function = raw_line2dict if into_dicts else raw_line2dataclass
    current_table: Type[RowType] = type(
        RowType
    )  # this assignment is to make the linter shut up later
    tables: Dict[str, Type[RowType]] = {}
    if not strictness:
        strictness = 998

    # the actual processing loop
    for line in rows:
        if not line.ending and not line.content:
            continue  # file is all done

        # read table header
        # also covers the empty string at the start of reading
        if last_ending in FILE:
            # print("next table!")
            table_name = to_class_name(line.content[0])
            if table_name not in tables.keys():  # define a new table
                # next(rows) is called here and not in new_table() so
                # that last_ending may be properly set
                column_headers: RawLine = next(rows)

                tables[table_name] = new_table_from_raw_lines(line, column_headers)
                last_ending = column_headers.ending
            else:
                last_ending = line.ending
            current_table = tables[table_name]
            continue

        # read the row into an object (or collect an error)
        o = conversion_function(current_table, line, line.row_index)
        if not into_dicts:
            # handle errors
            errors.extend(o.errors)
            for e in o.errors:
                if e.severity > strictness and not any(
                    isinstance(e, a) for a in acceptable_errors
                ):
                    raise e
        yield o  # congratulations, we've parsed a new row!
        # save the ending for reading the next line
        last_ending = line.ending

    return errors


def bsv_dict2str(d: Dict[Any, List]) -> str:
    main_content = RECORD.join(
        (
            UNIT.join(i for i in c)
            for col, c in d.items()
            if isinstance(col, str) and c is not None
        )
    )
    if not d[None]:
        return main_content
    return RECORD.join([main_content] + [UNIT.join(i for i in c) for c in d[None]])


def write_from_dicts(f, rows: Iterable[Dict[Any, List]], *, use_n: bool = True):
    first_row: bool = True
    for row in rows:
        if first_row:
            first_row = False
        else:
            f.write(GROUP + "\n" if use_n else "")
        f.write(bsv_dict2str(row))


def write_to_file(
    f,
    rows: Iterable[RowType],
    *,
    use_n: bool = True,
    sort: bool = False,
    fill: bool = True,
    trim_columns: bool = False,
    trim_values: bool = False,
):
    def w(content):
        f.write(content + "\n" if use_n else "")

    def wx(content):
        f.write(content)

    if sort:
        rows = sorted(rows, key=_type_sort_key)
    table_list = set()
    current_table = None

    for row in rows:
        this_table = type(row)
        if this_table != current_table:
            if current_table is not None:
                w(FILE)
            current_table = this_table
            if this_table not in table_list:
                wx(row.table_str(use_n))
                table_list.add(this_table)
            else:
                wx(this_table.original_name)
        w(GROUP)
        wx(row.as_bsv_str(fill, trim_columns, trim_values))


def _type_sort_key(t):
    return hash(type(t))
