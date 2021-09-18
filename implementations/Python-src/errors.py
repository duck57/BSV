import itertools
from collections import defaultdict
from enum import IntEnum
from typing import Optional, List, Dict, Iterable
from pprint import pprint


class ErrorSeverity(IntEnum):
    FATAL = 999
    BAD_COLUMN_HEADER = 151
    ROW_MALFORMATION = 88
    INCORRECT_DATA_TYPE = 66
    WRONG_NUMBER_VALUES = 50
    INFO = 1


class InputError(ValueError):
    type: str = "Input Error"
    severity = ErrorSeverity.INFO

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
        self.file_offset = row.starts_after if hasattr(row, "starts_after") else None
        self.details = details
        self.line_num = line_num

        super().__init__(str(self))

    def __str__(self):
        return f"{self.description}!\n{self.details}\n{self.row}"

    def __repr__(self):
        return f"{type(self).__name__}: {self.description} on line {self.line_num}.  {self.details}"


class RowError(InputError):
    type = "Row"
    severity = ErrorSeverity.ROW_MALFORMATION


class ColumnError(InputError):
    type = "Column"
    severity = ErrorSeverity.BAD_COLUMN_HEADER


class LengthError(InputError):
    type = "Length"


class TooLongError(LengthError):
    type = "Too Long"


class TooShortError(LengthError):
    type = "Too Short"


class ValueTypeError(ColumnError):
    type = "Incorrect Input Type"
    severity = ErrorSeverity.INCORRECT_DATA_TYPE


class RowTooLongError(TooLongError, RowError):
    type = "Row Too Long"


class RowTooShortError(TooShortError, RowError):
    type = "Row Too Short"


class TooFewValuesError(TooShortError, ColumnError):
    type = "Too Few Values"
    severity = ErrorSeverity.WRONG_NUMBER_VALUES


class TooManyValuesError(TooShortError, ColumnError):
    type = "Too Many Values"
    severity = ErrorSeverity.WRONG_NUMBER_VALUES


class ErrorList(List[InputError]):
    def __init__(self):
        super().__init__()
        self.by_severity: Dict[ErrorSeverity, List[InputError]] = defaultdict(list)
        self.by_line: Dict[int, List[InputError]] = defaultdict(list)
        self.by_type: Dict[str, List[InputError]] = defaultdict(list)

    def append(self, error: InputError) -> None:
        super(ErrorList, self).append(error)
        self.by_type[error.type].append(error)
        self.by_line[error.line_num].append(error)
        self.by_severity[error.severity].append(error)

    def extend(self, list_of_errors: Iterable[InputError]) -> None:
        for e in list_of_errors:
            self.append(e)


def add_error2el(error: InputError, el: Optional[ErrorList] = None):
    if el is None:
        raise error
    el.append(error)
