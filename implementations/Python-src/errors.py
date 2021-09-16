import itertools
from collections import defaultdict
from enum import IntEnum
from typing import Optional, List, Dict, Iterable
from pprint import pprint


class ErrorSeverity(IntEnum):
    FATAL = 999
    ROW_MALFORMATION = 88
    INCORRECT_DATA_TYPE = 66
    WRONG_NUMBER_VALUES = 50
    INFO = 1


class InputError(ValueError):
    type: str = "Input Error"

    def __init__(
        self,
        description: Optional[str] = "Incorrectly-formatted file",
        row: Optional = None,
        details: Optional[str] = "",
        line_num: Optional[int] = None,
        severity=ErrorSeverity.INFO,
        *_args,
        **_kwargs,
    ):
        self.description = description
        self.row = row
        self.file_offset = row.starts_after if hasattr(row, "starts_after") else None
        self.details = details
        self.line_num = line_num
        self.severity = severity

        super().__init__(str(self))

    def __str__(self):
        return f"{self.description}!\n{self.details}\n{self.row}"

    def __repr__(self):
        return f"{type(self).__name__}: {self.description} on line {self.line_num}.  {self.details}"


class RowError(InputError):
    type = "Row"


class ColumnError(InputError):
    type = "Column"


class LengthError(InputError):
    type = "Length"


class TooLongError(LengthError):
    type = "Too Long"


class TooShortError(LengthError):
    type = "Too Short"


class WrongValueType(ColumnError):
    type = "Incorrect Input Type"


"""
Dynamically create error classes
"""
for span, direction in itertools.product(
    [RowError, ColumnError], [TooLongError, TooShortError]
):
    _e = type(
        f"{span.type.replace(' ', '')}{direction.type.replace(' ', '')}Error",
        (span, direction),
        {"type": f"{span.type} {direction.type}"},
    )
    globals()[_e.__name__] = _e


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
