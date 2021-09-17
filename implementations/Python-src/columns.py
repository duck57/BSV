from __future__ import annotations

import abc
import dataclasses
import sys
from dataclasses import make_dataclass, dataclass, field
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import (
    NamedTuple,
    Generator,
    Dict,
    Callable,
    TypeVar,
    Any,
    List,
    Optional,
    Type,
)
import re
from errors import *

"""
Separated into a second file for shorter file lengths.
May be re-integrated in a future commit.
"""

# FILE and GROUP are defined here
# even though they are only used in BSV.py
# so that these definitions stay together in one place
FILE: chr = "\x1C"  # table
GROUP: chr = "\x1D"  # line
RECORD: chr = "\x1E"  # field/column
UNIT: chr = "\x1F"  # value
LINE: chr = "\n"  # newline?
TAB: chr = "\t"  # alternate value separator


def to_class_name(s: str) -> str:
    return re.sub(r"\W", "", s.title())


def _make_header_str(headers: List[Optional[str]], sep: chr):
    return sep.join(h for h in headers if h is not None)


class ColumnDefinition:
    is_template: bool = True

    def __init__(
        self,
        name: str,
        data_type: Type = str,
        max_vals: int = sys.maxsize,
        sep: chr = UNIT,
        min_vals: int = -1,
        range_str: str = "",
        **misc_attrs,
    ):
        self.name = name
        self.type = data_type
        if range_str:
            vals = range_str.split("-")
            try:
                min_vals = int(vals[0])
            except ValueError:
                pass
            try:
                max_vals = int(vals[-1])
            except ValueError:
                pass
            if len(vals) > 2:
                sep = vals[1]
        self.max = max_vals
        self.min = min_vals
        self.sep = sep

        for a, v in misc_attrs.items():
            self.__setattr__(a, v)

    @property
    def range_string(self) -> str:
        min_st = "" if self.min < 1 else str(self.min)
        max_st = "" if self.max == sys.maxsize else str(self.max)
        sep = {UNIT: "", TAB: "t", " ": "s"}.get(self.sep, "ERROR")
        result = f"{min_st}-{sep}-{max_st}"
        return "" if result == "--" else result

    @property
    def definition_string(self) -> str:
        """This is the header which will recreate the column definition"""
        return _make_header_str(
            [
                self.name,
                self.type.dh_chr,
                self.range_string,
                getattr(self, "comment", None),
                getattr(self, "client", None),
            ]
            + getattr(self, "extra_headers", []),
            UNIT,
        )

    @property
    def dataclass_field(self) -> dataclasses.Field:
        return field(
            default=None,
            metadata={
                "original_name": self.name,
                "mod_name": to_class_name(self.name),
                "max_items": self.max,
                "min_items": self.min,
                "type": self.type,
            },
        )

    def __call__(
        self, val_str: str, el: ErrorList = None, line_num: int = -1
    ) -> Optional[List]:
        if val_str is None:
            return val_str
        vals: List = []
        for v in val_str.split(self.sep):
            try:
                v = self.type(v)
            except ValueError as e:
                add_error2el(
                    ValueTypeError(
                        "Conversion failure", details=str(e), line_num=line_num
                    ),
                    el,
                )
            vals.append(v)
        if len(vals) < self.min:
            add_error2el(TooFewValuesError(), el)
        if len(vals) > self.max:
            add_error2el(TooManyValuesError(), el)

        return vals

    def to_str(self, vals: List) -> str:
        return self.sep.join(str(v) for v in vals)


class ColumnTemplate(ColumnDefinition):
    is_template = True

    def __call__(self, column_name: str, **kwargs) -> ColumnDefinition:
        """
        :param column_name: the name of the new ColumnDefinition to return
        :param kwargs: junk to maintain class hierarchy compatibility
        :return: A ColumnDefinition based on self
        """
        return ColumnDefinition(column_name, self.type, self.min, self.sep, self.max)


DataHintDict: "Dict[chr, Type[BaseValue]]" = {}


class BaseValue(abc.ABC):
    dh_chr: chr

    @staticmethod
    @abc.abstractmethod
    def from_str(value: str):
        ...

    @abc.abstractmethod
    def __str__(self):
        ...

    def __init_subclass__(cls, **kwargs):
        if "dh_chr" in kwargs.keys():
            DataHintDict[kwargs["dh_chr"]] = cls
        else:
            raise AttributeError(f"'dh_chr' is not among the class' keys")


class RelativeDatetimeValue(BaseValue, dh_chr="E"):
    def __init__(self, unit: chr, distance: int):
        self.distance = distance
        self.unit = unit.upper()
        super().__init__(str(self))

    @property
    def value(self):
        return f"{self.unit}{'' if self.distance < 0 else '+'}{self.distance}"

    @staticmethod
    def from_str(value: str):
        ...

    def __call__(self, dt: datetime | int):
        if isinstance(dt, int):
            return dt + self.distance

    def __str__(self):
        ...


class CurrencyValue(BaseValue, Decimal, dh_chr="C"):
    def __init__(self, amount: str, code: str, precision: str):
        super(Decimal, self).__init__(Decimal(amount))

    def from_str(self, value: str):
        ...

    def __str__(self):
        ...


ColumnType = TypeVar("ColumnType", bound=ColumnDefinition)
ValueType = TypeVar("ValueType", bound=BaseValue)
