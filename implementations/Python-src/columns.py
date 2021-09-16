from __future__ import annotations

import abc
import dataclasses
import sys
from dataclasses import make_dataclass, dataclass, field
from datetime import datetime
from decimal import Decimal
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
FILE: chr = "\x1C"
GROUP: chr = "\x1D"
RECORD: chr = "\x1E"
UNIT: chr = "\x1F"
LINE: chr = "\n"
TAB: chr = "\t"


def to_class_name(s: str) -> str:
    return re.sub(r"\W", "", s.title())


def is_valid_field(self: dataclasses.Field):
    return True


class ColumnDefinition:
    def __init__(
        self,
        name: str,
        data_type: Type = str,
        max_vals: int = sys.maxsize,
        sep: chr = UNIT,
        min_vals: int = -1,
        range_str: str = "",
        template: bool = False,
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
        self.is_template = template

        for a, v in misc_attrs.items():
            self.__setattr__(a, v)

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

    def __call__(self, *args, **kwargs):
        if not self.is_template:
            return
        return ColumnDefinition(
            args[0], self.type, self.min, self.sep, self.max, template=False
        )


class RelativeDatetimeValue:
    def __init__(self, unit: chr, distance: int):
        self.distance = distance
        self.unit = unit.upper()
        super().__init__(str(self))

    @property
    def value(self):
        return f"{self.unit}{'' if self.distance < 0 else '+'}{self.distance}"

    @staticmethod
    def from_str(value: str):
        pass

    def __call__(self, dt: datetime | int):
        if isinstance(dt, int):
            return dt + self.distance


class CurrencyValue:
    def __init__(self, amount: str, code: str, precision: str):
        super().__init__(Decimal(amount))

    def from_str(self, value: str, parent):
        ...


ColumnType = TypeVar("ColumnType", bound=ColumnDefinition)
