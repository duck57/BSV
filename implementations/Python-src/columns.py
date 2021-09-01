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


def to_class_name(s: str) -> str:
    return re.sub(r"\W", "", s.title())


def is_valid_field(self: dataclasses.Field):
    return True


class ColumnDefinition(NamedTuple):
    name: str
    type: Type = str
    max: int = sys.maxsize
    sep: chr = UNIT
    min: int = -1

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

    def make_column_type(self, table: str):
        return type(
            "cell_" + to_class_name(table) + "_" + to_class_name(self.name),
            (BaseCell,),
            {"max_items": self.max, "min_items": self.min, "type": self.type,},
        )


class BaseCell:
    values: "List[ValueType]"
    separator: chr = UNIT

    def __init__(
        self,
        column_name: str,
        max_length: int = sys.maxsize,
        min_length: int = -1,
        tab_separator: bool = False,
        **_options,
    ):
        if tab_separator:
            self.separator = "\t"
        self.max_length = max_length
        self.min_length = min_length
        pass

    def __str__(self):
        return self.separator.join([str(v) for v in self.values])

    @property
    def is_valid(self) -> bool:
        return self.min_length < len(self.values) < self.max_length


@dataclass
class BaseValue(abc.ABC):
    def __init__(self, value, **_others):
        self.value = value

    def __str__(self):
        """
        :return: A string representing the value compatible with BSV files
        """
        return str(self.value)

    @staticmethod
    @abc.abstractmethod
    def from_str(value: str) -> "ValueType":
        ...


class StringValue(BaseValue):
    @staticmethod
    def from_str(value: str) -> "StringValue":
        return StringValue(value)


class IntegerValue(BaseValue):
    @staticmethod
    def from_str(value: str):
        return IntegerValue(int(value))


class RelativeDatetimeValue(BaseValue):
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


class CurrencyValue(BaseValue):
    def __init__(self, amount: str, code: str, precision: str):
        super().__init__(Decimal(amount))

    def from_str(self, value: str, parent):
        ...


ColumnType = TypeVar("ColumnType", bound=BaseCell)
ValueType = TypeVar("ValueType", bound=BaseValue)
