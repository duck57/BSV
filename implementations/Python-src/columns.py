from __future__ import annotations

import abc
import dataclasses
import decimal
import itertools
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
    Union,
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
        data_type: Union[Type, str] = str,
        max_vals: int = sys.maxsize,
        sep: chr = UNIT,
        min_vals: int = -1,
        range_str: str = "",
        currency_code: str = "XXX",
        decimal_places: int = 3,
        el: Optional[ErrorList] = None,
        **misc_attrs,
    ):
        self.name = name
        if isinstance(data_type, str):
            data_type += "S"
            data_type = data_type.upper()
            data_type, data_hint = data_type[0], data_type[1:-1].split()
            if data_type == "C":
                currency_code, decimal_places = data_hint[1], int(data_hint[0])
        self.type = DataHintDict.get(data_type, StringWrapper)
        self.currency_code, self.currency_precision = currency_code, decimal_places

        # override provided min-sep-max if range_str is provided
        if range_str:
            range_str: List[str] = range_str.split("-")
            sep = ""
        else:
            range_str = []  # don't confuse later if statements
        if len(range_str) == 1:
            if range_str[0].isnumeric():
                max_vals = int(range_str[0])
            else:
                sep = range_str[0]
        if len(range_str) > 1:
            try:
                min_vals, max_vals = (
                    int("0" + range_str[0]),
                    int(range_str[-1]) if range_str[-1] else max_vals,
                )
            except ValueError as e:
                # yes, both min and max will fail to set if one of them
                # is not an integer in the range string
                add_error2el(
                    ColumnError(
                        "Non-integer range!",
                        range_str,
                        f"min: '{range_str[0]}' / max: '{range_str[-1]}'",
                    ),
                    el,
                )
        if len(range_str) > 2:
            sep = range_str[1]
        if range_str:
            sep = {"u": UNIT, "t": TAB, "s": " "}[(sep + "u").lower()[0]]
        if sep == " " and self.type not in []:
            add_error2el(
                ColumnError(
                    "Incorrect separator+datatype combo",
                    details=f"Spaces are not allowed for rows of type {self.type}",
                )
            )
            sep = UNIT

        self.max = max_vals
        self.min = min_vals
        self.sep = sep

        # print(self.name, self.range_string)

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
        self, in_vals: str | List, el: ErrorList = None, line_num: int = -1
    ) -> Optional[List]:
        """Converts the input string into a list of values

        :param in_vals:
        :param el:
        :param line_num:
        :return:
        """
        if in_vals is None:
            return in_vals
        if isinstance(in_vals, str):
            in_vals = [v for v in in_vals.split(self.sep) if v]
        out_vals = [
            self.type.from_str(v, el, line_num, calling_column=self) for v in in_vals
        ]

        e_st = f"{len(out_vals)} values provided for a column which expects "
        if len(out_vals) < self.min:
            add_error2el(
                TooFewValuesError(
                    "Not enough values", out_vals, e_st + str(self.min), line_num
                ),
                el,
            )
        if len(out_vals) > self.max:
            add_error2el(
                TooManyValuesError(
                    "Too many values", out_vals, e_st + str(self.max), line_num
                ),
                el,
            )

        return out_vals

    def to_str(self, vals: List) -> str:
        return self.sep.join(str(v) for v in vals)

    def is_valid(self, vals: List) -> bool:
        return (
            all(isinstance(v, self.type.wraps) for v in vals)
            and self.min <= len(vals) <= self.max
        )


class ColumnTemplate(ColumnDefinition):
    is_template = True

    def __call__(self, column_name: str, **kwargs) -> ColumnDefinition:
        """
        :param column_name: the name of the new ColumnDefinition to return
        :param kwargs: junk to maintain class hierarchy compatibility
        :return: A ColumnDefinition based on self
        """
        return ColumnDefinition(column_name, self.type, self.min, self.sep, self.max)

    @staticmethod
    def at_least_range(min_: int) -> str:
        return f"{min_}-"

    @staticmethod
    def at_least(min_: int, type_) -> ColumnTemplate:
        return ColumnTemplate(
            f"at least {min_} {type_}", min_vals=min_, data_type=type_
        )

    @staticmethod
    def at_most(max_: int, type_) -> ColumnTemplate:
        return ColumnTemplate(f"at most {max_} {type_}", max_vals=max_, data_type=type_)

    @staticmethod
    def at_most_range(max_: int) -> str:
        return f"-{max_}"

    @staticmethod
    def between_range(min_: int, max_: int) -> str:
        return f"{min_}-{max_}"

    @staticmethod
    def between(min_: int, max_: int, type_) -> ColumnTemplate:
        return ColumnTemplate(
            f"between {min_} and {max_} {type_}",
            data_type=type_,
            min_vals=min_,
            max_vals=max_,
        )


DataHintDict: "Dict[Union[chr, Type], Type[BaseValue]]" = {}
_TemplateRangeDict: Dict[str, str] = {
    "AtLeastOne": ColumnTemplate.at_least_range(1),
    "AtMostOne": ColumnTemplate.at_most_range(1),
    "ExactlyOne": ColumnTemplate.between_range(1, 1),
    "Unlimited": "",
}
_TemplateTypeDict: Dict[str, Type] = {
    "String": str,
    "Int": int,
    "Float": float,
    # "Money": CurrencyValue,
    # "Date": ,
    # "Time": ,
    # "RelativeTime": RelativeDatetimeValue,
    # "Fraction": ,
}
for (range_name, range_string), (type_name, type_type) in itertools.product(
    _TemplateRangeDict.items(), _TemplateTypeDict.items()
):
    ...  # generate ColumnTemplates
    # print(f"{range_name}{type_name}")


class BaseValue(abc.ABC):
    @staticmethod
    @abc.abstractmethod
    def from_str(
        value: str,
        error_list: Optional[ErrorList] = None,
        line_num: int = -1,
        *others,
        **extras,
    ):
        ...

    def __init_subclass__(cls, hint: str = None, wraps: Type = None, **kwargs):
        """
        Adds the new class to register its type handling.
        Useful for extending this module and overriding a class.
        For example, override any date and time wrapper classes with
        classes that use the DateUtil package rather than relying solely
        on the standard library.
        """
        super().__init_subclass__(**kwargs)
        DataHintDict[cls] = cls
        if hint:
            DataHintDict[hint.upper()] = cls
        if wraps:
            DataHintDict[wraps] = cls
            setattr(cls, "wraps", wraps)
        else:
            setattr(cls, "wraps", cls)


class IntWrapper(BaseValue, hint="I", wraps=int):
    @staticmethod
    def from_str(
        value: str,
        error_list: Optional[ErrorList] = None,
        line_num: int = -1,
        *others,
        **extras,
    ):
        try:
            x = float(value)
            if x.is_integer():
                return int(x)
            add_error2el(
                ValueTypeError(
                    f"Non-integer number", value, f"'{value}' = {x}", line_num=line_num
                ),
                error_list,
            )
            return x
        except ValueError as e:
            add_error2el(
                ValueTypeError(
                    f"Not a number",
                    value,
                    line_num=line_num,
                    details=f"{value} is not a number.",
                ),
                error_list,
            )
            return value


class FloatWrapper(BaseValue, hint="D", wraps=float):
    @staticmethod
    def from_str(
        value: str,
        error_list: Optional[ErrorList] = None,
        line_num: int = -1,
        *others,
        **extras,
    ):
        try:
            return float(value)
        except ValueError as e:
            add_error2el(ValueTypeError(f"Cannot create float", value), error_list)
            return value


class StringWrapper(BaseValue, hint="S", wraps=str):
    @staticmethod
    def from_str(
        value: str,
        error_list: Optional[ErrorList] = None,
        line_num: int = -1,
        *others,
        **extras,
    ):
        return value


class RelativeDatetimeValue(BaseValue, hint="E"):
    def __init__(self, unit: chr, distance: int):
        self.distance = distance
        self.unit = unit.upper()
        super().__init__(str(self))

    @property
    def value(self):
        return f"{self.unit}{'' if self.distance < 0 else '+'}{self.distance}"

    @staticmethod
    def from_str(
        value: str,
        error_list: Optional[ErrorList] = None,
        line_num: int = -1,
        *others,
        **extras,
    ):
        ...

    def __call__(self, dt: datetime | int):
        if isinstance(dt, int):
            return dt + self.distance

    def __str__(self):
        ...


class CurrencyError(ValueTypeError):
    pass


class CurrencyValue(BaseValue, hint="C"):
    def __init__(
        self,
        value,
        currency: str = None,
        precision: Optional[int] = None,
        line_num: int = -1,
        el: Optional[ErrorList] = None,
        called_by: Union[ColumnDefinition, CurrencyValue] = None,
        **_kwargs,
    ):
        changed = True
        self.value = Decimal(value)
        real_decimals = -self.value.as_tuple()[2]
        if precision is None and called_by is not None:
            precision = called_by.currency_precision
        if precision is not None and precision != real_decimals:
            add_error2el(
                CurrencyError(
                    "Incorrect decimal places in currency",
                    value,
                    f"{real_decimals} decimal places given in a currency with {precision}",
                    line_num,
                ),
                el,
            )
        if currency is None:
            currency = "XXX" if called_by is None else called_by.currency_code
        self.currency_code = currency
        if (
            called_by
            and precision == called_by.currency_precision
            and currency == called_by.currency_code
        ):
            changed = False
        self.currency_precision = real_decimals if precision is None else precision
        self.changed_from_column = changed

    @staticmethod
    def from_str(
        value: str,
        error_list: Optional[ErrorList] = None,
        line_no: int = -1,
        calling_column: Optional[ColumnDefinition] = None,
        *others,
        **extras,
    ):
        value = value.upper().split()
        amount = value[0]
        precision = None
        currency = None if len(value) < 2 else value[-1]
        if len(value) > 2:
            try:
                precision = int(value[1])
            except ValueError:
                add_error2el(
                    ValueTypeError(
                        "Undefined currency precision",
                        value[1],
                        line_num=line_no,
                        details=f"{value[1]} is not an integer",
                    ),
                    error_list,
                )
        try:
            return CurrencyValue(
                amount,
                currency,
                precision,
                line_num=line_no,
                called_by=calling_column,
                el=error_list,
            )
        except CurrencyError as e:
            add_error2el(e, error_list)
            return CurrencyValue(amount, currency, None, True, line_num=line_no)
        except (ValueError, decimal.InvalidOperation) as e:
            add_error2el(
                ValueTypeError("Cannot convert to currency", value, str(e), line_no),
                error_list,
            )
        finally:
            return " ".join(value)

    def __str__(self):
        if not self.changed_from_column:
            return str(self.value)
        return f"{self.value} {self.currency_precision} {self.currency_code}"

    def __repr__(self):
        return "".join(
            [
                f"currency_{self.currency_code}",
                "!" if self.changed_from_column else "",
                f"({self.value} / {self.currency_precision})",
            ]
        )

    def make_currency_column(
        self,
        name: str,
        max_vals: int = sys.maxsize,
        sep: chr = UNIT,
        min_vals: int = -1,
        el: Optional[ErrorList] = None,
    ) -> ColumnDefinition:
        return ColumnDefinition(
            name,
            CurrencyValue,
            max_vals,
            sep,
            min_vals,
            currency_code=self.currency_code,
            decimal_places=self.currency_precision,
            el=el,
        )


ColumnType = TypeVar("ColumnType", bound=ColumnDefinition)
ValueType = TypeVar("ValueType", bound=BaseValue)
