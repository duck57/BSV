from __future__ import annotations

import abc
import dataclasses
import decimal
import re
import sys
from datetime import datetime
from decimal import Decimal

from errors import *

"""
Classes and function for handling data validation in BSV files.

Plus some other definitions that are re-used in BSV.py


NOTE: importing this module also creates classes with names
starting with 'ExactlyOne', 'AtLeastOne', etc… into the global
namespace.  It is so you can use commands like 

>>> optional_dollar = AtMostOneCurrency('Optional Dollar Amount', '2 USD')

to create column definitions with commonly-used patterns.
Your IDE, however, may not like using these and complain about
names not being defined.
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
    """
    Class which defines a column.

    Once an instance has been created, the instance is callable.
    Calling the instance will convert the input list into a list of
    values that match the column type, rather than remaining as mere
    strings.
    """

    is_template: bool = True
    DataHints: "Dict[Union[chr, Type], Type[BaseValue]]" = {}

    def __init__(
        self,
        name: str,
        data_type: Union[Type, str] = str,
        max_vals: int = sys.maxsize,
        sep: chr = UNIT,
        min_vals: int = 0,
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
        self.type = self.DataHints.get(data_type)
        if self.type is None:
            self.type = StringWrapper
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
            except ValueError:
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
        if (
            sep == " "
            and not self.type.space_separable
            or sep == "\t"
            and not self.type.tab_separable
        ):
            sep = {" ": "Spaces", "\t": "Tabs"}.get(sep, "Invalid separators")
            add_error2el(
                ColumnError(
                    "Incorrect separator+datatype combo",
                    details=f"{sep} are not allowed for rows of type {self.type}",
                )
            )
            sep = UNIT

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
                self.type.dh_chr,  # noqa  set with setattr in __init_subclass__
                self.range_string,
                getattr(self, "comment", None),
                getattr(self, "client", None),
            ]
            + getattr(self, "extra_headers", []),
            UNIT,
        )

    @property
    def dataclass_field(self) -> dataclasses.Field:
        return dataclasses.field(
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

        :param in_vals: the value(s) to convert
        :param el: optional error list for error collection
        :param line_num: line number for debugging and error messages
        :return: The list of values converted into their respective datatype
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
            all(
                isinstance(v, self.type.wraps) for v in vals  # noqa
            )  # type.wraps is set in __init_subclass__ during type creation
            and self.min <= len(vals) <= self.max
        )


class ColumnTemplate(ColumnDefinition):
    """
    Template for other column definitions.

    Calling an instance of this class will create a column in the instance's
    image with the specified name.

    Class methods at_least, at_most, etc… are convenience wrappers for
    _meta_template.
    """

    is_template = True
    generic_ranges: Dict[str, Callable[[Any, str], ColumnTemplate]] = {}
    template_ready_types: Dict[str, Type] = {}

    def __call__(
        self, column_name: str, currency_string: str = "", el=None, **kwargs
    ) -> ColumnDefinition:
        """
        :param column_name: the name of the new ColumnDefinition to return
        :param currency_string: the "2 USD" precision and currency code specification
        to be used in currency columns
        :param el: optional ErrorList for catching errors during column creation
        :param kwargs: junk to maintain class hierarchy compatibility
        :return: A ColumnDefinition based on self
        """
        if currency_string:
            precision, code = currency_string.split()
            currency_info = {"currency_code": code, "decimal_places": int(precision)}
        else:
            currency_info = {}
        return ColumnDefinition(
            column_name,
            self.type,
            self.max,
            self.sep,
            self.min,
            **currency_info,
            el=el,
        )

    @staticmethod
    def _add_generic_template_to_globals(
        quantity_prefix: str,
        meta_template: Callable[[Any], ColumnTemplate],
        type_name: str,
        column_type,
    ):
        """Injects generic templates into the global namespace for client modules to use

        > ColumnTemplate._add_generic_template_to_globals("Single", ColumnTemplate.OnlyOne, "Int", int)

        Clients may now use SingleInt() as a template to create their specific columns
        """
        template = meta_template(column_type)
        globals()[quantity_prefix + type_name] = template

    @classmethod
    def register_meta_template(cls, name: str, template_creation_function: Callable):
        """Register a meta template and then inject concrete generic templates into globals()"""
        cls.generic_ranges[name] = template_creation_function
        for type_name, type_ in cls.template_ready_types.items():
            cls._add_generic_template_to_globals(
                name, template_creation_function, type_name, type_
            )

    @classmethod
    def _meta_template(
        cls,
        name_stem: str,
        min_: Optional[int] = None,
        max_: Optional[int] = None,
        _type_=None,
    ) -> ColumnTemplate | Callable[[Any, str], ColumnTemplate]:
        """Creation function for generic ColumnTemplates (used during module import)

        :param name_stem: The opening part of the class names for concrete
        generic column templates, typically a quantity, like `Several` or
        `ExactlyTwo`
        :param min_: minimum number of values in the column, typically 0 or 1
        :param max_: maximum number of values in the column, typically 1 or None
        :param _type_: setting a type will return a concrete generic ColumnTemplate
        instance of this type.  Leaving it None will return the function.
        :return: Either a concrete instance of a generic ColumnTemplate or
        a function which will create generic ColumnTemplates
        """

        def create_generic_template(type_, n: str,) -> ColumnTemplate:
            """Creates a concrete instance of a generic ColumnTemplate.

            :param type_: The type of column to create
            :param n: Same as name_stem in the outer function
            :return: A generic ColumnTemplate with min and max set by the
            outer function.
            """
            args = {"data_type": type_}
            if min_ is not None:
                args["min_vals"] = min_
                n += f" {min_}"
            if max_ is not None:
                args["max_vals"] = max_
                if max_ != min_:
                    n += f" and {max_}"
            n += f" {type_ if isinstance(type_, str) else type_.__name__}"
            return ColumnTemplate(n, **args)

        if _type_:  # template for a concrete type
            return create_generic_template(_type_, name_stem)

        def _new_template(type_) -> ColumnTemplate:
            return create_generic_template(type_, name_stem)

        return _new_template

    @classmethod
    def at_least(cls, min_: int, type_=None):
        return cls._meta_template("at least", min_, None, type_)

    @classmethod
    def at_most(cls, max_: int, type_=None):
        return cls._meta_template("at most", None, max_, type_)

    @classmethod
    def between(cls, min_: int, max_: int, type_=None):
        return cls._meta_template("between", min_, max_, type_)

    @classmethod
    def exactly(cls, number: int, type_=None):
        return cls._meta_template("exactly", number, number, type_)

    @classmethod
    def unlimited(cls, type_=None):
        return cls._meta_template("unlimited", None, None, type_)

    @classmethod
    def create_templates4type(cls, col_type: Type[BaseValue]):
        cls.template_ready_types[col_type.short_name] = col_type
        for r_name, r_fun in cls.generic_ranges.items():
            cls._add_generic_template_to_globals(
                r_name, r_fun, col_type.short_name, col_type
            )


# Not defined in the ColumnTemplate class due to initialization issues during import
for prefix, quant in {
    "AtLeastOne": ColumnTemplate.at_least(1),
    "SingleOptional": ColumnTemplate.at_most(1),
    "AtMostOne": ColumnTemplate.at_most(1),
    "ExactlyOne": ColumnTemplate.exactly(1),
    "Unlimited": ColumnTemplate.unlimited(),
}.items():
    ColumnTemplate.register_meta_template(prefix, quant)


class BaseValue(abc.ABC):
    tab_separable: bool = True
    space_separable: bool = False
    wraps: Type = None
    dh_chr: chr = ""
    short_name: str = ""

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

    # should some of this be done with a metaclass instead?
    # make wraps, short_name, and hint class attributes that are parsed
    # by the __new__ of the metaclass as an alternate strategy
    def __init_subclass__(
        cls, hint: chr = "", wraps: Type = None, short_name: str = None, **kwargs,
    ):
        """
        Adds the new class to register its type handling.
        Useful for extending this module and overriding a class.
        For example, override any date and time wrapper classes with
        classes that use the DateUtil package rather than relying solely
        on the standard library.
        """
        super().__init_subclass__(**kwargs)
        ColumnDefinition.DataHints[cls] = cls

        if hint:
            ColumnDefinition.DataHints[hint.upper()] = cls
            cls.dh_chr = hint
        else:
            raise KeyError(f"{cls.__name__} is missing a data hint")

        if wraps:
            ColumnDefinition.DataHints[wraps] = cls
            cls.wraps = wraps
        else:
            cls.wraps = cls

        if short_name:
            cls.short_name = short_name
            ColumnTemplate.create_templates4type(cls)


class IntWrapper(BaseValue, hint="I", wraps=int, short_name="Int"):
    space_separable = True

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
        except ValueError:
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


class FloatWrapper(BaseValue, hint="D", wraps=float, short_name="Float"):
    space_separable = True

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
        except ValueError:
            add_error2el(
                ValueTypeError(
                    f"Cannot create float",
                    value,
                    f"{repr(value)} is not a float",
                    line_num,
                ),
                error_list,
            )
            return value


class StringWrapper(BaseValue, hint="S", wraps=str, short_name="String"):
    tab_separable = False

    @staticmethod
    def from_str(
        value: str,
        error_list: Optional[ErrorList] = None,
        line_num: int = -1,
        *others,
        **extras,
    ):
        return value


class RelativeDatetimeValue(BaseValue, hint="E", short_name="RelativeDate"):
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


class CurrencyValue(BaseValue, hint="C", short_name="Currency"):
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
