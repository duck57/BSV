# Python implementation

## Implementation-specific information

* Data from `comment` headers is saved into the respective table and column
  object `__doc__`.
* Data from `client` headers gets saved into `__annotations__`?
* Where do `extras` go?
* Due to the limitations of the datetime objects of the standard library,
  (relative) date(time)s may not properly parse even though they are valid.
  Because of this, there is no strict validation on those items.  An extension
  module with the `dateutil` dependency may be written at a later date.

## Relative date rounding

`r` is the non-integer number

* Hour: `seconds = round(3600*r)`
* Day: `hours = round(24*r)` or maybe throw an error
* Week: `days = round(7*r)`
* Month: `days = round(30.5*r)`
* Year (current implementation): `days = round(365.25*r)`
* Year (`dateutil` implementation): `months = round(12*r)` when `0.9 > r % 1 >
  0.1`.  Otherwise, round to days.

# Help Wanted

* Suggest and/or write tests
* How to organize the files?

## Style considerations

Please use `black` to format your files before committing.
