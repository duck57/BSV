# Python implementation

## Implementation-specific information

* Data from `comment` headers is saved into the respective table and column
  object `__doc__`?
* Data from `client` headers gets saved into `__annotations__`?

## Relative date rounding

`r` is the non-integer number

* Hour: `seconds = round(3600*r)`
* Day: `hours = round(24*r)` or maybe throw an error
* Week: `days = round(7*r)`
* Month: `days = round(30.5*r)`
* Year (`dateutil` implementation): `months = round(12*r)` when `0.8 > r % 1 >
  0.2`.  Otherwise, round to days.

# Help Wanted

* Suggest and/or write tests
* How to organize the files?

## Style considerations

Please use `black` to format your files before committing.
