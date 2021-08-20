# Better Separated Values

An improvement to CSV.  Currently version 0.0.3

## Inspiration

This project was inspired by the discussions inspired by an article complaining
about the follies of CSV.  One of the points raised was that ASCII characters
28–31 specifically exist to delineate data using characters that should never
appear in real data.  The other improvement is an ability to store multiple values
in the same field without implementing a one-off custom delimiter.

## Project Milestones & Goals

### 0.1.x

* Minimal read/write implementations in at least two languages.
    * Elixir, Python, & Clojure are near the top of the list
* No obvious (to me) stupid decisions
* Seek community comments
* Resolve TODO sections and statements with question marks

### 1.0

* No further format changes are proposed after robust outisde feedback

# Semi-formal specification

## Notes

* Encoding in these documents is assumed to be UTF-8
* "Record" and "row" are used interchangably throughout.  Do remember that
  these are delimted with `\x1D` and not a newline.
* Whether to trim trailing & leading whitespace from string fields is left to
  the discretion of the client.  Libraries ought to be happy to serve either
options.  Non-string fields should strip such whitespace.

## Reserved delimiters

1. These should not appear in live data.  If they do, it's on the client to
   escape them.
2. Although one would think that \x1E (Record Separator) should be the
   delimiter between rows, it is, in fact \x1D that separates full rows.  \x1E
separates fields within a row.  This is in accordance with the MARC21 format.
3. Note that tabs, newlines, and most other ASCII control characters **are**
   valid within textual data.

|  seq  | decimal | hex  | <name> SEPARATOR | used for delimiting               |
|-------|---------|------|------------------|-----------------------------------|
| `^\`  | 28      | \x1C | FILE             | tables within a file              |
| `^]`  | 29      | \x1D | GROUP            | rows within a table               |
| `^^`  | 30      | \x1E | RECORD           | fields within a row               |
| `^_`  | 31      | \x1F | UNIT             | values within a multi-value field |
| `^I`  | 09      | \x09 | TAB              | alternate to unit separator       |

When referring to these values within the rest of this specification, the seq
code or `\x` hex value will be used.

## General Table Structure

Each table consists of:

1. Table header row
2. Column header row
3. 0 or more data rows


* Multiple tables within a single file are separated by `\x1F`.

### TODO decision before 0.1.x

How should multiple tables with the same name (case-insensitive?) be handled?

1. Overwrite/error as a malformed file
2. Continue previous table of same name (thus making the column header row a
   requirement for _only_ the first definition of a table name)
3. Leave this as deliberately unspecified/undefined behaviour that lets
   incidental implementation details shine through

## Table Header

table\_name[`\x1e`options string]

### Options String

One-character flags that set up the integrity options for the table as a whole.
Default for all these is no.  Order within this string is not important?

### Allow extra fields? `X`

If seven column headers are defined, allow records with eight or more fields.
The extra fields are to be treated as string fields with unlimited multi-values each.

TODO before 0.2.x: allow setting a maximum number of total fields per row?

### Allow short rows? `S`

If seven column headers are defined, allow records with six or fewer fields.

TODO before 0.2.x: allow setting for a minimum number of fields in each row?

### TODO before 0.2.x

Are there other table-wide options to consider?

## Column Header

field1`\x1E`field\_2`\x1F`option-2`\x1F`more`\x1E`field-3`\x1E`

The column header row is a series of column headers separated by `\x1E`.  Each
column header consists of a presumed-unique field name optionally followed by
`\x1F` and validation hinting options.  The behaviour of duplicate field names
is left deliberately undefined.

### Validation hinting options

In the general format of `X9t`.

* `X` is the letter of the format (see below)
* `9` is either a single digit or an asterisk (`*`).  If between 1–9, this is
  the maximum number of values in the field.  `0` stands for "only" and means
that a single value is required in the field.  `*` allows any number of values.
If you, for example, need to have between 3 nad 7 values, make four columns:
three of which are `0` and the fourth as `4`. 
* `t` is an optional flag whose presense means that a tab is used to separate
  values within a field rather than the unit separator.  This flag is not valid
on multi-response string fields.

#### TODO before 0.1.x

* Specify both a minumum and maximum value count?
* Allow specified maximum value counts over 9?
* Switch meanings of `0` and `*`?

### Format hints

#### String: `S`

The default format for a column (including spare fields) is `S*`.

#### Integer: `I`

#### Decimal: `F`

Stands for float, which is what it will presumed to be stored as in the
client's RAM.

#### Fraction: `R`

Fractions without a denominator will be assumed to be over 1.  Clients should
store both the numerator and denominator as integers if possible and otherwise
store them both as floats.

TODO: or should the integer/float conversion be separate for the numberator &
denominator?

#### Date: `D`

An ISO 8601 datetime.  Non-ISO 8601 time stamps should be stored as regular
integers or strings.

#### Time: `T`

An ISO 8601-flavored time.  Both fractional seconds and an additional `:` for
frames are acceptable.

* `18:03:16.5`
* `00:49:06:23`

TODO: is this redundant with the date hint?

#### Relative date/time: `E`

General format is `X±12`.  `+` represents the future and `-` is for the past.
`X` is one of the interval abbreviations outlined below.  The number should
generally be an integer, especially for `T`, `W`, or `M`.  However, clients are
encouraged to soften this requirement by rounding non-integer values to a
sensible whole number of days.

##### Hour: `H`

If the number is a non-integer, the behavior of the number depends on the separator.

* `H±hh:mm` adds hours & minutes
* `h±H.xx` adds fractions of an hour

For 0.2.x, should the colon separator also allow for seconds or frames or are
minutes are fine-grained as necessary?

##### Day: `T`

`t+1` means to incriment the day by 1.

##### Week: `W`

`w+1` == `t+7`

##### Month: `M`

`M+1` meant to increment the month by 1.

##### Year: `Y`

`Y+1` is one year in the future.

For fractional years, should they round to the nearest day or nearest month?
Should this be unspecified? 

#### Currency: `C`

First, some examples:

1. C0`\x1F`3 BHD _one required Bahraini dinar amount_
2. C\*t`\x1F`2 USD _multiple US dollar amounts separated by tabs_
3. C2`\x1F`8 _up to 2 amounts of unspecified currency with 8 digits after the
   decimal separated by the unit separator_

Note how currency fields require an additional `\x1F` followed by an integer.
This integer is to indicate the number of decimal places of precision in which
the currency should be stored.  Optionally, the precision may be followed by a
space and an ISO 4217-like currency code.  In a field with multiple values,
each individual value _may_ choose to include the same space and currency code.

## Data Rows

Each row is terminated with `\x1D` GROUP SEPARATOR.  Fields within the row are
delimited with the record separator `\x1E`.

# Reading .BSV files

Reader libraries should give their client the option whether to continue
loading files with malformed rows or to abort with an error.

# Writing BSV

Output data validation should probably be done by your preferred dataclass or ORM.

