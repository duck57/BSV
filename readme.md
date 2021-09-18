# Better Separated Values

An improvement to CSV.  Current version = 0.0.4 ^ 0/1

## Inspiration

This project was inspired by the discussions inspired by an article complaining
about the follies of CSV.  One of the points raised was that ASCII characters
28–31 specifically exist to delineate data using characters that should never
appear in real data.  The other improvement is an ability to store multiple values
in the same field without implementing a one-off custom delimiter.

## Project Milestones & Goals

### Pre-1.0 versioning

* In 0.0.x, each patch may be a major version in a post-1.0 semver release
* Likewise for 0.y.x for each y-increment.
* The ` ^ a/b` is the number of relatively-complete implementations and the
  number of heavily in-progress implementations

### 0.1.x

* Minimal read/write implementations in at least two languages.
    * Elixir, Python, & Clojure are near the top of the list
* No obvious (to me) stupid decisions
* Seek community comments
* Resolve TODO sections and statements with question marks

### 1.0

* No further format changes are proposed after robust outside feedback
* The implementations should probably be split into separate projects at this
  point.

# Semi-formal specification

## Notes

* Encoding in these documents is assumed to be UTF-8
* "Record" and "row" are used interchangeably throughout.  Do remember that
  these are delimited with `\x1D` and not a newline.
* Whether to trim trailing & leading whitespace from string fields is left to
  the discretion of the client.  Libraries ought to be happy to serve either
options.  Non-string fields should strip such whitespace.
* All requirements for "unique names" are to be treated as case-insensitive.

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

Furthermore, a newline character may optionally be placed after the FILE or
GROUP separators for ease of editing in text editors.  This newline will
promptly be stripped and ignored when the file is loaded.

## General Table Structure

Each table consists of:

1. Table header row
2. Column header row
3. 0 or more data rows


* Multiple tables within a single file are separated by `\x1C`.
* Re-using a table name header should pull up the previously-defined header and
  therefore does not need a second row to re-define columns.
* Table and column names should be insensitive to case and whitespace.

### Note on Comment, Client, and Extras… fields

In both the table and column definitions, there are fields labeled `comment`,
`client`, and `extras…`.  These fields have identical meaning in both.

#### Comment

A human-readable comment about the table or column.

#### Client

A field for miscellaneous data to be read by the client software.

#### Extras…

Other extraneous fields.  Their vacancy should not be relied upon when writing
BSV files with forward-compatibility in mind.

## Table Header

    table name<RECORD>options string 1<RECORD>comment<RECORD>client<RECORD>extras…

Additional fields after the comment are currently unused but should not be
relied upon to remain empty for future use.  The only required information is
the table name.

### Options String 1

One-character flags that set up the integrity options for the table as a whole.
Defaults are all no. 

Maximum and minimum numbers of fields per row should both be handled by client
software. A good use case for the `Client` field.

#### Allow extra fields? `X`

If seven column headers are defined, allow records with eight or more fields.
The extra fields are to be treated as string fields with unlimited multi-values
each.

#### Allow short rows? `S`

If seven column headers are defined, allow records with six or fewer fields.

### TODO before 0.2.x

Are there other table-wide options to consider?

## Column Headers

    field 1<RECORD>field 2<UNIT>options…<RECORD>field 3…

The column header row is a series of column headers separated by `\x1E`.  Each
column header consists of a presumed-unique field name optionally followed by
`\x1F` and validation hinting options (each separated by an additional `\x1F`).
The behaviour of duplicate field names  is left deliberately undefined.

The structure of a column's definition includes the following headers in order.
Only the field name is required.  Each header is separated with `\x1F`.

1. Field name
2. Data hint.  Defaults to String.  See below for the list of data types.  The
   first character of this header determines which validation is to be used.
3. Range of number of valid numbers of values per entry separated with `-`:
   `min-sep-max`.  If `min` is omitted, it defaults to 0; if `max` is omitted,
   it defaults to the system's maximum integer.  See below for notes on the
   separator.
4. Comment
5. Client
6. Extras…

### Range configurations

While there is no enforced limit to the values of `min`, the only sensible ones
are 0 and 1. Rather than have a range of `6--14`, it may be more sensible to
have six columns with range `1-1` and an overflow column of range `--8`.

TODO: really provide this much flexibility or just require the 3 value scenario?

#### All 3 values (2 separations)

* `min-sep-max`
* `--max`
* `min--`
* `-sep-max`
* `-sep-`
* `--`
* `min--max`
* `min-sep-`

#### 2 values `min-max`

No separator customization may be made in this configuration.

#### Single value `x`

* If `x` is numeric, `x` is `max`
* Otherwise, `x` is `sep`

### Notes on `sep`

The separator (and preceding hyphen) are optional characters which change the
separator between values in a column from the default UNIT SEPARATOR to
something easier for human editing.  Implementations should only consider the
first character of `sep`.

* `t` = tab-separated values
* `s` = space-delimited values (only valid for integers, floats, and relative
  datetime: all other types fall back to `\x1F`)
* `u` = unit separator
* anything else falls back to the unit separator
* This list may be expanded at a later date, but that is unlikely.

This is entirely ignored for columns with a String (or default) data hint.

## Supported data hints 

### String: `S`

### Integer: `I`

### Decimal: `F`

Stands for float, which is what it will presume to be stored as in the
client's RAM.

### Fraction: `R`

Fractions without a denominator will be assumed to be over 1.  Clients should
store both the numerator and denominator as integers if possible and otherwise
store them both as floats.

TODO: or should the integer/float conversion be separate for the numerator &
denominator?

### Date: `D`

An ISO 8601 date(time).  Non-ISO 8601 time stamps should be stored as regular
integers or strings.

### Time: `T`

An ISO 8601-flavored time.

### Relative date/time: `E`

General format is `X±12`.  `+` represents the future and `-` is for the past.
`X` is one of the interval abbreviations outlined below.  The number should
generally be an integer, especially for `T`, `W`, or `M`.  However, clients are
encouraged to soften this requirement by rounding non-integer values to a
sensible whole number of days.

#### Hour: `H`

If the number is a non-integer, the behavior of the number depends on the separator.

* `H±hh:mm` adds hours & minutes
* `h±H.xx` adds fractions of an hour

For 0.2.x, should the colon separator also allow for seconds or frames or are
minutes are fine-grained as necessary?

#### Day: `T`

`t+1` means to increment the day by 1.

#### Week: `W`

`w+1` == `t+7`

#### Month: `M`

`M+1` meant to increment the month by 1.

#### Year: `Y`

`Y+1` is one year in the future.

For fractional years, should they round to the nearest day or nearest month?
Should this be unspecified? 

### Currency: `C`

First, some examples:

1. C 3 BHD _one required Bahrain dinar amount_
2. C 2 USD _multiple US dollar amounts separated by tabs_
3. C 8 _up to 2 amounts of unspecified currency with 8 digits after the
   decimal separated by the unit separator_

Note how currency fields require an additional `\x1F` followed by an integer.
This integer is to indicate the number of decimal places of precision in which
the currency should be stored. Optionally, the precision may be followed by a
space and an ISO 4217-like currency code. In a field with multiple values, each
individual value _may_ choose to include a space and currency code further
followed by another optional space and precision integer. Precision integers
within individual values are not permitted without a currency code.

## Data Rows

Each row is terminated with `\x1D` GROUP SEPARATOR.  Fields within the row are
delimited with the record separator `\x1E`.

# Reading .BSV files

Reader libraries should give their client the option whether to continue
loading files with malformed rows or to abort with an error.

# Writing BSV

Output data validation should probably be done by your preferred dataclass or ORM.

# Repository structure

* `readme.md`: this document with the specification, about, and project
  information
* `implementions/`: folders containing implementations of this spec in
  different languages.  If the source and test folders are separate, create
  them as `Lang-src` and `Lang-tst`.
* `test_BSV_files/`: test case files to be shared when testing implementations

Please clean up any junk that's not already covered by .gitignore by adding new
rules there.
