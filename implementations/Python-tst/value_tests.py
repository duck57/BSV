import unittest
from columns import *


class ValueTests(unittest.TestCase):
    def test_int(self):
        self.assertEqual(-7, IntWrapper.from_str("\t\n-7.000\v"))
        self.assertRaises(ValueTypeError, IntWrapper.from_str, "  7.234 ")

    def test_str(self):
        ...

    def test_generic_int(self):
        # Note that class ExactlyOneInt is created at runtime during the import of columns.py
        single_int = ExactlyOneInt("test int")  # noqa
        self.assertEqual(42, single_int(["42"])[0])
        self.assertRaises(TooManyValuesError, single_int, ["0", 0, 0.00])

    def test_fractions(self):
        two_f = "  22\n / -7\t" + UNIT + "69/420"
        fraction_column = AtLeastOneFraction("test fractions")  # noqa
        two_f = fraction_column(two_f)
        self.assertEqual(two_f[0], Fraction(-22, 7))
        self.assertEqual(two_f[1], Fraction(69, 420))
        bad_fractions = "Incorrect slash" + UNIT + "14\\15"
        error_collector = ErrorList()
        fraction_column(bad_fractions, error_collector)
        self.assertEqual(2, len(error_collector))

    def test_RDV(self):
        self.assertEqual(0, RelativeDatetimeValue.from_str("Y-6.66")().days)
        self.assertEqual(56.789, RelativeDatetimeValue.from_str("H+12:34:56.789")().seconds)


if __name__ == "__main__":
    unittest.main()
