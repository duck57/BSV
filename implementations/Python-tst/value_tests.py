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

    def test_RDS(self):
        self.assertEqual(0, RelativeDatetimeString.from_str("Y-6.66")().days)
        self.assertEqual(
            56.789, RelativeDatetimeString.from_str("H+12:34:56.789")().seconds
        )

    def test_RDS_from_relativedelta(self):
        self.assertEqual(
            "H+3", RelativeDatetimeString(distance=relativedelta(hours=3)).__str__()
        )
        self.assertEqual(
            "H-6:45",
            RelativeDatetimeString(
                distance=relativedelta(hours=-6, minutes=-45)
            ).__str__(),
        )
        self.assertEqual(
            "H+6:06:06.666",
            RelativeDatetimeString(
                distance=relativedelta(hours=6, minutes=6, seconds=6.666)  # noqa
            ).__str__(),
        )

    def test_RDS_abs_neg(self):
        self.assertEqual("T+6", (-RelativeDatetimeString("T", -6)).__str__())
        self.assertEqual("Y+3", abs(RelativeDatetimeString("Y", -3)).__str__())


if __name__ == "__main__":
    unittest.main()
