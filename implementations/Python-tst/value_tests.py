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
        single_int = ExactlyOneInt("test int")
        self.assertEqual(42, single_int(["42"])[0])
        self.assertRaises(TooManyValuesError, single_int, ["0", 0, 0.00])


if __name__ == "__main__":
    unittest.main()
