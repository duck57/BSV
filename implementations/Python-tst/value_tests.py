import unittest
from columns import *


class ValueTests(unittest.TestCase):
    def test_int(self):
        self.assertEqual(-7, IntWrapper.from_str("\t\n-7.000\v"))
        self.assertRaises(ValueTypeError, IntWrapper.from_str, "  7.234 ")

    def test_str(self):
        ...


if __name__ == "__main__":
    unittest.main()
