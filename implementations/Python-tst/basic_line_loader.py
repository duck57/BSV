import unittest
from BSV import *
from BSV import _split_file_into_rows

"""
Proof of concept tests to test the tests.
"""


def generate_test_string() -> str:
    """
    :return: a test string with a bunch of non-printable characters
    """
    o = "File separator" + FILE
    o += "Group separator" + GROUP
    o += "Record separator" + RECORD
    o += "Unit separator" + UNIT
    o += "Newline\n"
    o += "Tab (horizontal)\t"
    o += "Tab (vertical)\v"
    o += "Bell\a\x07"
    o += "Carriage return\r"
    o += "End of file"
    return o


class TestTests(unittest.TestCase):
    def test_test_string(self):
        self.assertIn(RECORD, generate_test_string())

    def test_reading_file_into_lines(self, f="../../test_BSV_files/test_string.txt"):
        """
        Tests reading a file into lines with a very short reading buffer
        The short buffer exercises how the loader handles lines that span reads
        """
        with open(f) as opened_file:
            for z in _split_file_into_rows(opened_file, 40):
                print(z)
                self.assertIn(z.ending, LINE_BREAKS)


if __name__ == "__main__":
    unittest.main()
