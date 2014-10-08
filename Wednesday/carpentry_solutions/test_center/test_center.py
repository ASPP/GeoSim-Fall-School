import unittest
import string

class TestCenter(unittest.TestCase):

    def test_center_default(self):
        # cases are (arguments, [possible results of centering])
        cases = [(('a', 5),    ['  a  ']),
                 (('a', 4),    [' a  ', '  a ']),
                 (('[]', 5),   [' []  ', '  [] ']),
                 (('[]', 4),   [' [] ']),
                 (('test', 2), ['test']),
                 (('', 3),     ['   ']),
                 ((' a', 5),   ['  a  ', '   a '])]

        # test all cases
        for (str_, width), expected in cases:
            output = string.center(str_, width)
            self.assertEqual(len(output), max(width, len(str_)))
            self.assertIn(output, expected)

    def test_center_fillchar(self):
        cases = [(('a', 5), ['**a**']),
                 (('a', 4), ['*a**', '**a*'])]

        # test all cases
        for fillchar in ['#', '5', ' ']:
            for (str_, width), expected in cases:
                output = string.center(str_, width, fillchar)
                self.assertEqual(len(output), max(width, len(str_)))
                self.assertIn(output.replace(fillchar, '*'), expected)

    def test_errors(self):
        with self.assertRaises(TypeError):
            string.center('a', '')

        with self.assertRaises(TypeError):
            string.center('a', '*#')


if __name__ == '__main__':
    unittest.main()
