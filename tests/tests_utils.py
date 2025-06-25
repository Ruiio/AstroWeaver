# tests/test_text_utils.py

import unittest
from weaver.utils.text_utils import sanitize_filename, normalize_text


class TestTextUtils(unittest.TestCase):

    def test_sanitize_filename_basic(self):
        """测试基本的文件名清理功能。"""
        self.assertEqual(sanitize_filename("Mars (planet)"), "mars_planet")
        self.assertEqual(sanitize_filename("NGC 1333 / LBN 741"), "ngc_1333_lbn_741")
        self.assertEqual(sanitize_filename("  leading & trailing spaces  "), "leading_trailing_spaces")
        self.assertEqual(sanitize_filename("multiple--dashes__and_underscores"), "multiple_dashes_and_underscores")

    def test_sanitize_filename_unicode(self):
        """测试对Unicode和特殊字符的处理。"""
        self.assertEqual(sanitize_filename("Betelgeuse (α Orionis)"), "betelgeuse_orionis")
        self.assertEqual(sanitize_filename("München"), "munchen")

    def test_sanitize_filename_edge_cases(self):
        """测试边界情况。"""
        self.assertEqual(sanitize_filename(""), "unnamed_entity")
        self.assertEqual(sanitize_filename("___"), "unnamed_entity")
        self.assertEqual(sanitize_filename("!@#$%^&*()"), "unnamed_entity")

        long_name = "a" * 150
        self.assertEqual(len(sanitize_filename(long_name)), 100)

    def test_normalize_text(self):
        """测试文本规范化功能。"""
        text = "This   is a    sentence with\n\nmultiple spaces\tand newlines. "
        expected = "This is a sentence with multiple spaces and newlines."
        self.assertEqual(normalize_text(text), expected)

        self.assertEqual(normalize_text("  hello world  "), "hello world")
        self.assertEqual(normalize_text("\n\t  "), "")
        self.assertEqual(normalize_text(123), "")  # 测试非字符串输入


if __name__ == '__main__':
    unittest.main()