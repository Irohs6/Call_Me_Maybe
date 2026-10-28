"""Tests for the unfinished JSON structure parser."""

import unittest

from src.json_grammar import JsonGrammar


class JsonGrammarTests(unittest.TestCase):
    def setUp(self) -> None:
        self.grammar = JsonGrammar()

    def test_complete_empty_object(self) -> None:
        for text in ('{}', ' { } ', '\n{}\r\n'):
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), 0)

    def test_unfinished_keys(self) -> None:
        prefixes = (
            '', '{', '{"', '{"name', '{"name"', '{"name":',
            '{ "name" :   ', '{"na\\"me"', '{"name\\u0041"',
            '{"name\\u0041":', '{"name\\u00', '{"\\',
        )
        for text in prefixes:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -1)

    def test_invalid_structure(self) -> None:
        invalid = (
            '}', '{abc', '{,', '{"name";', '{"name"x',
            '{"name\\x', '{"name\\u00xz', '{"name\n"',
            '{"name\t"', '{}extra', '{},', '[]',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -2)

    def test_escape_helpers(self) -> None:
        for char in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
            with self.subTest(char=char):
                self.assertTrue(self.grammar.is_valid_escape(char))
        self.assertFalse(self.grammar.is_valid_escape('x'))
        self.assertTrue(self.grammar.is_hex_digit('F'))
        self.assertFalse(self.grammar.is_hex_digit('g'))

    def test_comma_requires_key(self) -> None:
        self.assertEqual(self.grammar.read_after_value(','), 'key_required')
        self.assertEqual(self.grammar.read_after_value('}'), 'done')
        self.assertEqual(self.grammar.read_key_required('}'), 'invalid')
        self.assertEqual(self.grammar.read_key_required('"'), 'key')

    def test_value_is_explicitly_not_implemented(self) -> None:
        with self.assertRaises(NotImplementedError):
            self.grammar.status('{"name": "Iroh"}')


if __name__ == '__main__':
    unittest.main()
