"""Tests for the unfinished JSON structure parser."""

import unittest

from src.json_grammar import (
    JsonBooleanParser,
    JsonGrammar,
    JsonNumberParser,
    JsonStringParser,
)


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
        parser = JsonStringParser()
        for char in ('"', '\\', '/', 'b', 'f', 'n', 'r', 't'):
            with self.subTest(char=char):
                self.assertTrue(parser.is_valid_escape(char))
        self.assertFalse(parser.is_valid_escape('x'))
        self.assertTrue(parser.is_hex_digit('F'))
        self.assertFalse(parser.is_hex_digit('g'))

    def test_string_parser_keeps_own_state(self) -> None:
        parser = JsonStringParser()
        self.assertEqual(parser.read('"'), -1)
        self.assertEqual(parser.read('H'), -1)
        self.assertEqual(parser.read('\\'), -1)
        self.assertEqual(parser.read('u'), -1)
        for digit in '0041':
            self.assertEqual(parser.read(digit), -1)
        self.assertEqual(parser.read('"'), 0)
        self.assertEqual(parser.read('x'), -2)

    def test_string_parser_rejects_bad_escapes(self) -> None:
        parser = JsonStringParser()
        self.assertEqual(parser.read('"'), -1)
        self.assertEqual(parser.read('\\'), -1)
        self.assertEqual(parser.read('x'), -2)

    def test_repeated_grammar_checks_restart_strings(self) -> None:
        text = '{"a":"hello","b":"world"}'
        for i in range(len(text) + 1):
            self.assertNotEqual(self.grammar.status(text[:i]), -2)
        self.assertEqual(self.grammar.status(text), 0)
        self.assertEqual(self.grammar.status('{"c":"ok"}'), 0)

    def test_comma_requires_key(self) -> None:
        self.assertEqual(self.grammar.read_after_value(','), 'key_required')
        self.assertEqual(self.grammar.read_after_value('}'), 'done')
        self.assertEqual(self.grammar.read_key_required('}'), 'invalid')
        self.assertEqual(self.grammar.read_key_required('"'), 'key')

    def test_complete_string_values(self) -> None:
        valid = (
            '{"name":"Iroh"}',
            '{ "name" : "Iroh" }',
            '{"name":""}',
            '{"name":"Iroh","city":"Mulhouse"}',
            r'{"name":"Hello \"Iroh\""}',
            r'{"name":"C:\\Users\\Iroh"}',
            r'{"name":"line\nnext"}',
            r'{"name":"\u0041"}',
            r'{"na\u006de":"Iroh"}',
            '{"name":"é😀"}',
        )
        for text in valid:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), 0)

    def test_incomplete_string_values(self) -> None:
        incomplete = (
            '{"name":',
            '{"name":"',
            '{"name":"Iroh',
            '{"name":"Iroh"',
            '{"name":"Iroh",',
            '{"name":"Iroh","age":"',
            '{"name":"Iroh' + "\\",
            r'{"name":"\u00',
            r'{"name":"\u0041',
        )
        for text in incomplete:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -1)

    def test_complete_number_values(self) -> None:
        numbers = (
            "0", "-0", "1", "-1", "123456",
            "0.5", "-0.75", "12.0", "1e2", "1E-2",
            "12.45e+6", "-1.2E-10",
        )
        for number in numbers:
            for suffix in ("}", " }", ',"other":"ok"}'):
                text = '{"value":' + number + suffix
                with self.subTest(text=text):
                    self.assertEqual(self.grammar.status(text), 0)

    def test_incomplete_number_values(self) -> None:
        numbers = (
            "-", "1", "-12", "0", "-0",
            "1.", "-1.2", "1e", "1e+", "1E-",
            "1.25e", "1.25e-",
        )
        for number in numbers:
            text = '{"value":' + number
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -1)

    def test_invalid_number_values(self) -> None:
        numbers = (
            "+1", ".5", "--1", "-.5", "01", "-01",
            "1..2", "1.e2", "1e}", "1e+}",
            "2e--3", "1.2.3", "NaN", "Infinity",
            "0x10", "1a", "1e2e3", "1,", "-}",
        )
        for number in numbers:
            text = '{"value":' + number + "}"
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -2)

    def test_number_parser_keeps_own_state(self) -> None:
        parser = JsonNumberParser()
        for char in "-12.3e+4":
            self.assertEqual(parser.read(char), -1)
        self.assertEqual(parser.read("}"), 0)
        self.assertEqual(parser.read("1"), -2)

    def test_number_followed_by_whitespace_and_next_key(self) -> None:
        text = '{"a": 42 \n, "b": -0.125E+3}'
        self.assertEqual(self.grammar.status(text), 0)
        for i in range(len(text) + 1):
            with self.subTest(prefix=text[:i]):
                self.assertNotEqual(self.grammar.status(text[:i]), -2)

    def test_complete_boolean_values(self) -> None:
        valid = (
            '{"enabled":true}',
            '{"enabled":false}',
            '{ "enabled" : true }',
            '{"enabled":false,"active":true}',
            '{"enabled":false, "count":12, "name":"test"}',
            '{"enabled":true \n , "active":false}',
        )
        for text in valid:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), 0)

    def test_incomplete_boolean_values(self) -> None:
        prefixes = (
            '{"enabled":t',
            '{"enabled":tr',
            '{"enabled":tru',
            '{"enabled":true',
            '{"enabled":f',
            '{"enabled":fa',
            '{"enabled":fal',
            '{"enabled":fals',
            '{"enabled":false',
            '{"enabled":false,',
        )
        for text in prefixes:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -1)

    def test_invalid_boolean_values(self) -> None:
        invalid = (
            '{"enabled":True}',
            '{"enabled":FALSE}',
            '{"enabled":tru}',
            '{"enabled":fals}',
            '{"enabled":truth}',
            '{"enabled":falsee}',
            '{"enabled":truefalse}',
            '{"enabled":t2}',
            '{"enabled":false,}',
            '{"enabled":true "other":false}',
            '{"enabled":true.0}',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -2)

    def test_boolean_parser_keeps_own_state(self) -> None:
        for literal in ("true", "false"):
            parser = JsonBooleanParser()
            for char in literal:
                self.assertEqual(parser.read(char), -1)
            self.assertEqual(parser.read("}"), 0)
            self.assertEqual(parser.read("x"), -2)

    def test_boolean_parser_requires_literal_before_separator(self) -> None:
        parser = JsonBooleanParser()
        self.assertEqual(parser.read("}"), -2)

    def test_boolean_prefixes_reparse_from_start(self) -> None:
        text = '{"enabled":true,"disabled":false}'
        for index in range(len(text) + 1):
            with self.subTest(prefix=text[:index]):
                self.assertNotEqual(self.grammar.status(text[:index]), -2)
        self.assertEqual(self.grammar.status(text), 0)

    def test_invalid_string_values(self) -> None:
        invalid = (
            '{"name":Iroh}',
            '{"name":}',
            '{"name":"bad\ntext"}',
            '{"name":"bad\ttab"}',
            r'{"name":"\x"}',
            r'{"name":"\u0X41"}',
            '{"name":"ok" "age":"33"}',
            '{"name":"ok",}',
            '{"name":"ok"}extra',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(self.grammar.status(text), -2)


if __name__ == '__main__':
    unittest.main()
