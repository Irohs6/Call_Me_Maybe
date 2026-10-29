"""Tests for the original JsonChecker and its required parameter schema."""

import unittest

from src.json_checker import (
    JsonBooleanParser, JsonChecker, JsonNumberParser, JsonStringParser,
)


class JsonCheckerTests(unittest.TestCase):
    """Check valid, incomplete and invalid JSON object prefixes."""

    def test_string_values(self) -> None:
        checker = JsonChecker(parameters={"name": "string"})
        valid = (
            '{"name":""}',
            '{"name":"Iroh"}',
            r'{"name":"Hello \"Iroh\""}',
            r'{"name":"line\nnext"}',
            r'{"name":"\u0041"}',
            r'{"na\u006de":"Iroh"}',
            '{"name":"é😀"}',
        )
        for text in valid:
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), 0)

    def test_incomplete_strings(self) -> None:
        checker = JsonChecker(parameters={"name": "string"})
        prefixes = (
            "", "{", '{"', '{"na', '{"name"', '{"name":',
            '{"name":"', '{"name":"hello', '{"name":"hello"',
            '{"name":"\\', r'{"name":"\u00',
        )
        for text in prefixes:
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), -1)

    def test_invalid_strings(self) -> None:
        checker = JsonChecker(parameters={"name": "string"})
        invalid = (
            '{"name":Iroh}',
            '{"name":"bad\ntext"}',
            '{"name":"bad\ttab"}',
            r'{"name":"\x"}',
            r'{"name":"\u0X41"}',
            '{"name":"ok" "other":"x"}',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), -2)

    def test_numbers(self) -> None:
        checker = JsonChecker(parameters={"value": "number"})
        valid = (
            "0", "-0", "1", "-1", "123456",
            "0.5", "-0.75", "12.0", "1e2",
            "1E-2", "12.45e+6", "-1.2E-10",
        )
        for number in valid:
            text = '{"value":' + number + '}'
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), 0)
                self.assertEqual(checker.status(text[:-1]), -1)

    def test_invalid_numbers(self) -> None:
        checker = JsonChecker(parameters={"value": "number"})
        invalid = (
            "+1", ".5", "--1", "-.5", "01", "-01",
            "1..2", "1.e2", "1e", "1e+",
            "2e--3", "1.2.3", "NaN", "Infinity",
            "0x10", "1a", "1e2e3", "-",
        )
        for number in invalid:
            text = '{"value":' + number + '}'
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), -2)

    def test_booleans(self) -> None:
        checker = JsonChecker(parameters={"enabled": "boolean"})
        for value in ("true", "false"):
            with self.subTest(value=value):
                self.assertEqual(
                    checker.status('{"enabled":' + value + '}'), 0
                )
        for prefix in ("t", "tr", "tru", "f", "fa", "fal", "fals"):
            with self.subTest(prefix=prefix):
                self.assertEqual(
                    checker.status('{"enabled":' + prefix), -1
                )

    def test_invalid_booleans(self) -> None:
        checker = JsonChecker(parameters={"enabled": "boolean"})
        for value in ("True", "FALSE", "tru", "fals", "truth", "falsee"):
            with self.subTest(value=value):
                self.assertEqual(
                    checker.status('{"enabled":' + value + '}'), -2
                )

    def test_schema_keys_any_order(self) -> None:
        checker = JsonChecker(parameters={"a": "number", "b": "number"})
        valid = (
            '{"a":2,"b":3}',
            '{"b":3,"a":2}',
            '{"a":2 , "b":-3.5e2}',
            r'{"\u0061":2,"b":3}',
        )
        for text in valid:
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), 0)

    def test_schema_rejects_missing_unknown_and_duplicate(self) -> None:
        checker = JsonChecker(parameters={"a": "number", "b": "number"})
        invalid = (
            "{}", '{"a":2}', '{"b":3}',
            '{"x":1,"a":2,"b":3}',
            '{"a":2,"c":3}',
            '{"a":2,"a":3,"b":4}',
            '{"a":2,"b":3,"a":4}',
            '{"a":2,"b":3,',
            '{"a":2,"b":3},',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), -2)

    def test_key_prefixes_are_checked_immediately(self) -> None:
        checker = JsonChecker(parameters={"a": "number", "b": "number"})
        self.assertEqual(checker.status('{"z'), -2)
        self.assertEqual(checker.status('{"a":2,"a'), -2)
        self.assertEqual(checker.status('{"a":2,"b":3,"'), -2)
        self.assertEqual(checker.status('{"a'), -1)
        self.assertEqual(checker.status('{"a":2,"b'), -1)

    def test_key_prefix_does_not_reject_valid_escapes(self) -> None:
        checker = JsonChecker(parameters={"name": "string"})
        self.assertEqual(checker.status('{"na'), -1)
        self.assertEqual(checker.status('{"namz'), -2)
        self.assertEqual(checker.status('{"xyz'), -2)
        self.assertEqual(checker.status(r'{"\u006e'), -1)
        self.assertEqual(checker.status(r'{"\u006eame":"ok"}'), 0)

        quoted = JsonChecker(parameters={'a"b': "string"})
        self.assertEqual(quoted.status(r'{"a\"'), -1)
        self.assertEqual(quoted.status(r'{"a\"b":"ok"}'), 0)

    def test_small_structure_helpers(self) -> None:
        checker = JsonChecker(parameters={})
        self.assertTrue(checker.is_open_brace("{"))
        self.assertTrue(checker.is_close_brace("}"))
        self.assertTrue(checker.is_colon(":"))
        self.assertTrue(checker.is_comma(","))
        self.assertFalse(checker.is_colon(","))

    def test_schema_rejects_wrong_types(self) -> None:
        checker = JsonChecker(parameters={
            "name": "string",
            "enabled": "boolean",
            "value": "number",
        })
        invalid = (
            '{"name":true,"enabled":false,"value":3}',
            '{"name":"ok","enabled":"false","value":3}',
            '{"name":"ok","enabled":false,"value":"3"}',
            '{"name":"ok","enabled":false,"value":true}',
        )
        for text in invalid:
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), -2)

    def test_integer_rejects_decimal_and_exponent(self) -> None:
        checker = JsonChecker(parameters={"count": "integer"})
        for value in ("0", "-12", "123"):
            with self.subTest(value=value):
                self.assertEqual(
                    checker.status('{"count":' + value + '}'), 0
                )
        for value in ("1.0", "-1.5", "1e2", "1E-3"):
            with self.subTest(value=value):
                self.assertEqual(
                    checker.status('{"count":' + value + '}'), -2
                )

    def test_complete_mixed_values(self) -> None:
        checker = JsonChecker(parameters={
            "name": "string",
            "count": "integer",
            "enabled": "boolean",
        })
        text = '{"enabled":false, "name":"Iroh", "count":-42}'
        self.assertEqual(checker.status(text), 0)
        for i in range(len(text) + 1):
            with self.subTest(prefix=text[:i]):
                self.assertNotEqual(checker.status(text[:i]), -2)

    def test_empty_schema_only_accepts_empty_object(self) -> None:
        checker = JsonChecker(parameters={})
        self.assertEqual(checker.status("{}"), 0)
        self.assertEqual(checker.status(" { } "), 0)
        self.assertEqual(checker.status('{"a'), -2)

    def test_incomplete_schema(self) -> None:
        checker = JsonChecker(parameters={"a": "number", "b": "number"})
        incomplete = (
            '{"a', '{"a"', '{"a":', '{"a":2', '{"a":2,',
            '{"a":2,"b', '{"a":2,"b":', '{"a":2,"b":-',
        )
        for text in incomplete:
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), -1)

    def test_primitive_parsers(self) -> None:
        string_parser = JsonStringParser()
        for char in r'"\u0041"':
            result = string_parser.read(char)
        self.assertEqual(result, 0)

        incomplete = JsonStringParser()
        for char in r'"\u00':
            self.assertEqual(incomplete.read(char), -1)
        self.assertEqual(JsonStringParser().read("x"), -2)

        number_parser = JsonNumberParser()
        for char in "-12.4e+2}":
            result = number_parser.read(char)
        self.assertEqual(result, 0)

        invalid_number = JsonNumberParser()
        invalid_number.read("0")
        self.assertEqual(invalid_number.read("1"), -2)

        boolean_parser = JsonBooleanParser()
        for char in "false}":
            result = boolean_parser.read(char)
        self.assertEqual(result, 0)

    def test_escaped_key_quotes(self) -> None:
        checker = JsonChecker(parameters={'a"b': "string"})
        self.assertEqual(checker.status(r'{"a\"b":"ok"}'), 0)


if __name__ == "__main__":
    unittest.main()
