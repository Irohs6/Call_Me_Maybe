"""Validate schema inputs and graceful CLI failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from src.__main__ import main
from src.model.models import Function
from src.model.parser import JsonParser


class InputValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.path = Path(self.tempdir.name) / "functions.json"

    @staticmethod
    def function(name: str = "fn_test") -> dict[str, object]:
        return {
            "name": name,
            "description": "Example function",
            "parameters": {"value": {"type": "integer"}},
            "returns": {"type": "boolean"},
        }

    def save(self, functions: list[object]) -> None:
        self.path.write_text(json.dumps(functions), encoding="utf-8")

    def test_supported_types_are_accepted(self) -> None:
        function = Function.model_validate(self.function())
        self.assertEqual(function.parameters["value"].type, "integer")

    def test_unknown_parameter_type_is_rejected(self) -> None:
        function = self.function()
        function["parameters"] = {"value": {"type": "object"}}
        with self.assertRaises(ValidationError):
            Function.model_validate(function)

    def test_empty_parameter_name_is_rejected(self) -> None:
        function = self.function()
        function["parameters"] = {"  ": {"type": "string"}}
        with self.assertRaises(ValidationError):
            Function.model_validate(function)

    def test_duplicate_function_names_are_rejected(self) -> None:
        self.save([self.function(), self.function()])
        with self.assertRaisesRegex(ValueError, "Duplicate function"):
            JsonParser().load_functions(self.path)

    def test_empty_function_list_is_rejected(self) -> None:
        self.save([])
        with self.assertRaisesRegex(ValueError, "No functions"):
            JsonParser().load_functions(self.path)

    def test_invalid_json_is_reported(self) -> None:
        self.path.write_text("[", encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            JsonParser().load_functions(self.path)

    def test_runtime_errors_return_failure_exit_code(self) -> None:
        with patch("src.__main__.Application", side_effect=OSError("denied")):
            with patch("sys.argv", ["src"]):
                self.assertEqual(main(), 1)


if __name__ == "__main__":
    unittest.main()
