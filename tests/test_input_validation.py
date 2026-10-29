"""Validate schema inputs and graceful CLI failures."""

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
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

    def test_duplicate_parameter_json_keys_are_rejected(self) -> None:
        self.path.write_text(
            '[{"name":"fn_add_numbers","description":"Add numbers",'
            '"parameters":{"b":{"type":"number"},'
            '"b":{"type":"number"}},'
            '"returns":{"type":"number"}}]',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key: 'b'"):
            JsonParser().load_functions(self.path)

    def test_duplicate_escaped_json_keys_are_rejected(self) -> None:
        path = Path(self.tempdir.name) / "prompts.json"
        path.write_text(
            r'[{"prompt":"first","\u0070rompt":"second"}]',
            encoding="utf-8",
        )
        with self.assertRaisesRegex(
            ValueError, "Duplicate JSON key: 'prompt'"
        ):
            JsonParser().load_prompts(path)

    def test_same_json_key_in_different_objects_is_allowed(self) -> None:
        path = Path(self.tempdir.name) / "prompts.json"
        path.write_text(
            '[{"prompt":"first"},{"prompt":"second"}]',
            encoding="utf-8",
        )
        prompts = JsonParser().load_prompts(path)
        self.assertEqual([item.prompt for item in prompts], ["first", "second"])

    def test_empty_function_list_is_rejected(self) -> None:
        self.save([])
        with self.assertRaisesRegex(ValueError, "No functions"):
            JsonParser().load_functions(self.path)

    def test_invalid_json_is_reported(self) -> None:
        self.path.write_text("[", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "line 1, column 2"):
            JsonParser().load_functions(self.path)

    def test_runtime_errors_return_failure_exit_code(self) -> None:
        output = io.StringIO()
        with patch("src.model.app.Application", side_effect=OSError("denied")):
            with patch("sys.argv", ["src"]):
                with redirect_stderr(output):
                    self.assertEqual(main(), 1)
        self.assertEqual(output.getvalue().strip(), "Error: denied")

    def test_validation_error_is_concise(self) -> None:
        function = self.function()
        function["parameters"] = {"": {"type": "string"}}
        self.save([function])
        with self.assertRaises(ValueError) as error:
            JsonParser().load_functions(self.path)
        self.assertEqual(
            str(error.exception),
            "Invalid function #1: parameters: Parameter name cannot be empty",
        )

    def test_invalid_prompt_identifies_position(self) -> None:
        path = Path(self.tempdir.name) / "prompts.json"
        path.write_text(
            json.dumps([{"prompt": "ok"}, {"prompt": ""}]),
            encoding="utf-8",
        )
        with self.assertRaises(ValueError) as error:
            JsonParser().load_prompts(path)
        self.assertIn("Invalid prompt #2: prompt:", str(error.exception))
        self.assertNotIn("errors.pydantic.dev", str(error.exception))

    def test_unreadable_file_has_context(self) -> None:
        with patch.object(Path, "open", side_effect=PermissionError("denied")):
            with self.assertRaisesRegex(OSError, "Cannot read file"):
                JsonParser().load_functions(self.path)

    def test_missing_dependency_is_reported(self) -> None:
        output = io.StringIO()
        with patch.dict(sys.modules, {"llm_sdk": None}):
            with patch("sys.argv", ["src"]):
                with redirect_stderr(output):
                    self.assertEqual(main(), 1)
        self.assertIn("Missing Python module 'llm_sdk'", output.getvalue())
        self.assertNotIn("Traceback", output.getvalue())


if __name__ == "__main__":
    unittest.main()
