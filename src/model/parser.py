"""Load and validate JSON input files."""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from .models import Function, Prompt


class JsonParser(BaseModel):
    @staticmethod
    def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
        """Reject repeated keys before JSON objects become dictionaries."""
        values: dict[str, object] = {}
        for key, value in pairs:
            if key in values:
                raise ValueError(f"Duplicate JSON key: '{key}'")
            values[key] = value
        return values

    @staticmethod
    def _validation_message(error: ValidationError) -> str:
        """Keep the first useful Pydantic error without its debug metadata."""
        issue = error.errors()[0]
        field = ".".join(str(part) for part in issue["loc"])
        message = issue["msg"].removeprefix("Value error, ")
        return f"{field}: {message}" if field else message

    @staticmethod
    def _load(path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(
                    file, object_pairs_hook=JsonParser._unique_keys
                )
        except FileNotFoundError as error:
            raise FileNotFoundError(f"File not found: {path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON in '{path}' "
                f"(line {error.lineno}, column {error.colno}): {error.msg}"
            ) from error
        except UnicodeDecodeError as error:
            raise ValueError(f"Invalid UTF-8 file: {path}") from error
        except OSError as error:
            raise OSError(f"Cannot read file '{path}': {error}") from error

    def load_functions(self, path: Path) -> list[Function]:
        """Load function definitions and check name uniqueness."""
        data = self._load(path)
        if not isinstance(data, list):
            raise ValueError("Functions definition must be a JSON array")

        functions: list[Function] = []
        names: set[str] = set()

        for i, item in enumerate(data, start=1):
            try:
                function = Function.model_validate(item)
            except ValidationError as error:
                raise ValueError(
                    f"Invalid function #{i}: "
                    f"{self._validation_message(error)}"
                ) from error

            if function.name in names:
                raise ValueError(
                    f"Duplicate function name: {function.name}"
                )
            names.add(function.name)
            functions.append(function)

        if not functions:
            raise ValueError("No functions defined")
        return functions

    def load_prompts(self, path: Path) -> list[Prompt]:
        """Load prompts and identify invalid entries by their position."""
        data = self._load(path)
        if not isinstance(data, list):
            raise ValueError("Input must be a JSON array")

        prompts: list[Prompt] = []
        for i, item in enumerate(data, start=1):
            try:
                prompts.append(Prompt.model_validate(item))
            except ValidationError as error:
                raise ValueError(
                    f"Invalid prompt #{i}: "
                    f"{self._validation_message(error)}"
                ) from error
        return prompts
