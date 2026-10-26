"""Load and validate JSON input files."""

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from .models import Function, Prompt


class JsonParser(BaseModel):
    """Parse input JSON files into validated Pydantic models."""

    model_config = ConfigDict(extra="forbid")

    @staticmethod
    def _load(path: Path) -> object:
        """Load one UTF-8 JSON file and normalize file errors."""
        try:
            with path.open(
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"File not found: {path}"
            ) from error
        except json.JSONDecodeError as error:
            raise ValueError(
                f"Invalid JSON: {path}"
            ) from error
        except UnicodeDecodeError as error:
            raise ValueError(
                f"Invalid UTF-8 file: {path}"
            ) from error
        except OSError as error:
            raise OSError(
                f"Unable to read file: {path}"
            ) from error

    def load_functions(
        self,
        path: Path,
    ) -> list[Function]:
        """Load and validate the function catalogue."""
        data = self._load(path)

        if not isinstance(data, list):
            raise ValueError(
                "Functions definition must be a JSON array."
            )

        try:
            return [
                Function.model_validate(item)
                for item in data
            ]
        except ValidationError as error:
            raise ValueError(
                "Invalid function definition."
            ) from error

    def load_prompts(
        self,
        path: Path,
    ) -> list[Prompt]:
        """Load and validate the prompt list."""
        data = self._load(path)

        if not isinstance(data, list):
            raise ValueError(
                "Input must be a JSON array."
            )

        try:
            return [
                Prompt.model_validate(item)
                for item in data
            ]
        except ValidationError as error:
            raise ValueError(
                "Invalid prompt definition."
            ) from error
