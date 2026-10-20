import json
from pathlib import Path
from json.decoder import JSONDecodeError

from pydantic import BaseModel, ValidationError

from .models import Function, Prompt


class JsonParser(BaseModel):

    @staticmethod
    def _load(path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)

        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"File not found: {path}"
            ) from error

        except json.JSONDecodeError as error:
            raise JSONDecodeError(
                f"Invalid JSON: {path}",
                doc=error.doc,
                pos=error.pos
            ) from error

        except UnicodeDecodeError as error:
            raise ValueError(
                f"Invalid UTF-8 file: {path}"
            ) from error

    def load_functions(
            self,
            path: Path) -> list[Function]:
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
