import json
from pathlib import Path

from pydantic import BaseModel, ValidationError


class TypeDefinition(BaseModel):
    type: str


class FunctionDefinition(BaseModel):
    name: str
    description: str
    parameters: dict[str, TypeDefinition]
    returns: TypeDefinition


class PromptDefinition(BaseModel):
    prompt: str


class JsonParser(BaseModel):

    @staticmethod
    def _load(path: Path) -> object:
        try:
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)

        except FileNotFoundError as error:
            raise ValueError(
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

    def load_functions(
            self,
            path: Path) -> list[FunctionDefinition]:
        data = self._load(path)

        if not isinstance(data, list):
            raise ValueError(
                "Functions definition must be a JSON array."
            )

        try:
            return [
                FunctionDefinition.model_validate(item)
                for item in data
            ]

        except ValidationError as error:
            raise ValueError(
                "Invalid function definition."
            ) from error

    def load_prompts(
        self,
        path: Path,
    ) -> list[PromptDefinition]:
        data = self._load(path)

        if not isinstance(data, list):
            raise ValueError(
                "Input must be a JSON array."
            )

        try:
            return [
                PromptDefinition.model_validate(item)
                for item in data
            ]

        except ValidationError as error:
            raise ValueError(
                "Invalid prompt definition."
            ) from error
