from pathlib import Path

from pydantic import BaseModel, Field

from .models import Function, Prompt
from .parser import JsonParser


class Application(BaseModel):
    """Load validated input files for the function-calling pipeline."""

    functions_path: Path
    input_path: Path
    output_path: Path
    functions: list[Function] = Field(default_factory=list)
    prompts: list[Prompt] = Field(default_factory=list)

    def run(self) -> None:
        """Load function definitions and natural-language prompts."""
        parser = JsonParser()

        self.functions = parser.load_functions(
            self.functions_path
        )

        self.prompts = parser.load_prompts(
            self.input_path
        )
