"""Application input-loading service."""

from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .models import Function, Prompt
from .parser import JsonParser


class Application(BaseModel):
    """Load validated functions and prompts from configured paths."""

    model_config = ConfigDict(extra="forbid")

    functions_path: Path
    input_path: Path

    def load(
        self,
    ) -> tuple[list[Function], list[Prompt]]:
        """Load and return the function catalogue and prompts."""
        parser = JsonParser()

        functions = parser.load_functions(
            self.functions_path
        )
        prompts = parser.load_prompts(
            self.input_path
        )

        print(f"Functions: {len(functions)}")
        print(f"Prompts: {len(prompts)}")

        return functions, prompts
