from pathlib import Path

from pydantic import BaseModel, Field

from .models import Function, Prompt
from .parser import JsonParser


class Application(BaseModel):
    functions_path: Path
    input_path: Path
    output_path: Path
    functions: list[Function] = Field(default_factory=list)
    prompts: list[Prompt] = Field(default_factory=list)

    def run(self) -> None:
        parser = JsonParser()

        self.functions = parser.load_functions(
            self.functions_path
        )

        self.prompts = parser.load_prompts(
            self.input_path
        )

        self._display_loaded_data(
            self.functions,
            self.prompts,
        )

    def _display_loaded_data(
        self,
        functions: list[Function],
        prompts: list[Prompt],
    ) -> None:
        print(f"Functions: {len(functions)}")
        print(f"Prompts: {len(prompts)}")
