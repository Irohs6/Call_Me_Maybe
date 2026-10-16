from pathlib import Path

from pydantic import BaseModel

from .models import Function, Prompt
from .parser import JsonParser


class Application(BaseModel):
    functions_path: Path
    input_path: Path
    output_path: Path

    def run(self) -> None:
        parser = JsonParser()

        functions = parser.load_functions(
            self.functions_path
        )

        prompts = parser.load_prompts(
            self.input_path
        )

        self._display_loaded_data(
            functions,
            prompts,
        )

    def _display_loaded_data(
        self,
        functions: list[Function],
        prompts: list[Prompt],
    ) -> None:
        print(f"Functions: {len(functions)}")
        print(f"Prompts: {len(prompts)}")
        for function in functions:
            print(function.model_dump_json(indent=2))
        for prompt in prompts:
            print(prompt.model_dump_json(indent=2))
