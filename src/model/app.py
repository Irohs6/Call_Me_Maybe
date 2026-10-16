from pathlib import Path

from pydantic import BaseModel

from .models import FunctionDefinition, PromptDefinition
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
        functions: list[FunctionDefinition],
        prompts: list[PromptDefinition],
    ) -> None:
        print(f"Functions: {len(functions)}")
        print(f"Prompts: {len(prompts)}")