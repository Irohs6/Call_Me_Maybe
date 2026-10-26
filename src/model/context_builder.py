"""Build Qwen/Hermes contexts for function calling."""

import json

from pydantic import BaseModel

from .models import Function, Prompt


class ContextBuilder(BaseModel):
    """Build contexts for function selection and argument extraction."""

    functions: list[Function]

    @staticmethod
    def tool_definition(
        function: Function,
        parameters: list[str] | None = None,
    ) -> dict[str, object]:
        """Build one tool definition from a function model."""
        parameter_names = (
            list(function.parameters)
            if parameters is None
            else parameters
        )
        properties = {
            name: {
                "type": function.parameters[name].type
            }
            for name in parameter_names
        }

        return {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": parameter_names,
                },
            },
        }

    def build_with_prompt(
        self,
        prompt: Prompt,
    ) -> str:
        """Build context used to choose one function."""
        tools = "\n".join(
            json.dumps(
                self.tool_definition(function),
                ensure_ascii=False,
            )
            for function in self.functions
        )

        return (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Choose the single "
            "function that matches the operation requested by the user. "
            "Base your choice on each function's description and parameter "
            "schema. Do not execute the function.\n\n"
            "# Tools\n\n"
            "Available function definitions:\n<tools>\n"
            f"{tools}\n"
            "</tools>\n\n"
            "Write one JSON function call inside <tool_call> and "
            "</tool_call>, with the keys name and arguments.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            "<tool_call>\n"
            '{"name": "'
        )

    def build_argument_context(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> str:
        """Build the shared context used for argument generation."""
        tool = self.tool_definition(
            function,
            parameters=parameters,
        )
        previous_values = json.dumps(
            values,
            ensure_ascii=False,
        )

        return (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Supply only the "
            "arguments described in the provided tool schema. "
            "Do not execute the function or return its result.\n"
            "Copy literal text without changing its spelling, case, "
            "or whitespace. Translate symbol names into literal "
            "characters when appropriate. For a requested class of text, "
            "use a pattern matching that class rather than one example.\n"
            "Arguments already generated: "
            f"{previous_values}\n\n"
            "# Tools\n\n"
            "Available function definitions:\n<tools>\n"
            f"{json.dumps(tool, ensure_ascii=False)}\n"
            "</tools>\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            "<tool_call>\n"
            '{"name": '
            f'{json.dumps(function.name)}, '
            '"arguments": {'
        )
