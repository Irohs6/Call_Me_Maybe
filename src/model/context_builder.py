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
    ) -> dict[str, object]:
        """Build one tool definition from a function model."""
        properties = {
            name: {"type": definition.type}
            for name, definition in function.parameters.items()
        }

        return {
            "type": "function",
            "function": {
                "name": function.name,
                "description": function.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": list(function.parameters),
                },
            },
        }

    def build_function_selection_context(
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
        parameter: str,
        values: dict[str, object],
    ) -> str:
        """Build a context to generate one parameter value."""
        tool = self.tool_definition(function)
        previous = "".join(
            f"{json.dumps(name)}: "
            f"{json.dumps(value, ensure_ascii=False)}, "
            for name, value in values.items()
        )

        return (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Fill the arguments "
            "of the provided function using the user's request. "
            "Do not execute the function or return its result. "
            "Copy literal text exactly, including spelling, case, "
            "and whitespace. For requested text classes, produce "
            "a regex matching the class, not a single example.\n"
            f"Continue with the JSON value for parameter {parameter}, "
            f"which must be of type {function.parameters[parameter].type}. "
            "Do not write explanations.\n\n"
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
            f'{previous}{json.dumps(parameter)}: '
        )
