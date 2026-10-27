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
    ) -> str:
        """Build one context for the complete argument object."""
        tool = self.tool_definition(function)

        return (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Supply every "
            "argument described in the provided tool schema in one "
            "JSON object. Do not execute the function or return its result.\n"
            "Copy literal text without changing its spelling, case, "
            "or whitespace. Translate symbol names into literal "
            "characters when appropriate. For a requested class of text, "
            "use a pattern matching that class rather than one example.\n\n"
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
