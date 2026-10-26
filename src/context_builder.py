"""Build prompts used for selection and argument generation."""

import json

from pydantic import BaseModel, ConfigDict

from .models import Function, Prompt


class ContextBuilder(BaseModel):
    """Build LLM contexts without executing any declared function."""

    model_config = ConfigDict(extra="forbid")

    functions: list[Function]

    @staticmethod
    def tool_definition(
        function: Function,
        name: str | None = None,
        description: str | None = None,
        parameters: list[str] | None = None,
    ) -> dict[str, object]:
        """Build a JSON-schema-style tool definition."""
        parameter_names = (
            list(function.parameters)
            if parameters is None
            else parameters
        )
        properties = {
            parameter: {
                "type": function.parameters[parameter].type
            }
            for parameter in parameter_names
        }

        return {
            "type": "function",
            "function": {
                "name": (
                    function.name
                    if name is None
                    else name
                ),
                "description": (
                    function.description
                    if description is None
                    else description
                ),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": parameter_names,
                },
            },
        }

    def build_selection_context(
        self,
        prompt: Prompt,
        comparison: bool = False,
    ) -> str:
        """Build the Qwen/Hermes context used to select a function."""
        if comparison and len(self.functions) != 2:
            raise ValueError(
                "A comparison requires exactly two functions"
            )

        tools: list[dict[str, object]] = []

        for index, function in enumerate(
            self.functions
        ):
            name = (
                chr(ord("A") + index)
                if comparison
                else function.name
            )
            description = function.description

            if comparison:
                description += (
                    " Actual function name: "
                    f"{function.name}."
                )

            tools.append(
                self.tool_definition(
                    function,
                    name,
                    description,
                )
            )

        tool_definitions = "\n".join(
            json.dumps(
                tool,
                ensure_ascii=False,
            )
            for tool in tools
        )

        return (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Choose the single "
            "function that matches the operation requested by the user. "
            "Base your choice on each function's description and parameter "
            "schema, rather than isolated words or values in the request. "
            "Do not execute the function.\n\n"
            "# Tools\n\n"
            "Available function definitions:\n<tools>\n"
            f"{tool_definitions}\n"
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
        """Build one shared context for typed argument generation."""
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
            "Use the JSON types required by the schema. "
            "Do not execute the function or return its result.\n"
            "Copy literal text without changing its spelling, case, "
            "or whitespace. Translate symbol names into literal "
            "characters when appropriate. For a requested class of text, "
            "use a pattern matching that class rather than only one "
            "example.\n"
            "Arguments already generated: "
            f"{previous_values}\n\n"
            "# Tools\n\n"
            "Available function definitions:\n<tools>\n"
            f"{json.dumps(tool, ensure_ascii=False)}\n"
            "</tools>\n\n"
            "Write exactly one JSON function call between <tool_call> "
            "and </tool_call>, with the keys name and arguments.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{prompt.prompt}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
            "<tool_call>\n"
            '{"name": '
            f'{json.dumps(function.name)}, '
            '"arguments": '
        )
