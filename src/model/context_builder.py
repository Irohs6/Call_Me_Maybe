import json

from pydantic import BaseModel

from .models import Function, Prompt


class ContextBuilder(BaseModel):
    """Build selection and argument prompts without executing functions."""

    functions: list[Function]

    @staticmethod
    def tool_definition(
        function: Function,
        name: str | None = None,
        description: str | None = None,
    ) -> dict[str, object]:
        """Build the shared JSON Schema representation of a function."""
        return {
            "type": "function",
            "function": {
                "name": function.name if name is None else name,
                "description": (
                    function.description if description is None else description
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        key: {"type": definition.type}
                        for key, definition in function.parameters.items()
                    },
                    "required": list(function.parameters),
                },
            },
        }

    def build_with_prompt(
        self, prompt: Prompt, comparison: bool = False
    ) -> str:
        if comparison and len(self.functions) != 2:
            raise ValueError("A comparison requires exactly two functions")

        tools = []
        for index, function in enumerate(self.functions):
            name = chr(ord("A") + index) if comparison else function.name
            description = function.description
            if comparison:
                description += f" Actual function name: {function.name}."
            tools.append(self.tool_definition(function, name, description))

        tool_definitions = "\n".join(
            json.dumps(tool, ensure_ascii=False) for tool in tools
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

    @staticmethod
    def extract_user_request(context: str) -> str:
        _, user_marker, remaining = context.partition("<|im_start|>user\n")
        request, assistant_marker, _ = remaining.rpartition(
            "<|im_end|>\n<|im_start|>assistant\n"
        )
        if not user_marker or not assistant_marker:
            raise ValueError("Selection context must contain a user message")
        return request

    def build_parameter_context(
        self,
        prompt: Prompt,
        function: Function,
        parameter: str,
        values: dict[str, object],
    ) -> str:
        context = ""

        context += f"Prompt: {prompt.prompt}\n"
        context += f"Function: {function.name}\n"
        context += f"Description: {function.description}\n"

        context += "Function signature: "
        context += f"{function.name}("

        parameters = []

        for name, definition in function.parameters.items():
            parameters.append(f"{name}: {definition.type}")

        context += ", ".join(parameters)
        context += ")\n"

        context += "Arguments already generated:\n"

        for name, value in values.items():
            context += f"{name}={value}\n"

        context += f"{parameter}="

        return context

    def build_string_parameter_context(
        self,
        prompt: Prompt,
        function: Function,
    ) -> str:
        tool = self.tool_definition(function)

        return (
            "<|im_start|>system\n"
            "You are a function-calling assistant. Supply the arguments "
            "needed to fulfill the user's request. Do not execute the "
            "function or return its result.\n"
            "Copy literal text without changing its spelling, case, "
            "or whitespace.\n"
            "Supply concrete argument values, translating symbol names "
            "into literal characters. For a requested class of text, "
            "use a pattern matching that class rather than only one "
            "example.\n\n"
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
        )
