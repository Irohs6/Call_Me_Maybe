from pydantic import BaseModel

from .models import Function, Prompt


class ContextBuilder(BaseModel):
    functions: list[Function]

    def build(self) -> str:
        context: str = ""
        for function in self.functions:
            context += (
                f"Function: {function.name}\n"
                f"Description: {function.description}\n\n"
            )
        return context

    def build_with_prompt(self, prompt: Prompt) -> str:
        context: str = ""

        context += "Available functions:\n\n"

        for function in self.functions:
            context += f"Name: {function.name}\n"
            context += f"Description: {function.description}\n"
            context += "Parameters:\n"

            for name, parameter in function.parameters.items():
                context += f"- {name}: {parameter.type}\n"

            context += f"Returns: {function.returns.type}\n\n"

        context += f"User request: {prompt.prompt}\n\n"

        context += (
            "Choose the single function that should be called to satisfy "
            "the user request.\n"
            "Choose based on the requested operation and the function "
            "description, not only on words or values appearing in the request.\n"
            "Do not execute the function.\n"
            "Do not generate parameter values.\n"
            "Output only the exact function name.\n"
            "Function:"
        )

        return context

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

        context += "Function parameters:\n"

        parameters = list(function.parameters)

        for name in parameters:
            context += (
                f"- {name}: {function.parameters[name].type}\n"
            )

        position = parameters.index(parameter) + 1

        context += (
            f"\nCurrent parameter: {parameter}\n"
            f"Parameter position: {position} of {len(parameters)}\n"
            f"Type: {function.parameters[parameter].type}\n"
            f"Extract the value of parameter '{parameter}' "
            f"from the original prompt.\n"
            "Value:\n"
        )

        return context

    def build_string_parameter_context(
        self,
        prompt: Prompt,
        function: Function,
        parameter: str,
        values: dict[str, object],
    ) -> str:
        context = ""

        context += f"User prompt: {prompt.prompt}\n"
        context += f"Selected function: {function.name}\n"
        context += f"Function description: {function.description}\n"

        context += "Function parameters:\n"
        for name, parameter_type in function.parameters.items():
            context += f"- {name}: {parameter_type.type}\n"

        if values:
            context += "Parameters already extracted:\n"
            for name, value in values.items():
                context += f"- {name}: {value}\n"

        context += (
            f"\nCurrent parameter to extract: {parameter}\n"
            "You are extracting an INPUT argument for the function.\n"
            "Do NOT execute the function.\n"
            "Do NOT return the result of the function.\n"
            "Do NOT transform the input value.\n"
            "Return only the value that must be passed to this parameter.\n"
            "Do not explain.\n"
            "Do not add a label.\n"
            "End after the value with a newline.\n"
            "Value:"
        )

        return context
