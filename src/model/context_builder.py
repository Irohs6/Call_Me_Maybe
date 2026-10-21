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
        context: str = self.build()
        context += f"Prompt: {prompt.prompt}\n\n"
        context += "Select the function that best matches the prompt.\n"
        context += "Function:"

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
