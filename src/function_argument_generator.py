from llm_sdk import Small_LLM_Model

from .argument_generator import ArgumentGenerator
from .function_selector import FunctionSelector
from .model.models import Function, Prompt
from .model.regex_review import RegexInterpretations
from .number_argument_generator import NumberArgumentGenerator
from .string_argument_generator import StringArgumentGenerator


class FunctionArgumentGenerator:

    def __init__(
        self,
        model: Small_LLM_Model,
    ) -> None:
        self.model = model

        self.generators: dict[str, ArgumentGenerator] = {
            "number": NumberArgumentGenerator(model),
            "string": StringArgumentGenerator(model),
        }

    def generate(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:

        values: dict[str, object] = {}

        for type_name, generator in self.generators.items():
            parameters = [
                name
                for name, definition in function.parameters.items()
                if definition.type == type_name
            ]

            if not parameters:
                continue

            generated_values = generator.generate(
                prompt=prompt,
                function=function,
                parameters=parameters,
                values=values,
            )

            values.update(generated_values)

        return self._refine_regex_arguments(
            prompt,
            values,
        )

    def _refine_regex_arguments(
        self,
        prompt: Prompt,
        arguments: dict[str, object],
    ) -> dict[str, object]:

        if set(arguments) != {
            "source_string",
            "regex",
            "replacement",
        }:
            return arguments

        pattern = arguments["regex"]

        if not isinstance(pattern, str):
            return arguments

        candidates = RegexInterpretations(
            pattern=pattern
        ).candidates()

        if len(candidates) < 2:
            return arguments

        functions = [
            function
            for _, function in candidates
        ]

        selector = FunctionSelector(
            self.model,
            functions,
        )

        selected = selector.select(
            prompt
        )

        for candidate, function in candidates:
            if function.name == selected.name:
                return {
                    **arguments,
                    "regex": candidate,
                }

        return arguments
