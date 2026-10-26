"""Orchestrate typed generation for one selected function."""

from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from .argument_generator import ArgumentGenerator
from .boolean_argument_generator import BooleanArgumentGenerator
from .constrained_decoding import ConstrainedDecoder
from .model.models import Function, Prompt
from .number_argument_generator import NumberArgumentGenerator
from .regex_argument_reviewer import RegexArgumentReviewer
from .string_argument_generator import StringArgumentGenerator


GeneratorClass: TypeAlias = type[ArgumentGenerator]

GENERATOR_TYPES: dict[str, GeneratorClass] = {
    "number": NumberArgumentGenerator,
    "string": StringArgumentGenerator,
    "boolean": BooleanArgumentGenerator,
}


class FunctionArgumentGenerator(BaseModel):
    """Generate every argument required by a selected function."""

    model_config = ConfigDict(extra="forbid")

    decoder: ConstrainedDecoder

    def generate(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:
        """Generate arguments by delegating each type to its strategy."""
        values: dict[str, object] = {}

        parameter_groups: dict[str, list[str]] = {}
        for name, definition in function.parameters.items():
            parameter_groups.setdefault(
                definition.type,
                [],
            ).append(name)

        for type_name, parameters in parameter_groups.items():
            generator_class = GENERATOR_TYPES.get(
                type_name
            )
            if generator_class is None:
                raise ValueError(
                    "Unsupported parameter type: "
                    f"{type_name}"
                )

            generator = generator_class(
                decoder=self.decoder
            )
            generated_values = generator.generate(
                prompt=prompt,
                function=function,
                parameters=parameters,
                values=values,
            )
            values.update(generated_values)

        missing = [
            name
            for name in function.parameters
            if name not in values
        ]
        if missing:
            raise ValueError(
                "Missing generated arguments: "
                + ", ".join(missing)
            )

        ordered_values = {
            name: values[name]
            for name in function.parameters
        }

        reviewer = RegexArgumentReviewer(
            decoder=self.decoder
        )
        return reviewer.review(
            prompt,
            ordered_values,
        )
