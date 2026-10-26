"""Generate typed arguments for a selected function."""

import json
import re
from abc import ABC, abstractmethod
from typing import TypeAlias

from pydantic import BaseModel, ConfigDict

from .constrained_decoder import ConstrainedDecoder
from .context_builder import ContextBuilder
from .models import Function, Prompt
from .regex_reviewer import RegexArgumentReviewer
from .string_grammar import StringArgumentsGrammar


NumberTokens = dict[int, tuple[str, list[int]]]


class ArgumentGenerator(BaseModel, ABC):
    """Define the common interface for typed argument generators."""

    model_config = ConfigDict(extra="forbid")

    decoder: ConstrainedDecoder

    @abstractmethod
    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate the requested subset of function arguments."""
        raise NotImplementedError


class NumberArgumentGenerator(ArgumentGenerator):
    """Choose numeric arguments from numbers present in the prompt."""

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate numeric values for the requested parameters."""
        if not parameters:
            return {}

        number_tokens = self._number_tokens_available(
            prompt.prompt
        )
        if len(number_tokens) < len(parameters):
            raise ValueError(
                "Not enough numeric values found in prompt"
            )

        builder = ContextBuilder(
            functions=[function]
        )
        generated_values: dict[str, object] = {}

        for parameter in parameters:
            current_values = {
                **values,
                **generated_values,
            }
            context = builder.build_argument_context(
                prompt=prompt,
                function=function,
                parameters=[parameter],
                values=current_values,
            )
            context += (
                "{"
                f"{json.dumps(parameter)}: "
            )
            context_ids = self.decoder.encode(context)

            candidate_tokens = {
                index: tokens
                for index, (_, tokens)
                in number_tokens.items()
            }
            selected_index = self.decoder.select_candidate(
                context_ids,
                candidate_tokens,
            )
            number, _ = number_tokens.pop(
                selected_index
            )
            generated_values[parameter] = float(number)

        return generated_values

    @staticmethod
    def _extract_numbers(prompt: str) -> list[str]:
        """Extract signed integer and decimal literals."""
        return re.findall(
            r"-?\d+(?:\.\d+)?",
            prompt,
        )

    def _number_tokens_available(
        self,
        prompt: str,
    ) -> NumberTokens:
        """Tokenize numeric occurrences while preserving duplicates."""
        number_tokens: NumberTokens = {}

        for index, number in enumerate(
            self._extract_numbers(prompt)
        ):
            number_tokens[index] = (
                number,
                self.decoder.encode(number + "\n"),
            )

        return number_tokens


class BooleanArgumentGenerator(ArgumentGenerator):
    """Generate JSON boolean arguments as true or false."""

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate boolean values for the requested parameters."""
        if not parameters:
            return {}

        builder = ContextBuilder(
            functions=[function]
        )
        candidates = {
            "true": self.decoder.encode("true\n"),
            "false": self.decoder.encode("false\n"),
        }
        generated_values: dict[str, object] = {}

        for parameter in parameters:
            current_values = {
                **values,
                **generated_values,
            }
            context = builder.build_argument_context(
                prompt=prompt,
                function=function,
                parameters=[parameter],
                values=current_values,
            )
            context += (
                "{"
                f"{json.dumps(parameter)}: "
            )
            context_ids = self.decoder.encode(context)

            selected = self.decoder.select_candidate(
                context_ids,
                candidates,
            )
            generated_values[parameter] = (
                selected == "true"
            )

        return generated_values


class StringArgumentGenerator(ArgumentGenerator):
    """Generate string arguments through one JSON grammar."""

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate all requested string parameters as one object."""
        if not parameters:
            return {}

        builder = ContextBuilder(
            functions=[function]
        )
        context = builder.build_argument_context(
            prompt=prompt,
            function=function,
            parameters=parameters,
            values=values,
        )
        context += "{"
        context_ids = self.decoder.encode(context)

        grammar = StringArgumentsGrammar(
            parameters=parameters
        )
        generated_ids: list[int] = []
        progress = "{"

        for _ in range(512):
            logits = self.decoder.logits(
                context_ids + generated_ids
            )

            def valid_token(token_id: int) -> bool:
                """Check whether a token keeps the prefix valid."""
                candidate = (
                    "{"
                    + self.decoder.decode(
                        generated_ids + [token_id]
                    )
                )
                return (
                    candidate != progress
                    and grammar.status(candidate)
                    != "invalid"
                )

            next_token_id = (
                self.decoder.highest_valid_token(
                    logits,
                    valid_token,
                )
            )
            generated_ids.append(next_token_id)
            progress = (
                "{"
                + self.decoder.decode(generated_ids)
            )

            if grammar.status(progress) != "complete":
                continue

            arguments, _ = (
                json.JSONDecoder().raw_decode(progress)
            )
            if not isinstance(arguments, dict):
                raise ValueError(
                    "Generated string arguments are not "
                    "a JSON object"
                )

            return {
                parameter: arguments[parameter]
                for parameter in parameters
            }

        raise ValueError(
            "String arguments exceeded maximum token count"
        )


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
        """Generate arguments by delegating each type."""
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
            values.update(
                generator.generate(
                    prompt=prompt,
                    function=function,
                    parameters=parameters,
                    values=values,
                )
            )

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
