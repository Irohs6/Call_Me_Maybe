"""Generate numeric function arguments with constrained decoding."""

import re

from .argument_generator import ArgumentGenerator
from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt


NumberTokens = dict[int, tuple[str, list[int]]]


class NumberArgumentGenerator(ArgumentGenerator):
    """Choose numeric arguments from numbers present in the prompt."""

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate numeric values for the requested parameter names."""
        if not parameters:
            return {}

        number_tokens = self._number_tokens_available(
            prompt.prompt
        )

        if len(number_tokens) < len(parameters):
            raise ValueError(
                "Not enough numeric values found in prompt"
            )

        context_builder = ContextBuilder(
            functions=[function]
        )
        generated_values: dict[str, object] = {}

        for parameter in parameters:
            current_values = {
                **values,
                **generated_values,
            }
            parameter_context = (
                context_builder.build_parameter_context(
                    prompt,
                    function,
                    parameter,
                    current_values,
                )
            )
            context_ids = self.decoder.encode(
                parameter_context
            )
            candidate_tokens = {
                index: tokens
                for index, (_, tokens) in number_tokens.items()
            }
            selected_index = self.decoder.select_candidate(
                context_ids,
                candidate_tokens,
            )
            number, _ = number_tokens.pop(selected_index)
            generated_values[parameter] = float(number)

        return generated_values

    @staticmethod
    def _extract_numbers(prompt: str) -> list[str]:
        """Extract signed integer and decimal literals from a prompt."""
        return re.findall(r"-?\d+(?:\.\d+)?", prompt)

    def _number_tokens_available(
        self,
        prompt: str,
    ) -> NumberTokens:
        """Tokenize each numeric occurrence while preserving duplicates."""
        number_tokens: NumberTokens = {}

        for index, number in enumerate(
            self._extract_numbers(prompt)
        ):
            tokens = self.decoder.encode(number + "\n")
            number_tokens[index] = (number, tokens)

        return number_tokens
