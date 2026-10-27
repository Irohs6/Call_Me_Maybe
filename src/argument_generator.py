"""Generate arguments for a selected function."""

import json
import re

from llm_sdk import Small_LLM_Model

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt


class ArgumentGenerator:
    """Generate number and string arguments for one function."""

    def __init__(
        self,
        model: Small_LLM_Model,
    ) -> None:
        self.model = model

    @staticmethod
    def allowed_ids_for_position(
        token_sequences: dict[int, list[int]],
        position: int,
    ) -> list[int]:
        """Return token IDs allowed at one candidate position."""
        allowed_ids: list[int] = []

        for tokens in token_sequences.values():
            if len(tokens) <= position:
                continue

            token_id = tokens[position]
            if token_id not in allowed_ids:
                allowed_ids.append(token_id)

        return allowed_ids

    @staticmethod
    def extract_numbers(
        prompt: str,
    ) -> list[str]:
        """Extract signed integer and decimal literals."""
        return re.findall(
            r"-?\d+(?:\.\d+)?",
            prompt,
        )

    def number_tokens_available(
        self,
        prompt: str,
    ) -> dict[int, tuple[str, list[int]]]:
        """Encode every numeric value found in the prompt."""
        return {
            index: (
                number,
                self.model.encode(number)[0].tolist(),
            )
            for index, number in enumerate(
                self.extract_numbers(prompt)
            )
        }

    def generate_number(
        self,
        context_ids: list[int],
        number_tokens: dict[int, tuple[str, list[int]]],
    ) -> float:
        """Choose one number from values present in the prompt."""
        candidates = number_tokens.copy()
        generated_ids: list[int] = []

        if not candidates:
            raise ValueError("No number found in prompt")

        while candidates:
            position = len(generated_ids)
            allowed_ids = self.allowed_ids_for_position(
                {
                    key: tokens
                    for key, (_, tokens) in candidates.items()
                },
                position,
            )

            if not allowed_ids:
                raise ValueError(
                    "No number can continue constrained decoding"
                )

            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )
            generated_ids.append(next_token_id)

            candidates = {
                index: (number, tokens)
                for index, (number, tokens) in candidates.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            for key, (number, tokens) in candidates.items():
                if tokens == generated_ids:
                    number_tokens.pop(key)
                    return float(number)

        raise ValueError("No number selected")

    def generate_strings(
        self,
        context_ids: list[int],
        parameters: list[str],
    ) -> dict[str, object]:
        """Generate all string arguments together as one JSON object."""
        generated_ids: list[int] = []

        for _ in range(256):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = max(
                range(len(logits)),
                key=lambda token_id: logits[token_id],
            )
            generated_ids.append(next_token_id)

            candidate = (
                "{"
                + self.model.decode(generated_ids)
            )

            try:
                arguments, _ = json.JSONDecoder().raw_decode(
                    candidate
                )
            except json.JSONDecodeError:
                continue

            if not isinstance(arguments, dict):
                raise ValueError(
                    "Generated arguments are not a JSON object"
                )
            if set(arguments) != set(parameters):
                raise ValueError(
                    "Generated string arguments do not match "
                    "the function parameters"
                )
            if not all(
                isinstance(arguments[name], str)
                for name in parameters
            ):
                raise ValueError(
                    "Generated string arguments must be strings"
                )

            return {
                name: arguments[name]
                for name in parameters
            }

        raise ValueError(
            "String arguments exceeded maximum token count"
        )

    def generate(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:
        """Generate all arguments required by the selected function."""
        context_builder = ContextBuilder(
            functions=[function]
        )
        values: dict[str, object] = {}
        number_tokens = self.number_tokens_available(
            prompt.prompt
        )
        string_parameters = [
            name
            for name, definition in function.parameters.items()
            if definition.type == "string"
        ]

        for parameter, definition in function.parameters.items():
            if definition.type != "number":
                continue

            context = context_builder.build_argument_context(
                prompt,
                function,
                [parameter],
                values,
            )
            context += f'{json.dumps(parameter)}: '
            context_ids = self.model.encode(
                context
            )[0].tolist()

            value = self.generate_number(
                context_ids,
                number_tokens,
            )
            values[parameter] = value
            print(
                f"Generated number for parameter "
                f"'{parameter}': {value}"
            )

        if string_parameters:
            context = context_builder.build_argument_context(
                prompt,
                function,
                string_parameters,
                values,
            )
            context_ids = self.model.encode(
                context
            )[0].tolist()

            string_values = self.generate_strings(
                context_ids,
                string_parameters,
            )
            values.update(string_values)

            for parameter in string_parameters:
                print(
                    f"Generated string for parameter "
                    f"'{parameter}': {values[parameter]}"
                )

        return values
