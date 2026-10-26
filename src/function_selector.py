"""Select functions and extract arguments with constrained decoding."""

import re

from llm_sdk import Small_LLM_Model

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt


class FunctionSelector:
    """Select a function and generate its argument values."""

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[Function],
    ) -> None:
        self.model = model
        self.functions = functions

    def allowed_ids_for_position(
        self,
        function_tokens: dict[object, list[int]],
        position: int,
    ) -> list[int]:
        """Return token IDs allowed at the current candidate position."""
        allowed_ids: list[int] = []

        for tokens in function_tokens.values():
            if len(tokens) <= position:
                continue

            token_id = tokens[position]
            if token_id not in allowed_ids:
                allowed_ids.append(token_id)

        return allowed_ids

    def token_function_available(
        self,
    ) -> dict[str, list[int]]:
        """Encode the available function names for constrained selection."""
        return {
            function.name: self.model.encode(
                function.name + '"'
            )[0].tolist()
            for function in self.functions
        }

    def select(
        self,
        context: str,
    ) -> Function:
        """Choose the function with the highest allowed token logits."""
        context_ids = self.model.encode(context)[0].tolist()
        function_tokens = self.token_function_available()
        generated_ids: list[int] = []

        while function_tokens:
            position = len(generated_ids)
            allowed_ids = self.allowed_ids_for_position(
                function_tokens,
                position,
            )
            if not allowed_ids:
                raise ValueError(
                    "No function can continue constrained decoding"
                )

            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )
            generated_ids.append(next_token_id)

            function_tokens = {
                name: tokens
                for name, tokens in function_tokens.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            for function in self.functions:
                tokens = function_tokens.get(function.name)
                if tokens == generated_ids:
                    return function

        raise ValueError("No function selected")

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
                for index, (number, tokens)
                in candidates.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            for key, (number, tokens) in candidates.items():
                if tokens == generated_ids:
                    number_tokens.pop(key)
                    return float(number)

        raise ValueError("No number selected")

    def generate_free_string(
        self,
        context_ids: list[int],
    ) -> str:
        """Generate a string until the JSON closing quote."""
        generated_ids: list[int] = []

        while True:
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = max(
                range(len(logits)),
                key=lambda token_id: logits[token_id],
            )
            generated_ids.append(next_token_id)

            progress = self.model.decode(generated_ids)
            if '"' in progress:
                return progress.split('"', 1)[0]
            if "\n" in progress:
                return progress.split("\n", 1)[0]

    def final_selection(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:
        """Generate every argument of the selected function."""
        context_builder = ContextBuilder(
            functions=[function]
        )
        values: dict[str, object] = {}
        number_tokens = self.number_tokens_available(
            prompt.prompt
        )

        for parameter, definition in function.parameters.items():
            context = context_builder.build_parameter_context(
                prompt,
                function,
                parameter,
                values,
            )
            context_ids = self.model.encode(
                context
            )[0].tolist()

            if definition.type == "number":
                value = self.generate_number(
                    context_ids,
                    number_tokens,
                )
            elif definition.type == "string":
                value = self.generate_free_string(
                    context_ids
                )
            else:
                raise ValueError(
                    "Unsupported parameter type: "
                    f"{definition.type}"
                )

            values[parameter] = value
            print(
                f"Generated {definition.type} for "
                f"parameter '{parameter}': {value}"
            )

        return values
