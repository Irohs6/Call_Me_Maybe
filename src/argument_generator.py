"""Generate typed arguments for the selected function."""

import json
import re

from llm_sdk import Small_LLM_Model

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt


class ArgumentGenerator:
    """Generate every argument in a single JSON sequence."""

    def __init__(
        self,
        model: Small_LLM_Model,
    ) -> None:
        self.model = model

    @staticmethod
    def create_argument_context(
        prompt: Prompt,
        function: Function,
    ) -> str:
        """Prepare one context containing every function parameter."""
        return ContextBuilder(
            functions=[function]
        ).build_argument_context(
            prompt,
            function,
        )

    @staticmethod
    def validate_arguments(
        arguments: object,
        function: Function,
    ) -> dict[str, object]:
        """Verify the generated object matches the function schema."""
        if not isinstance(arguments, dict):
            raise ValueError("Generated arguments must be a JSON object")

        if set(arguments) != set(function.parameters):
            raise ValueError(
                "Generated argument names do not match the function schema"
            )

        validated: dict[str, object] = {}

        for name, definition in function.parameters.items():
            value = arguments[name]
            kind = definition.type

            if kind == "string":
                valid = isinstance(value, str)
            elif kind == "number":
                valid = isinstance(value, (int, float)) and not isinstance(
                    value, bool
                )
                if valid:
                    value = float(value)
            elif kind == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif kind == "boolean":
                valid = isinstance(value, bool)
            else:
                raise ValueError(f"Unsupported argument type: {kind}")

            if not valid:
                raise ValueError(
                    f"Invalid value for '{name}': expected {kind}"
                )

            validated[name] = value

        return validated

    @staticmethod
    def is_valid_number_prefix(value: str) -> bool:
        """Accept a complete number or a number being generated."""
        return re.fullmatch(
            r"(?:|-|-?(?:0|[1-9][0-9]*)(?:\.[0-9]*)?)",
            value,
        ) is not None

    def generate(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:
        """Generate and validate one complete JSON argument object."""
        if not function.parameters:
            return {}

        context = self.create_argument_context(prompt, function)
        context_ids = self.model.encode(context)[0].tolist()
        generated_ids: list[int] = []
        decoder = json.JSONDecoder()
        for _ in range(256):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = max(
                range(len(logits)),
                key=lambda token_id: logits[token_id],
            )
            generated_ids.append(next_token_id)

            candidate = "{" + self.model.decode(generated_ids)

            try:
                arguments, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue

            return self.validate_arguments(arguments, function)

        raise ValueError(
            "Argument generation exceeded the 256-token limit"
        )
