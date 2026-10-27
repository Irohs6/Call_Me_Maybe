"""Generate typed arguments for the selected function."""

import json
import re

_NUMBER_PREFIX = re.compile(
    r"(?:|-|-?(?:0|[1-9][0-9]*)(?:\\.[0-9]*)?"
    r"|-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?[eE][+-]?[0-9]*)"
    r"\\Z"
)
_NUMBER_COMPLETE = re.compile(
    r"-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\\Z"
)

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
    def valid_number_structure(
        generated: str,
        number_fields: set[str],
    ) -> bool:
        """Check numeric fields in a partially generated JSON object."""
        in_string = False
        escaped = False
        start = 1
        fields: list[tuple[str, bool]] = []

        for index, char in enumerate(generated[1:], start=1):
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\\\":
                    escaped = True
                elif char == '"':
                    in_string = False
            elif char == '"':
                in_string = True
            elif char in ",}":
                fields.append((generated[start:index], True))
                start = index + 1

        fields.append((generated[start:], False))

        for field, complete in fields:
            match = re.fullmatch(
                r'\\s*"([^"\\\\]+)"\\s*:\\s*(.*?)\\s*',
                field,
                re.DOTALL,
            )
            if match is None or match.group(1) not in number_fields:
                continue

            pattern = _NUMBER_COMPLETE if complete else _NUMBER_PREFIX
            if pattern.fullmatch(match.group(2)) is None:
                return False

        return True

    def select_valid_token(
        self,
        logits: list[float],
        generated_ids: list[int],
        number_fields: set[str],
    ) -> int:
        """Block tokens that would break a number's JSON syntax."""
        best_id = max(range(len(logits)), key=logits.__getitem__)
        if not number_fields:
            return best_id

        def valid(token_id: int) -> bool:
            generated = self.model.decode(generated_ids + [token_id])
            return self.valid_number_structure(
                "{" + generated,
                number_fields,
            )

        if valid(best_id):
            return best_id

        for token_id in sorted(
            range(len(logits)),
            key=logits.__getitem__,
            reverse=True,
        ):
            if valid(token_id):
                return token_id

        raise ValueError("No valid numeric token available")

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
        number_fields = {
            name
            for name, definition in function.parameters.items()
            if definition.type == "number"
        }

        for _ in range(256):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = self.select_valid_token(
                logits,
                generated_ids,
                number_fields,
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
