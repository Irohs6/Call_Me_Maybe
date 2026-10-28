"""Generate typed arguments for the selected function."""

import json

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt


class ArgumentGenerator(BaseModel):
    """Generate values separately and assemble the argument dictionary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model

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
    def build_value_prefix(
        arguments: dict[str, object],
        name: str,
    ) -> str:
        """Build a partial JSON dictionary ending just after name:."""
        fields = dict(arguments)
        fields[name] = ""
        return json.dumps(fields, ensure_ascii=False)[:-3]

    @staticmethod
    def read_generated_value(
        text: str,
        type_value: str,
    ) -> object | None:
        """Read one complete primitive using Python's JSON decoder."""
        try:
            value, end = json.JSONDecoder().raw_decode(text)
        except json.JSONDecodeError:
            return None

        if type_value == "string":
            return value if isinstance(value, str) else None
        if type_value == "boolean":
            return value if isinstance(value, bool) else None
        if type_value not in ("integer", "number"):
            raise ValueError(f"Unsupported argument type: {type_value}")

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if type_value == "integer" and not isinstance(value, int):
            return None

        # A number alone could still grow: 2 -> 265, 1 -> 1.5.
        # Wait for the model to emit a separator before accepting it.
        if end >= len(text) or text[end] not in " \t\r\n,}":
            return None
        return value

    def generate_value(
        self,
        context: str,
        type_value: str,
    ) -> object:
        """Generate one JSON primitive after a fixed dictionary prefix."""
        context_ids = self.model.encode(context)[0].tolist()
        generated_ids: list[int] = []

        for _ in range(256):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            token_id = max(
                range(len(logits)),
                key=lambda index: logits[index],
            )
            generated_ids.append(token_id)
            candidate = self.model.decode(generated_ids)
            value = self.read_generated_value(candidate, type_value)
            if value is not None:
                return value

        raise ValueError(
            f"Could not generate a complete {type_value} value"
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
            type_value = definition.type

            if type_value == "string":
                valid = isinstance(value, str)
            elif type_value == "number":
                valid = isinstance(value, (int, float)) and not isinstance(
                    value, bool
                )
                if valid:
                    value = float(value)
            elif type_value == "integer":
                valid = isinstance(value, int) and not isinstance(value, bool)
            elif type_value == "boolean":
                valid = isinstance(value, bool)
            else:
                raise ValueError(f"Unsupported argument type: {type_value}")

            if not valid:
                raise ValueError(
                    f"Invalid value for '{name}': expected {type_value}"
                )

            validated[name] = value

        return validated

    def generate(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:
        """Build the dictionary in order, generating one value at a time."""
        if not function.parameters:
            return {}

        base_context = self.create_argument_context(prompt, function)
        arguments: dict[str, object] = {}

        for name, definition in function.parameters.items():
            prefix = self.build_value_prefix(arguments, name)
            # The existing context already ends with the opening '{'.
            context = base_context + prefix[1:]
            value = self.generate_value(context, definition.type)
            arguments[name] = value

        # Serialize only Python values, never raw model-generated JSON.
        json.dumps(arguments, ensure_ascii=False, allow_nan=False)
        return self.validate_arguments(arguments, function)
