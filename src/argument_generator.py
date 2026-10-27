"""Generate function arguments separately and assemble a Python dict."""

import json

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, Field

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .string_grammar import StringGrammar


class ArgumentGenerator(BaseModel):
    """Constrain strings, leaving other types unconstrained for now."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    token_bytes: dict[int, bytes] = Field(default_factory=dict)

    def __init__(self, model: Small_LLM_Model) -> None:
        super().__init__(model=model)
        self.token_bytes = StringGrammar.load_vocabulary(
            model.get_path_to_vocab_file()
        )

    @staticmethod
    def validate_arguments(
        arguments: object,
        function: Function,
    ) -> dict[str, object]:
        """Verify every generated value matches the function schema."""
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

    def generate_string(self, context_ids: list[int]) -> str:
        """Select only tokens preserving a valid JSON string prefix."""
        grammar = StringGrammar()
        progress = b""
        generated_ids: list[int] = []

        for _ in range(256):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            candidates = (
                token_id
                for token_id in self.token_bytes
                if token_id < len(logits)
            )

            next_token_id = None
            for token_id in sorted(
                candidates, key=logits.__getitem__, reverse=True,
            ):
                if grammar.status(
                    progress + self.token_bytes[token_id]
                ) != -2:
                    next_token_id = token_id
                    break

            if next_token_id is None:
                raise ValueError("No valid string token available")

            generated_ids.append(next_token_id)
            progress += self.token_bytes[next_token_id]

            if grammar.status(progress) == 0:
                return json.loads(progress.decode("utf-8"))

        raise ValueError("String exceeded the 256-token limit")

    def generate_unconstrained(self, context_ids: list[int]) -> object:
        """Generate a raw scalar until a JSON separator follows it."""
        generated_ids: list[int] = []
        decoder = json.JSONDecoder()

        for _ in range(128):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = max(
                range(len(logits)), key=logits.__getitem__,
            )
            generated_ids.append(next_token_id)
            text = self.model.decode(generated_ids).lstrip()

            try:
                value, end = decoder.raw_decode(text)
            except json.JSONDecodeError:
                continue

            remaining = text[end:]
            if remaining.startswith((",", "}", "\n", "\r")):
                return value
            if remaining.lstrip(" \t").startswith((",", "}", "\n", "\r")):
                return value

        raise ValueError("Scalar exceeded the 128-token limit")

    def generate(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:
        """Generate every parameter, then build a Python dictionary."""
        arguments: dict[str, object] = {}
        builder = ContextBuilder(functions=[function])

        for name, definition in function.parameters.items():
            context = builder.build_argument_context(
                prompt, function, name, arguments,
            )
            context_ids = self.model.encode(context)[0].tolist()

            if definition.type == "string":
                value = self.generate_string(context_ids)
            else:
                value = self.generate_unconstrained(context_ids)

            arguments[name] = value

        return self.validate_arguments(arguments, function)
