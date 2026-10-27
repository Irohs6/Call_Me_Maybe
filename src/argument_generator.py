"""Generate typed arguments for the selected function."""

import json

from llm_sdk import Small_LLM_Model

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .json_grammar import JsonArgumentsGrammar


class ArgumentGenerator:
    """Generate every argument in a single JSON sequence."""

    def __init__(
        self,
        model: Small_LLM_Model,
    ) -> None:
        self.model = model
        self.token_bytes = JsonArgumentsGrammar.load_vocabulary(
            model.get_path_to_vocab_file()
        )

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

    def select_valid_token(
        self,
        logits: list[float],
        progress: bytes,
        grammar: JsonArgumentsGrammar,
    ) -> int:
        """Choose the highest-scoring token allowed by the JSON grammar."""
        token_ids = [
            token_id
            for token_id in self.token_bytes
            if token_id < len(logits)
        ]

        def valid(token_id: int) -> bool:
            return grammar.status(
                progress + self.token_bytes[token_id]
            ) != "invalid"

        best_id = max(token_ids, key=logits.__getitem__)
        if valid(best_id):
            return best_id

        for token_id in sorted(
            token_ids,
            key=logits.__getitem__,
            reverse=True,
        ):
            if valid(token_id):
                return token_id

        raise ValueError("No token can continue valid JSON")

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
        progress = b"{"
        grammar = JsonArgumentsGrammar(
            parameters={
                name: definition.type
                for name, definition in function.parameters.items()
            }
        )

        for _ in range(256):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            next_token_id = self.select_valid_token(
                logits, progress, grammar,
            )
            generated_ids.append(next_token_id)
            progress += self.token_bytes[next_token_id]

            if grammar.status(progress) == "complete":
                arguments = json.loads(progress.decode("utf-8"))
                return self.validate_arguments(arguments, function)

        raise ValueError(
            "Argument generation exceeded the 256-token limit"
        )
