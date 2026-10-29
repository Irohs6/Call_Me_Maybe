"""Generate typed arguments for the selected function."""

import heapq
import json
import math

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, PrivateAttr

from .json_checker import JsonChecker
from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt


class ArgumentGenerator(BaseModel):
    """Generate every argument in a single JSON sequence."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    _vocabulary_ids: tuple[int, ...] | None = PrivateAttr(default=None)

    def vocabulary_ids(self) -> tuple[int, ...]:
        """Load and reuse the model vocabulary across prompts."""
        if self._vocabulary_ids is None:
            with open(
                self.model.get_path_to_vocab_file(),
                encoding="utf-8",
            ) as file:
                vocabulary = json.load(file)

            if not isinstance(vocabulary, dict):
                raise ValueError("The vocabulary must be a JSON object")

            ids = {
                token_id for token_id in vocabulary.values()
                if type(token_id) is int and token_id >= 0
            }
            if not ids:
                raise ValueError("The vocabulary contains no token IDs")
            self._vocabulary_ids = tuple(sorted(ids))
        return self._vocabulary_ids

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
        checker = JsonChecker(parameters={
            name: definition.type
            for name, definition in function.parameters.items()
        })

        vocabulary_ids = self.vocabulary_ids()

        for _ in range(256):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )
            previous_text = self.model.decode(generated_ids)
            candidates = [
                (-logits[token_id], token_id)
                for token_id in vocabulary_ids
                if token_id < len(logits)
                and math.isfinite(logits[token_id])
            ]
            heapq.heapify(candidates)

            while candidates:
                _, token_id = heapq.heappop(candidates)
                next_ids = generated_ids + [token_id]
                text = self.model.decode(next_ids)
                if text == previous_text or "\ufffd" in text:
                    continue

                candidate = "{" + text
                status = checker.status(candidate)
                if status == -2:
                    continue

                generated_ids.append(token_id)
                if status == 0:
                    return self.validate_arguments(
                        json.loads(candidate), function,
                    )
                break
            else:
                raise ValueError("No valid token can continue the JSON")

        raise ValueError(
            "Argument generation exceeded the 256-token limit"
        )
