"""Select the function that best matches a prompt."""

from collections.abc import Iterable

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict

from .model.models import Function


class FunctionSelector(BaseModel):
    """Select one function through constrained decoding."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Small_LLM_Model
    functions: list[Function]

    @staticmethod
    def allowed_ids_for_position(
        func_name_ids: Iterable[list[int]],
        position: int,
    ) -> list[int]:
        """Return unique token IDs for the current position."""
        allowed_ids: list[int] = []

        for tokens in func_name_ids:
            if position >= len(tokens):
                continue

            token_id = tokens[position]
            if token_id not in allowed_ids:
                allowed_ids.append(token_id)

        return allowed_ids

    def token_function_available(
        self,
    ) -> dict[str, list[int]]:
        """Encode available function names."""
        return {
            function.name: self.model.encode(
                function.name + '\n'
            )[0].tolist()
            for function in self.functions
        }

    def select(
        self,
        context: str,
    ) -> Function:
        """Return the function preferred by the model."""
        context_ids = self.model.encode(context)[0].tolist()
        candidates = self.token_function_available()
        generated_ids: list[int] = []

        while candidates:
            position = len(generated_ids)
            allowed_ids = self.allowed_ids_for_position(
                candidates.values(),
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

            candidates = {
                name: tokens
                for name, tokens in candidates.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            for function in self.functions:
                if candidates.get(function.name) == generated_ids:
                    return function

        raise ValueError("No function selected")
