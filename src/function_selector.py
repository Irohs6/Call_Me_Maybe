"""Select functions and generate their arguments through the public SDK."""

from llm_sdk import Small_LLM_Model

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .constrained_decoding import allowed_ids_for_position


class FunctionSelector:
    """Choose one function with A/B comparisons and prepare its inputs."""

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[Function],
    ):
        self.model = model
        self.functions = functions

    def select(self, prompt: Prompt) -> Function:
        if not self.functions:
            raise ValueError("No functions available for selection")

        selected = self.functions[0]
        if len(self.functions) == 1:
            return selected

        label_tokens = {
            label: self.model.encode(label + '"')[0].tolist()
            for label in ("A", "B")
        }
        for candidate in self.functions[1:]:
            builder = ContextBuilder(functions=[selected, candidate])
            comparison_context = builder.build_with_prompt(
                prompt, comparison=True
            )
            context_ids = self.model.encode(comparison_context)[0].tolist()
            remaining = label_tokens.copy()
            generated_ids: list[int] = []

            while len(remaining) > 1:
                allowed_ids = allowed_ids_for_position(
                    remaining, len(generated_ids)
                )
                logits = self.model.get_logits_from_input_ids(
                    context_ids + generated_ids
                )
                next_token_id = max(
                    allowed_ids, key=lambda token_id: logits[token_id]
                )
                generated_ids.append(next_token_id)
                remaining = {
                    label: tokens
                    for label, tokens in remaining.items()
                    if tokens[:len(generated_ids)] == generated_ids
                }

            if "B" in remaining:
                selected = candidate

        return selected
