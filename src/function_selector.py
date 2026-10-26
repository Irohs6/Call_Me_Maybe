"""Select the function that best matches a prompt."""

from pydantic import BaseModel, ConfigDict

from .constrained_decoder import ConstrainedDecoder
from .context_builder import ContextBuilder
from .models import Function, Prompt


class FunctionSelector(BaseModel):
    """Choose one function through constrained LLM A/B comparisons."""

    model_config = ConfigDict(extra="forbid")

    decoder: ConstrainedDecoder
    functions: list[Function]

    def select(self, prompt: Prompt) -> Function:
        """Return the function preferred by successive LLM comparisons."""
        if not self.functions:
            raise ValueError(
                "No functions available for selection"
            )

        selected = self.functions[0]
        if len(self.functions) == 1:
            return selected

        label_tokens = {
            label: self.decoder.encode(label + '"')
            for label in ("A", "B")
        }

        for candidate in self.functions[1:]:
            builder = ContextBuilder(
                functions=[selected, candidate]
            )
            context = builder.build_selection_context(
                prompt,
                comparison=True,
            )
            context_ids = self.decoder.encode(context)
            choice = self.decoder.select_candidate(
                context_ids,
                label_tokens,
            )

            if choice == "B":
                selected = candidate

        return selected
