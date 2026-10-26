"""Review ambiguous regular-expression arguments with the LLM."""

from pydantic import BaseModel, ConfigDict

from .constrained_decoding import ConstrainedDecoder
from .function_selector import FunctionSelector
from .model.models import Prompt
from .model.regex_review import RegexInterpretations


class RegexArgumentReviewer(BaseModel):
    """Resolve regex/literal ambiguity after argument generation."""

    model_config = ConfigDict(extra="forbid")

    decoder: ConstrainedDecoder

    def review(
        self,
        prompt: Prompt,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        """Return arguments with a refined regex when review is applicable."""
        expected_keys = {
            "source_string",
            "regex",
            "replacement",
        }
        if set(arguments) != expected_keys:
            return arguments

        pattern = arguments["regex"]
        if not isinstance(pattern, str):
            return arguments

        candidates = RegexInterpretations(
            pattern=pattern
        ).candidates()
        if len(candidates) < 2:
            return arguments

        selector = FunctionSelector(
            decoder=self.decoder,
            functions=[
                function
                for _, function in candidates
            ],
        )
        selected = selector.select(prompt)

        for candidate, function in candidates:
            if function.name == selected.name:
                return {
                    **arguments,
                    "regex": candidate,
                }

        return arguments
