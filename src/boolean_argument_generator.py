"""Generate boolean function arguments with constrained decoding."""

import json

from .argument_generator import ArgumentGenerator
from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt


class BooleanArgumentGenerator(ArgumentGenerator):
    """Generate JSON boolean arguments as true or false."""

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate boolean values for the requested parameter names."""
        if not parameters:
            return {}

        context_builder = ContextBuilder(
            functions=[function]
        )
        boolean_tokens = {
            "true": self.decoder.encode("true\n"),
            "false": self.decoder.encode("false\n"),
        }
        generated_values: dict[str, object] = {}

        for parameter in parameters:
            current_values = {
                **values,
                **generated_values,
            }
            context = context_builder.build_argument_context(
                prompt=prompt,
                function=function,
                parameters=[parameter],
                values=current_values,
            )
            context += (
                "{"
                f"{json.dumps(parameter)}: "
            )
            context_ids = self.decoder.encode(context)

            selected = self.decoder.select_candidate(
                context_ids,
                boolean_tokens,
            )
            generated_values[parameter] = (
                selected == "true"
            )

        return generated_values
