"""Generate boolean function arguments with constrained decoding."""

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
            parameter_context = (
                context_builder.build_parameter_context(
                    prompt,
                    function,
                    parameter,
                    current_values,
                )
            )
            context_ids = self.decoder.encode(
                parameter_context
            )
            selected = self.decoder.select_candidate(
                context_ids,
                boolean_tokens,
            )
            generated_values[parameter] = (
                selected == "true"
            )

        return generated_values
