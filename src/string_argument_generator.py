"""Generate string arguments with an incremental JSON grammar."""

import json

from .argument_generator import ArgumentGenerator
from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .model.string_arguments import StringArgumentsGrammar


class StringArgumentGenerator(ArgumentGenerator):
    """Generate one or more string arguments through one JSON grammar."""

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:
        """Generate all requested string parameters as one JSON object."""
        if not parameters:
            return {}

        builder = ContextBuilder(
            functions=[function]
        )
        context = builder.build_string_arguments_context(
            prompt=prompt,
            function=function,
            parameters=parameters,
            values=values,
        )
        context += (
            '{"name": '
            f'{json.dumps(function.name)}, '
            '"arguments": {'
        )
        context_ids = self.decoder.encode(context)

        grammar = StringArgumentsGrammar(
            parameters=parameters
        )
        generated_ids: list[int] = []
        progress = "{"

        for _ in range(512):
            logits = self.decoder.logits(
                context_ids + generated_ids
            )

            def valid_token(token_id: int) -> bool:
                """Check whether one token keeps the JSON prefix valid."""
                candidate = (
                    "{"
                    + self.decoder.decode(
                        generated_ids + [token_id]
                    )
                )
                return (
                    candidate != progress
                    and grammar.status(candidate) != "invalid"
                )

            next_token_id = self.decoder.highest_valid_token(
                logits,
                valid_token,
            )
            generated_ids.append(next_token_id)
            progress = (
                "{"
                + self.decoder.decode(generated_ids)
            )
            status = grammar.status(progress)

            if status != "complete":
                continue

            arguments, _ = json.JSONDecoder().raw_decode(
                progress
            )

            if not isinstance(arguments, dict):
                raise ValueError(
                    "Generated string arguments are not a JSON object"
                )

            return {
                parameter: arguments[parameter]
                for parameter in parameters
            }

        raise ValueError(
            "String arguments exceeded maximum token count"
        )
