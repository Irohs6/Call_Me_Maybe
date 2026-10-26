import json

import numpy as np

from .argument_generator import ArgumentGenerator
from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .model.string_arguments import StringArgumentsGrammar


class StringArgumentGenerator(ArgumentGenerator):

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:

        if len(parameters) > 1:
            return self._generate_string_arguments(
                prompt,
                function,
                parameters,
            )

        generated_values: dict[str, object] = {}

        context_builder = ContextBuilder(
            functions=[function]
        )

        for parameter in parameters:
            current_values = {
                **values,
                **generated_values,
            }

            context_ids = self._string_context_ids(
                context_builder,
                prompt,
                function,
                parameter,
                current_values,
            )

            value = self._generate_free_string(
                context_ids
            )

            generated_values[parameter] = value

        return generated_values

    def _string_context_ids(
        self,
        context_builder: ContextBuilder,
        prompt: Prompt,
        function: Function,
        parameter: str,
        values: dict[str, object],
    ) -> list[int]:

        chat_context = (
            context_builder.build_string_parameter_context(
                prompt,
                function,
            )
        )

        previous_arguments = json.dumps(
            values,
            ensure_ascii=False,
        )[1:-1]

        if previous_arguments:
            previous_arguments += ", "

        chat_context += (
            '{"name": '
            f'{json.dumps(function.name)}, "arguments": {{'
            f'{previous_arguments}'
            f'{json.dumps(parameter)}: "'
        )

        return self.model.encode(
            chat_context
        )[0].tolist()

    def _generate_free_string(
        self,
        context_ids: list[int],
    ) -> str:

        generated_ids: list[int] = []
        decoder = json.JSONDecoder()

        for _ in range(100):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            next_token_id = max(
                range(len(logits)),
                key=lambda token_id: logits[token_id],
            )

            generation = self.model.decode(
                [next_token_id]
            )

            if generation == "":
                raise ValueError(
                    "Model ended before closing the JSON string"
                )

            generated_ids.append(next_token_id)

            argument_progress = self.model.decode(
                generated_ids
            )

            try:
                value, _ = decoder.raw_decode(
                    '"' + argument_progress
                )
            except json.JSONDecodeError:
                continue

            if isinstance(value, str):
                return value

        raise ValueError(
            "String generation exceeded maximum token count"
        )

    def _generate_string_arguments(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
    ) -> dict[str, object]:

        builder = ContextBuilder(
            functions=[function]
        )

        context = (
            builder.build_string_parameter_context(
                prompt,
                function,
            )
        )

        context += (
            '{"name": '
            f'{json.dumps(function.name)}, "arguments": {{'
        )

        context_ids = self.model.encode(
            context
        )[0].tolist()

        grammar = StringArgumentsGrammar(
            parameters=parameters
        )

        generated_ids: list[int] = []
        progress = "{"

        for _ in range(512):
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            for token_id in np.argsort(logits)[::-1]:
                token_id = int(token_id)

                if not np.isfinite(logits[token_id]):
                    continue

                candidate = (
                    "{"
                    + self.model.decode(
                        generated_ids + [token_id]
                    )
                )

                if candidate == progress:
                    continue

                status = grammar.status(candidate)

                if status == "invalid":
                    continue

                generated_ids.append(token_id)
                progress = candidate

                if status == "complete":
                    arguments, _ = (
                        json.JSONDecoder().raw_decode(
                            progress
                        )
                    )
                    return arguments

                break

            else:
                raise ValueError(
                    "No token can continue "
                    "the string arguments"
                )

        raise ValueError(
            "String arguments exceeded maximum token count"
        )
