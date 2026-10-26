import re
from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .argument_generator import ArgumentGenerator
from .constrained_decoding import allowed_ids_for_position

NumberTokens = dict[int, tuple[str, list[int]]]


class NumberArgumentGenerator(ArgumentGenerator):

    def generate(
        self,
        prompt: Prompt,
        function: Function,
        parameters: list[str],
        values: dict[str, object],
    ) -> dict[str, object]:

        number_tokens = self._number_tokens_available(
            prompt.prompt
        )

        context_builder = ContextBuilder(
            functions=[function]
        )

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

            context_ids = self.model.encode(
                parameter_context
            )[0].tolist()

            value = self._generate_number(
                context_ids,
                number_tokens,
            )

            generated_values[parameter] = value

        return generated_values

    def _extract_numbers(self, prompt: str) -> list[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", prompt)

    def _number_tokens_available(
        self,
        prompt: str,
    ) -> NumberTokens:
        number_tokens: NumberTokens = {}

        for index, number in enumerate(self._extract_numbers(prompt)):
            tokens = self.model.encode(number + "\n")[0].tolist()
            number_tokens[index] = (number, tokens)

        return number_tokens

    def _generate_number(
        self,
        context_ids: list[int],
        number_tokens: NumberTokens,
    ) -> float:
        candidates = number_tokens.copy()
        generated_ids: list[int] = []

        if not number_tokens:
            raise ValueError("No number found in prompt")

        while True:
            position = len(generated_ids)

            allowed_ids = allowed_ids_for_position(
                {k: v[1] for k, v in candidates.items()}, position
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
                index: (number, tokens)
                for index, (number, tokens) in number_tokens.items()
                if tokens[: len(generated_ids)] == generated_ids
            }

            for key, (number, tokens) in candidates.items():
                if tokens == generated_ids:
                    number_tokens.pop(key)
                    return float(number)