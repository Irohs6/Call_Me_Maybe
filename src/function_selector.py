"""Select functions and generate their arguments through the public SDK."""

from llm_sdk import Small_LLM_Model

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .model.regex_review import RegexInterpretations
from .constrained_decoding import allowed_ids_for_position
from .number_argument_generator import NumberArgumentGenerator
from .string_argument_generator import StringArgumentGenerator


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

    def refine_regex_arguments(
        self,
        prompt: Prompt,
        arguments: dict[str, object],
    ) -> dict[str, object]:
        """Review ambiguous regex syntax through the LLM."""
        if set(arguments) != {"source_string", "regex", "replacement"}:
            return arguments
        pattern = arguments["regex"]
        if not isinstance(pattern, str):
            return arguments
        candidates = RegexInterpretations(pattern=pattern).candidates()
        if len(candidates) < 2:
            return arguments
        functions = [function for _, function in candidates]
        selector = FunctionSelector(self.model, functions)
        selected = selector.select(prompt)
        for candidate, function in candidates:
            if function.name == selected.name:
                return {**arguments, "regex": candidate}
        return arguments

    def final_selection(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:

        values: dict[str, object] = {}

        number_parameters = [
            name
            for name, definition in function.parameters.items()
            if definition.type == "number"
        ]

        if number_parameters:
            number_generator = NumberArgumentGenerator(
                model=self.model,
            )

            generated_numbers = number_generator.generate(
                prompt=prompt,
                function=function,
                parameters=number_parameters,
                values=values,
            )

            values.update(generated_numbers)

        string_parameters = [
            name
            for name, definition in function.parameters.items()
            if definition.type == "string"
        ]

        if string_parameters:
            string_generator = StringArgumentGenerator(
                model=self.model,
            )

            generated_strings = string_generator.generate(
                prompt=prompt,
                function=function,
                parameters=string_parameters,
                values=values,
            )

            values.update(generated_strings)

        return self.refine_regex_arguments(
            prompt=prompt,
            arguments=values,
        )
