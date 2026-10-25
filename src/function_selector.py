"""Select functions and generate their arguments through the public SDK."""

import json
import re
from typing import TypeVar

import numpy as np
from llm_sdk import Small_LLM_Model

from .model.context_builder import ContextBuilder
from .model.models import Function, Prompt
from .model.regex_review import RegexInterpretations
from .model.string_arguments import StringArgumentsGrammar


CandidateKey = TypeVar("CandidateKey")
NumberTokens = dict[int, tuple[str, list[int]]]


class FunctionSelector:
    """Choose one function with A/B comparisons and prepare its inputs."""

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[Function],
    ):
        self.model = model
        self.functions = functions

    def allowed_ids_for_position(
        self, function_tokens: dict[CandidateKey, list[int]], position: int
    ) -> list[int]:
        allowed_ids: list[int] = []
        for tokens in function_tokens.values():
            if len(tokens) > position:
                token_id = tokens[position]
                if token_id not in allowed_ids:
                    allowed_ids.append(token_id)
        return allowed_ids

    def select(self, context: str) -> Function:
        if not self.functions:
            raise ValueError("No functions available for selection")

        selected = self.functions[0]
        if len(self.functions) == 1:
            return selected

        request = ContextBuilder.extract_user_request(context)
        label_tokens = {
            label: self.model.encode(label + '"')[0].tolist()
            for label in ("A", "B")
        }
        for candidate in self.functions[1:]:
            builder = ContextBuilder(functions=[selected, candidate])
            comparison_context = builder.build_with_prompt(
                Prompt(prompt=request), comparison=True
            )
            context_ids = self.model.encode(comparison_context)[0].tolist()
            remaining = label_tokens.copy()
            generated_ids: list[int] = []

            while len(remaining) > 1:
                allowed_ids = self.allowed_ids_for_position(
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

    def extract_numbers(self, prompt: str) -> list[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", prompt)

    def number_tokens_available(
        self,
        prompt: str,
    ) -> NumberTokens:
        number_tokens: NumberTokens = {}

        for index, number in enumerate(self.extract_numbers(prompt)):
            tokens = self.model.encode(number + "\n")[0].tolist()
            number_tokens[index] = (number, tokens)

        return number_tokens

    def generate_number(
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

            allowed_ids = self.allowed_ids_for_position(
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

    def string_context_ids(
        self,
        context_builder: ContextBuilder,
        prompt: Prompt,
        function: Function,
        parameter: str,
        values: dict[str, object],
    ) -> list[int]:
        chat_context = context_builder.build_string_parameter_context(
            prompt,
            function,
        )

        previous_arguments = json.dumps(values, ensure_ascii=False)[1:-1]
        if previous_arguments:
            previous_arguments += ", "

        chat_context += (
            '{"name": '
            f'{json.dumps(function.name)}, "arguments": {{'
            f'{previous_arguments}{json.dumps(parameter)}: "'
        )

        return self.model.encode(
            chat_context
        )[0].tolist()

    def generate_free_string(
        self,
        context_ids: list[int],
    ) -> str:
        generated_ids: list[int] = []
        argument_progress = ""
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
                value, _ = decoder.raw_decode('"' + argument_progress)
            except json.JSONDecodeError:
                continue

            if isinstance(value, str):
                return value

        raise ValueError(
            "String generation exceeded maximum token count"
        )

    def generate_string_arguments(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:
        """Generate related strings together, constrained by their JSON schema."""
        builder = ContextBuilder(functions=[function])
        context = builder.build_string_parameter_context(prompt, function)
        context += (
            '{"name": '
            f'{json.dumps(function.name)}, "arguments": {{'
        )
        context_ids = self.model.encode(context)[0].tolist()
        grammar = StringArgumentsGrammar(parameters=list(function.parameters))
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
                candidate = "{" + self.model.decode(
                    generated_ids + [token_id]
                )
                if candidate == progress:
                    continue
                status = grammar.status(candidate)
                if status == "invalid":
                    continue
                generated_ids.append(token_id)
                progress = candidate
                if status == "complete":
                    arguments, _ = json.JSONDecoder().raw_decode(progress)
                    return arguments
                break
            else:
                raise ValueError("No token can continue the string arguments")

        raise ValueError("String arguments exceeded maximum token count")

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
        context = ContextBuilder(functions=functions).build_with_prompt(prompt)
        selected = selector.select(context)
        for candidate, function in candidates:
            if function.name == selected.name:
                return {**arguments, "regex": candidate}
        return arguments

    def final_selection(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:

        if len(function.parameters) > 1 and all(
            definition.type == "string"
            for definition in function.parameters.values()
        ):
            arguments = self.generate_string_arguments(prompt, function)
            return self.refine_regex_arguments(prompt, arguments)

        context_builder = ContextBuilder(functions=[function])

        values: dict[str, object] = {}
        number_tokens = self.number_tokens_available(prompt.prompt)

        value: object
        for parameter in function.parameters:
            type_ = function.parameters[parameter].type

            if type_ == "number":
                parameter_context = context_builder.build_parameter_context(
                    prompt,
                    function,
                    parameter,
                    values,
                )

                context_ids = self.model.encode(
                    parameter_context
                )[0].tolist()

                value = self.generate_number(
                    context_ids, number_tokens
                )
                values[parameter] = value
                print(f"Generated number for parameter '{parameter}': {value}")

            elif type_ == "string":
                context_ids = self.string_context_ids(
                    context_builder,
                    prompt,
                    function,
                    parameter,
                    values,
                )

                value = self.generate_free_string(context_ids)
                values[parameter] = value
                print(f"Generated string for parameter '{parameter}': {value}")

        return values
