from .model.models import Function
from llm_sdk import Small_LLM_Model
from .model.context_builder import ContextBuilder
from .model.models import Prompt
import json
import re


class FunctionSelector:

    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[Function],
    ):
        self.model = model
        self.functions = functions

    def allowed_ids_for_position(
        self, function_tokens: dict[str, list[int]], position: int
    ) -> list[int]:
        allowed_ids: list[int] = []
        for tokens in function_tokens.values():
            if len(tokens) > position:
                token_id = tokens[position]
                if token_id not in allowed_ids:
                    allowed_ids.append(token_id)
        return allowed_ids

    def token_function_available(self) -> dict[str, list[int]]:
        function_tokens: dict[str, list[int]] = {}
        for function in self.functions:
            tokens = self.model.encode(function.name + "\n")[0].tolist()
            function_tokens[function.name] = tokens
        return function_tokens

    def select(self, context: str) -> Function:
        context_ids = self.model.encode(context)[0].tolist()
        function_tokens = self.token_function_available()
        generated_ids: list[int] = []
        while True:
            position = len(generated_ids)
            allowed_ids = self.allowed_ids_for_position(
                function_tokens, position
            )
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            next_token_id = max(
                allowed_ids, key=lambda token_id: logits[token_id]
            )
            generated_ids.append(next_token_id)

            # Remove candidates that are no longer possible
            function_tokens = {
                name: tokens
                for name, tokens in function_tokens.items()
                if tokens[: len(generated_ids)] == generated_ids
            }

            for function in self.functions:
                if function.name in function_tokens:
                    if function_tokens[function.name] == generated_ids:
                        return function

    def extract_numbers(self, prompt: str) -> list[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", prompt)

    def number_tokens_available(
        self,
        prompt: str,
    ) -> dict[str, list[int]]:
        number_tokens: dict[str, list[int]] = {}

        for number in self.extract_numbers(prompt):
            tokens = self.model.encode(number + "\n")[0].tolist()
            number_tokens[number] = tokens

        return number_tokens

    def generate_number(
        self,
        context_ids: list[int],
        prompt: str,
    ) -> int | float:
        number_tokens_ids: list[int] = self.number_tokens_available(prompt)
        generated_ids: list[int] = []

        if not number_tokens_ids:
            raise ValueError("No number found in prompt")

        while True:
            position = len(generated_ids)

            allowed_ids = self.allowed_ids_for_position(number_tokens_ids,
                                                        position)
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            next_token_id = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )

            generated_ids.append(next_token_id)

            number_tokens_ids = {
                number: tokens
                for number, tokens in number_tokens_ids.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            for number, tokens in number_tokens_ids.items():
                if tokens == generated_ids:
                    return float(number)

    def final_selection(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:

        context_builder = ContextBuilder(functions=[function])

        values: dict[str, object] = {}

        for parameter in function.parameters:
            parameter_context: str = context_builder.build_parameter_context(
                prompt,
                function,
                parameter,
                values=values,
            )
            print(parameter_context)

            context_ids = self.model.encode(parameter_context)[0].tolist()

            type_ = function.parameters[parameter].type

            if type_ == "number":
                value = self.generate_number(context_ids, prompt.prompt)
                values[parameter] = value

        return values
