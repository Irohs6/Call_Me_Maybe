from .model.models import Function
from llm_sdk import Small_LLM_Model
from .model.context_builder import ContextBuilder
from .model.models import Prompt
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

        for index, number in enumerate(self.extract_numbers(prompt)):
            tokens = self.model.encode(number + "\n")[0].tolist()
            number_tokens[index] = (number, tokens)

        return number_tokens

    def extract_strings(self, prompt: str) -> list[str]:
        quoted_strings = re.findall(r"'([^']*)'|\"([^\"]*)\"", prompt)

        strings: list[str] = []

        for single_quote, double_quote in quoted_strings:
            value = single_quote or double_quote
            strings.append(value)

        strings.extend(prompt.split())

        return strings

    def string_tokens_available(
        self,
        prompt: str,
    ) -> dict[int, tuple[str, list[int]]]:
        string_tokens: dict[int, tuple[str, list[int]]] = {}

        for index, string in enumerate(self.extract_strings(prompt)):
            tokens = self.model.encode(string + "\n")[0].tolist()
            string_tokens[index] = (string, tokens)

        return string_tokens

    def generate_number(
        self,
        context_ids: list[int],
        prompt: str,
        number_tokens: dict[int, tuple[str, list[int]]]
    ) -> int | float:
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
                if tokens[:len(generated_ids)] == generated_ids
            }

            for key, (number, tokens) in candidates.items():
                if tokens == generated_ids:
                    number_tokens.pop(key)
                    return float(number)

    def generate_free_string(
        self,
        context_ids: list[int],
    ) -> str:
        generated_ids: list[int] = []

        end_id = self.model.encode("\n")[0].tolist()[0]

        while True:
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            next_token_id = max(
                range(len(logits)),
                key=lambda token_id: logits[token_id],
            )

            if next_token_id == end_id:
                return self.model.decode(generated_ids)

            generated_ids.append(next_token_id)

    def generate_string(
        self,
        context_ids: list[int],
        prompt: str,
        string_tokens: dict[int, tuple[str, list[int]]]
    ) -> str:
        candidates = string_tokens.copy()
        generated_ids: list[int] = []

        if not string_tokens:
            raise ValueError("No string found in prompt")

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
                index: (string, tokens)
                for index, (string, tokens) in string_tokens.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            for key, (string, tokens) in candidates.items():
                if tokens == generated_ids:
                    string_tokens.pop(key)
                    return string

    def final_selection(
        self,
        prompt: Prompt,
        function: Function,
    ) -> dict[str, object]:

        context_builder = ContextBuilder(functions=[function])

        values: dict[str, object] = {}
        number_tokens: list[int] = self.number_tokens_available(prompt.prompt)
        string_tokens = self.string_tokens_available(prompt.prompt)

        for parameter in function.parameters:
            parameter_context: str = context_builder.build_parameter_context(
                prompt,
                function,
                parameter,
                values=values,
            )

            context_ids = self.model.encode(parameter_context)[0].tolist()

            type_ = function.parameters[parameter].type

            if type_ == "number":
                value = self.generate_number(context_ids, prompt.prompt,
                                             number_tokens)
                values[parameter] = value
                print(f"Generated number for parameter '{parameter}': {value}")
            elif type_ == "string":
                value = self.generate_free_string(context_ids)
                values[parameter] = value
                print(f"Generated string for parameter '{parameter}': {value}")

        return values
