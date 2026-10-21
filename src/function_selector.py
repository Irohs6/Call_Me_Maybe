from .model.models import Function
from llm_sdk import Small_LLM_Model
from .model.context_builder import ContextBuilder
from .model.models import Prompt
import json


class FunctionSelector:

    NUMBER_ALLOWED_CHARS = ["-", "0", "1", "2", "3", "4", "5", "6",
                            "7", "8", "9", ".", "\n",]

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

    def generate_number(
        self,
        context_ids: list[int],
    ) -> int | float:
        generated_ids: list[int] = []
        argument_progress: str = ""

        number_token_ids: list[int] = []

        for char in self.NUMBER_ALLOWED_CHARS:
            token_ids = self.model.encode(char)[0].tolist()

            if len(token_ids) != 1:
                continue

            number_token_ids.append(token_ids[0])

        while True:
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids
            )

            allowed_ids: list[int] = []

            for token_id in number_token_ids:
                generation = self.model.decode([token_id])
                candidate = argument_progress + generation

                if candidate.count(".") >= 2:
                    continue

                if candidate.count("-") >= 2:
                    continue

                if "-" in candidate and not candidate.startswith("-"):
                    continue

                if generation == "\n":
                    try:
                        float(argument_progress)
                    except ValueError:
                        continue

                allowed_ids.append(token_id)

            if not allowed_ids:
                raise ValueError("Cannot generate a valid number")

            next_token_id = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )

            generation = self.model.decode([next_token_id])
            print(
                "generated:",
                repr(generation),
                "progress:",
                repr(argument_progress),
            )
            if generation == "\n":
                return float(argument_progress)

            argument_progress += generation
            generated_ids.append(next_token_id)

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
                value = self.generate_number(context_ids)
                values[parameter] = value

        return values
