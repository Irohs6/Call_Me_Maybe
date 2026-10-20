from .model.models import Function
from llm_sdk import Small_LLM_Model


class FunctionSelector:
    def __init__(
        self,
        model: Small_LLM_Model,
        functions: list[Function],
    ):
        self.model = model
        self.functions = functions

    def select(self, context: str) -> Function:
        context_ids = self.model.encode(context)[0].tolist()

        function_tokens: dict[str, list[int]] = {}
        for function in self.functions:
            tokens = self.model.encode(function.name)[0].tolist()
            function_tokens[function.name] = tokens

        generated_ids: list[int] = []

        candidates = function_tokens.copy()
        while True:
            position = len(generated_ids)
            allowed_ids: list[int] = []

            for tokens in candidates.values():
                if len(tokens) > position:
                    token_id = tokens[position]
                    if token_id not in allowed_ids:
                        allowed_ids.append(token_id)
            logits = self.model.get_logits_from_input_ids(
                context_ids + generated_ids)

            print("allowed_ids, generated_ids", allowed_ids, generated_ids)
            next_token_id = max(
                allowed_ids,
                key=lambda token_id: logits[token_id]
            )
            generated_ids.append(next_token_id)

            # Remove candidates that are no longer possible
            candidates = {
                name: tokens
                for name, tokens in function_tokens.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            for function in self.functions:
                if function.name in candidates:
                    if candidates[function.name] == generated_ids:
                        return function
