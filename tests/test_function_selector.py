"""Check function selection, including forced-token optimizations."""

import unittest
from unittest.mock import MagicMock

from src.function_selector import FunctionSelector
from src.model.models import Function, Type


class FunctionSelectorTests(unittest.TestCase):
    @staticmethod
    def function(name: str = "fn_test") -> Function:
        return Function(
            name=name,
            description="Test function",
            parameters={},
            returns=Type(type="string"),
        )

    @staticmethod
    def model_with_tokens(
        encoded: dict[str, list[int]],
        logits: list[float],
    ) -> MagicMock:
        model = MagicMock()

        def encode(text: str) -> MagicMock:
            result = MagicMock()
            result.__getitem__.return_value.tolist.return_value = encoded[text]
            return result

        model.encode.side_effect = encode
        model.get_logits_from_input_ids.return_value = logits
        return model

    def test_empty_function_list_has_clear_error(self) -> None:
        model = MagicMock()
        selector = FunctionSelector.model_construct(model=model, functions=[])
        with self.assertRaisesRegex(ValueError, "No functions available"):
            selector.select("context")

    def test_out_of_range_token_has_clear_error(self) -> None:
        model = self.model_with_tokens(
            {"context": [99], "fn_test\n": [0], "fn_other\n": [5]},
            [1.0],
        )
        selector = FunctionSelector.model_construct(
            model=model,
            functions=[self.function(), self.function("fn_other")],
        )
        with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
            selector.select("context")

    def test_forced_tokens_do_not_call_model(self) -> None:
        model = self.model_with_tokens(
            {"context": [99], "fn_test\n": [3, 7, 11]},
            [],
        )
        selector = FunctionSelector.model_construct(
            model=model, functions=[self.function()],
        )
        self.assertEqual(selector.select("context").name, "fn_test")
        model.get_logits_from_input_ids.assert_not_called()

    def test_function_names_are_encoded_once_across_prompts(self) -> None:
        encoded = {
            "first prompt": [98],
            "second prompt": [99],
            "fn_add_numbers\n": [3, 7, 21],
            "fn_add_strings\n": [3, 7, 22],
        }
        logits = [0.0] * 23
        logits[22] = 10.0
        model = self.model_with_tokens(encoded, logits)
        selector = FunctionSelector.model_construct(
            model=model,
            functions=[
                self.function("fn_add_numbers"),
                self.function("fn_add_strings"),
            ],
        )

        self.assertEqual(
            selector.select("first prompt").name, "fn_add_strings",
        )
        self.assertEqual(
            selector.select("second prompt").name, "fn_add_strings",
        )
        encoded_texts = [
            item.args[0] for item in model.encode.call_args_list
        ]
        self.assertEqual(
            encoded_texts,
            [
                "first prompt", "fn_add_numbers\n",
                "fn_add_strings\n", "second prompt",
            ],
        )
        self.assertEqual(
            model.get_logits_from_input_ids.call_count, 2,
        )

    def test_shared_prefix_only_scores_the_branch(self) -> None:
        encoded = {
            "context": [99],
            "fn_add_numbers\n": [3, 7, 21, 31],
            "fn_add_strings\n": [3, 7, 22, 32],
        }
        for expected, preferred in (
            ("fn_add_numbers", 21),
            ("fn_add_strings", 22),
        ):
            with self.subTest(expected=expected):
                logits = [0.0] * 33
                logits[preferred] = 10.0
                model = self.model_with_tokens(encoded, logits)
                selector = FunctionSelector.model_construct(
                    model=model,
                    functions=[
                        self.function("fn_add_numbers"),
                        self.function("fn_add_strings"),
                    ],
                )
                self.assertEqual(selector.select("context").name, expected)
                model.get_logits_from_input_ids.assert_called_once_with(
                    [99, 3, 7],
                )


if __name__ == "__main__":
    unittest.main()
