"""Validate function-selection errors without running the model."""

import unittest
from unittest.mock import MagicMock

from src.function_selector import FunctionSelector
from src.model.models import Function, Type


class FunctionSelectorTests(unittest.TestCase):
    @staticmethod
    def function() -> Function:
        return Function(
            name="fn_test",
            description="Test function",
            parameters={},
            returns=Type(type="string"),
        )

    def test_empty_function_list_has_clear_error(self) -> None:
        model = MagicMock()
        selector = FunctionSelector.model_construct(model=model, functions=[])
        with self.assertRaisesRegex(ValueError, "No functions available"):
            selector.select("context")

    def test_out_of_range_token_has_clear_error(self) -> None:
        model = MagicMock()
        model.encode.return_value[0].tolist.return_value = [99]
        model.get_logits_from_input_ids.return_value = [1.0]
        selector = FunctionSelector.model_construct(
            model=model,
            functions=[self.function()],
        )
        with self.assertRaisesRegex(ValueError, "outside model vocabulary"):
            selector.select("context")


if __name__ == "__main__":
    unittest.main()
