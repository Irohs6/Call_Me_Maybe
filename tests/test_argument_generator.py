"""Tests for constrained argument generation using a fake LLM."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.argument_generator import ArgumentGenerator
from src.model.models import Function, Prompt, Type


class ArgumentGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.prompt = Prompt(prompt="Test the function")

    @staticmethod
    def function(parameters: dict[str, str]) -> Function:
        return Function(
            name="fn_test",
            description="Test function",
            parameters={
                name: Type(type=type_name)
                for name, type_name in parameters.items()
            },
            returns=Type(type="string"),
        )

    def make_model(
        self,
        tokens: dict[int, str],
        scores: list[list[float]],
        vocabulary_ids: list[int] | None = None,
    ) -> MagicMock:
        path = Path(self.tempdir.name) / "vocab.json"
        if vocabulary_ids is None:
            vocabulary_ids = list(tokens)

        path.write_text(
            json.dumps({str(i): i for i in vocabulary_ids}),
            encoding="utf-8",
        )

        model = MagicMock()
        model.get_path_to_vocab_file.return_value = str(path)
        model.encode.return_value[0].tolist.return_value = [900]
        model.decode.side_effect = lambda ids: "".join(
            tokens[token_id] for token_id in ids
        )
        model.get_logits_from_input_ids.side_effect = lambda ids: (
            scores[len(ids) - 1]
        )
        return model

    def test_filters_invalid_tokens_before_selection(self) -> None:
        model = self.make_model(
            tokens={
                0: '"wrong":1}',
                1: '"a":2',
                2: ',"b":3}',
                3: '"a":"bad"',
                4: '"a":2}',
                5: "<special>",
            },
            scores=[
                [16, 10, 1, 15, 14, 20],
                [16, 10, 17, 15, 14, 20],
            ],
            vocabulary_ids=[0, 1, 2, 3, 4],
        )
        generator = ArgumentGenerator.model_construct(model=model)
        result = generator.generate(
            self.prompt,
            self.function({"a": "number", "b": "number"}),
        )

        self.assertEqual(result, {"a": 2.0, "b": 3.0})
        self.assertEqual(
            model.get_logits_from_input_ids.call_count, 2,
        )
        model.get_path_to_vocab_file.assert_called_once()

    def test_boolean_and_string_values(self) -> None:
        model = self.make_model(
            tokens={
                0: '"name":true,"enabled":false}',
                1: '"name":"Iroh"',
                2: ',"enabled":true}',
            },
            scores=[
                [20, 10, 5],
                [20, 10, 12],
            ],
        )
        generator = ArgumentGenerator.model_construct(model=model)
        result = generator.generate(
            self.prompt,
            self.function({"name": "string", "enabled": "boolean"}),
        )
        self.assertEqual(result, {"name": "Iroh", "enabled": True})

    def test_integer_rejects_decimal_tokens(self) -> None:
        model = self.make_model(
            tokens={
                0: '"count":1.5}',
                1: '"count":12}',
            },
            scores=[[20, 10]],
        )
        generator = ArgumentGenerator.model_construct(model=model)
        result = generator.generate(
            self.prompt,
            self.function({"count": "integer"}),
        )
        self.assertEqual(result, {"count": 12})

    def test_raises_if_no_valid_continuation(self) -> None:
        model = self.make_model(
            tokens={0: '"unknown":5}'},
            scores=[[10]],
        )
        generator = ArgumentGenerator.model_construct(model=model)
        with self.assertRaisesRegex(ValueError, "No valid token"):
            generator.generate(
                self.prompt,
                self.function({"count": "number"}),
            )

    def test_vocabulary_is_loaded_only_once(self) -> None:
        model = self.make_model(
            tokens={0: '"count":12}'},
            scores=[[10]],
        )
        generator = ArgumentGenerator.model_construct(model=model)
        first = generator.vocabulary_ids()
        second = generator.vocabulary_ids()
        self.assertEqual(first, second)
        model.get_path_to_vocab_file.assert_called_once()

    def test_non_finite_logits_are_skipped(self) -> None:
        model = self.make_model(
            tokens={0: '"count":1}', 1: '"count":2}'},
            scores=[[float("nan"), 10]],
        )
        generator = ArgumentGenerator.model_construct(model=model)
        result = generator.generate(
            self.prompt, self.function({"count": "integer"}),
        )
        self.assertEqual(result, {"count": 2})

    def test_non_finite_generated_number_tries_next_token(self) -> None:
        model = self.make_model(
            tokens={
                0: '"value":1e400}',
                1: '"value":12}',
            },
            scores=[[20, 10]],
        )
        generator = ArgumentGenerator.model_construct(model=model)
        result = generator.generate(
            self.prompt, self.function({"value": "number"}),
        )
        self.assertEqual(result, {"value": 12.0})

    def test_non_finite_numbers_are_rejected(self) -> None:
        function = self.function({"value": "number"})
        for value in (float("inf"), float("-inf"), float("nan")):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "must be finite"):
                    ArgumentGenerator.validate_arguments(
                        {"value": value}, function,
                    )

    def test_huge_integer_cannot_overflow_a_number(self) -> None:
        function = self.function({"value": "number"})
        with self.assertRaisesRegex(ValueError, "out of range"):
            ArgumentGenerator.validate_arguments(
                {"value": 10 ** 500}, function,
            )

    def test_empty_parameters_skip_model_generation(self) -> None:
        model = MagicMock()
        generator = ArgumentGenerator.model_construct(model=model)
        result = generator.generate(self.prompt, self.function({}))
        self.assertEqual(result, {})
        model.get_logits_from_input_ids.assert_not_called()
        model.get_path_to_vocab_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
