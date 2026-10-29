"""Unit tests for parameters_models.py, without loading the Qwen weights."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from src.input_models import Function, Type
from src.parameters_models import JsonPrefixChecker, ParametersGenerator


class JsonPrefixCheckerTests(unittest.TestCase):
    def test_valid_structure_and_values(self) -> None:
        checker = JsonPrefixChecker(parameters={
            "name": "string", "a": "number", "count": "integer",
            "ready": "boolean",
        })
        text = r'{"name":"a\"b","a":-1.25e+3,"count":7,"ready":true}'
        self.assertEqual(checker.status(text), 0)
        for index in range(len(text)):
            self.assertNotEqual(checker.status(text[:index]), -2)

    def test_invalid_prefixes(self) -> None:
        checker = JsonPrefixChecker(parameters={"a": "integer"})
        for text in (
            '{"wrong', '{"a":1.2}', '{"a":01}', '{"a":true}',
            '{"a":1,}', '{"a":1}{', '{}',
        ):
            with self.subTest(text=text):
                self.assertEqual(checker.status(text), -2)

    def test_unicode_and_escaped_keys(self) -> None:
        checker = JsonPrefixChecker(parameters={'a"b': "string"})
        self.assertEqual(checker.status(r'{"a\"b":"é"}'), 0)
        self.assertEqual(checker.status(r'{"a\"b":"\u0041"}'), 0)
        checker = JsonPrefixChecker(parameters={"name": "string"})
        self.assertEqual(checker.status(r'{"\u006eame":"ok"}'), 0)


class ParametersGeneratorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    @staticmethod
    def function(parameters: dict[str, str]) -> Function:
        return Function(
            name="fn_test",
            description="Test function",
            parameters={
                name: Type(type=kind)
                for name, kind in parameters.items()
            },
            returns=Type(type="number"),
        )

    def fake_model(
        self,
        tokens: dict[int, str],
        scores: list[list[float]],
    ) -> MagicMock:
        path = Path(self.folder.name) / "vocab.json"
        path.write_text(
            json.dumps({str(i): i for i in tokens if i != 9}),
            encoding="utf-8",
        )
        model = MagicMock()
        model.get_path_to_vocab_file.return_value = str(path)
        model.encode.return_value[0].tolist.return_value = [9]
        model.decode.side_effect = lambda ids: "".join(
            tokens[i] for i in ids
        )
        model.get_logits_from_input_ids.side_effect = lambda ids: (
            scores[len(ids) - 2]
        )
        return model

    def test_reject_invalid_tokens_and_generate_typed_values(self) -> None:
        model = self.fake_model(
            tokens={
                9: "{",
                0: '"bad":2}',
                1: '"a":2',
                2: ',"b":true}',
                3: '"a":"not a number"}',
            },
            scores=[
                [30, 20, 1, 25, 0, 0, 0, 0, 0, 0],
                [1, 1, 15, 1, 0, 0, 0, 0, 0, 0],
            ],
        )
        generated = ParametersGenerator().generate(
            self.function({"a": "number", "b": "boolean"}),
            "Test prompt", model, [100],
        )
        self.assertEqual(generated, {"a": 2.0, "b": True})
        self.assertEqual(model.get_logits_from_input_ids.call_count, 2)

    def test_reject_non_finite_numbers(self) -> None:
        model = self.fake_model(
            tokens={9: "{", 0: '"a":1e999}', 1: '"a":2}'},
            scores=[[30, 10, 0, 0, 0, 0, 0, 0, 0, 0]],
        )
        result = ParametersGenerator().generate(
            self.function({"a": "number"}), "Test", model, [100]
        )
        self.assertEqual(result, {"a": 2.0})

    def test_generate_exact_escaped_string(self) -> None:
        model = self.fake_model(
            tokens={9: "{", 0: r'"text":"Say \"hi\""}'},
            scores=[[20, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        )
        result = ParametersGenerator().generate(
            self.function({"text": "string"}), "Test", model, [100]
        )
        self.assertEqual(result, {"text": 'Say "hi"'})

    def test_empty_schema_skips_generation(self) -> None:
        model = MagicMock()
        result = ParametersGenerator().generate(
            self.function({}), "anything", model, [100]
        )
        self.assertEqual(result, {})
        model.get_logits_from_input_ids.assert_not_called()

    def test_raises_when_no_valid_token_exists(self) -> None:
        model = self.fake_model(
            tokens={9: "{", 0: '"bad":2}'},
            scores=[[15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
        )
        with self.assertRaisesRegex(ValueError, "No token"):
            ParametersGenerator().generate(
                self.function({"a": "integer"}),
                "Test", model, [100],
            )


if __name__ == "__main__":
    unittest.main()
