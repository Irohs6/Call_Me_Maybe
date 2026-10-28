"""Tests for the alternative one-value-at-a-time generator."""

import json
import unittest

from src.argument_generator import ArgumentGenerator
from src.model.models import Function, Prompt, Type


class TokenIds:
    """Match the minimal encode()[0].tolist() interface."""

    def __init__(self, text: str) -> None:
        self.ids = [ord(char) for char in text]

    def tolist(self) -> list[int]:
        return self.ids


class FakeModel:
    """Return scripted token streams without loading the actual LLM."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.index = 0
        self.context_ids: list[int] = []
        self.contexts: list[str] = []

    def encode(self, text: str) -> list[TokenIds]:
        self.contexts.append(text)
        self.context_ids = TokenIds(text).tolist()
        self.index = len(self.contexts) - 1
        return [TokenIds(text)]

    def get_logits_from_input_ids(
        self,
        token_ids: list[int],
    ) -> list[float]:
        response = self.responses[self.index]
        generated_count = len(token_ids) - len(self.context_ids)
        char = response[generated_count]
        logits = [-100.0] * 128
        logits[ord(char)] = 100.0
        return logits

    @staticmethod
    def decode(token_ids: list[int]) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class IncrementalGeneratorTests(unittest.TestCase):
    @staticmethod
    def make_function(parameters: dict[str, str]) -> Function:
        return Function(
            name="fn_test",
            description="Test function",
            parameters={
                name: Type(type=type_value)
                for name, type_value in parameters.items()
            },
            returns=Type(type="string"),
        )

    @staticmethod
    def generator(model: FakeModel) -> ArgumentGenerator:
        # Bypass Pydantic SDK type check for this isolated fake.
        return ArgumentGenerator.model_construct(model=model)

    def test_build_value_prefix(self) -> None:
        prefix = ArgumentGenerator.build_value_prefix({}, "a")
        self.assertEqual(prefix, '{"a": ')
        prefix = ArgumentGenerator.build_value_prefix({"a": 2}, "b")
        self.assertEqual(prefix, '{"a": 2, "b": ')
        prefix = ArgumentGenerator.build_value_prefix(
            {"a": 'a "quoted" value'}, "b"
        )
        self.assertEqual(
            prefix,
            json.dumps({"a": 'a "quoted" value', "b": ""})[:-3],
        )

    def test_numeric_values_are_built_separately(self) -> None:
        model = FakeModel(["2,", "3}"])
        generator = self.generator(model)
        function = self.make_function({"a": "number", "b": "number"})
        result = generator.generate(
            Prompt(prompt="What is 2 + 3?"), function
        )
        self.assertEqual(result, {"a": 2.0, "b": 3.0})
        self.assertEqual(len(model.contexts), 2)
        self.assertTrue(model.contexts[0].endswith('"a": '))
        self.assertTrue(model.contexts[1].endswith('"a": 2.0, "b": '))

    def test_string_keeps_quotes_as_python_data(self) -> None:
        model = FakeModel([r'"Hello \"Iroh\""'])
        function = self.make_function({"name": "string"})
        result = self.generator(model).generate(
            Prompt(prompt="Greet Iroh"), function
        )
        self.assertEqual(result, {"name": 'Hello "Iroh"'})
        self.assertEqual(
            json.loads(json.dumps(result)), result
        )

    def test_boolean_value(self) -> None:
        model = FakeModel(["false"])
        function = self.make_function({"enabled": "boolean"})
        result = self.generator(model).generate(
            Prompt(prompt="Disable option"), function
        )
        self.assertEqual(result, {"enabled": False})

    def test_numeric_requires_separator(self) -> None:
        self.assertIsNone(
            ArgumentGenerator.read_generated_value("123", "number")
        )
        self.assertEqual(
            ArgumentGenerator.read_generated_value("123,", "number"), 123
        )
        self.assertIsNone(
            ArgumentGenerator.read_generated_value("1e", "number")
        )

    def test_boolean_does_not_count_as_number(self) -> None:
        self.assertIsNone(
            ArgumentGenerator.read_generated_value("true,", "number")
        )

    def test_empty_parameters(self) -> None:
        model = FakeModel([])
        function = self.make_function({})
        result = self.generator(model).generate(
            Prompt(prompt="Do something"), function
        )
        self.assertEqual(result, {})
        self.assertEqual(model.contexts, [])


if __name__ == "__main__":
    unittest.main()
