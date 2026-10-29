"""Regression probes for Xyf-cyakisan/Call_Me_Maybe, no LLM download.

Usage: python docs/test_xyf_call_me_maybe.py /path/to/Xyf-cyakisan/Call_Me_Maybe
The default target is the current directory.
"""
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

TARGET = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
sys.argv = [sys.argv[0]]
sys.path.insert(0, str(TARGET))

# The SDK model is not needed for parser/decoder behavior tests.
sdk = types.ModuleType("llm_sdk")
sdk.Small_LLM_Model = type("Small_LLM_Model", (), {})
sys.modules["llm_sdk"] = sdk

from src.JSONParser import JSONParser  # noqa: E402
from src.FunctionSelector import FunctionSelector  # noqa: E402
from src.input_models import Function  # noqa: E402
from src.parameters_models import JsonPrefixChecker, ParametersGenerator  # noqa: E402


class Encoded:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        assert index == 0
        return self

    def tolist(self):
        return list(self.values)


class FakeModel:
    def __init__(self, tokens, scores, vocab_file=None, fused_newline=False):
        self.fused_newline = fused_newline
        self.tokens = tokens
        self.scores = scores
        self.vocab_file = vocab_file
        self.calls = 0

    def encode(self, text):
        if text == "{":
            return Encoded([50])
        if text.endswith("\n") and text[:-1] == "fn_test":
            return Encoded([5] if self.fused_newline else [5, 6])
        return Encoded([99])

    def decode(self, ids):
        return "".join(self.tokens.get(i, "") for i in ids)

    def get_logits_from_input_ids(self, ids):
        index = min(self.calls, len(self.scores) - 1)
        self.calls += 1
        return list(self.scores[index])

    def get_path_to_vocab_file(self):
        return str(self.vocab_file)


def function_dict(arg_type="number"):
    return {
        "name": "fn_test", "description": "Test a value",
        "parameters": {"value": {"type": arg_type}},
        "returns": {"type": "number"},
    }


class InputProbes(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.functions = self.root / "functions.json"
        self.prompts = self.root / "prompts.json"
        self.output = self.root / "out.json"
        self.functions.write_text(json.dumps([function_dict()]), encoding="utf-8")
        self.prompts.write_text('[{"prompt":"Test"}]', encoding="utf-8")

    def parser(self):
        return JSONParser(
            func_definition_file_name=str(self.functions),
            calling_tests_file_name=str(self.prompts),
            calls_json_file_name=str(self.output),
        )

    def test_valid_definitions(self):
        funcs, prompts = self.parser().parse()
        self.assertEqual(funcs[0].name, "fn_test")
        self.assertEqual(prompts[0].prompt, "Test")

    def test_duplicate_parameter_keys(self):
        text = self.functions.read_text().replace(
            '"value": {"type": "number"}',
            '"value": {"type": "number"}, "value": {"type": "number"}',
        )
        self.functions.write_text(text)
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.parser().parse()

    def test_duplicate_function_names(self):
        self.functions.write_text(json.dumps([function_dict(), function_dict()]))
        with self.assertRaises(ValueError):
            self.parser().parse()

    def test_missing_function_file(self):
        self.functions.unlink()
        with self.assertRaises(FileNotFoundError):
            self.parser().parse()

    def test_invalid_json(self):
        self.functions.write_text("[")
        with self.assertRaises(json.JSONDecodeError):
            self.parser().parse()

    @unittest.expectedFailure
    def test_wrong_root_type_rejected_cleanly(self):
        """Actual result is TypeError; CLI does not catch it."""
        self.functions.write_text('{"unexpected":"object"}')
        with self.assertRaises(ValueError):
            self.parser().parse()

    @unittest.expectedFailure
    def test_empty_functions_rejected_early(self):
        """An empty function list is currently accepted by parser."""
        self.functions.write_text("[]")
        with self.assertRaises(ValueError):
            self.parser().parse()

    @unittest.expectedFailure
    def test_unsupported_argument_type_rejected_early(self):
        """Input schema accepts type 'object'; checker rejects later."""
        self.functions.write_text(json.dumps([function_dict("object")]))
        with self.assertRaises(ValueError):
            self.parser().parse()

    @unittest.expectedFailure
    def test_output_not_truncated_during_input_validation(self):
        """Parser truncates an existing output before model execution."""
        self.output.write_text("PRESERVE_ME", encoding="utf-8")
        self.parser().parse()
        self.assertEqual(self.output.read_text(), "PRESERVE_ME")


class DecodingProbes(unittest.TestCase):
    def function(self, arg_type="number"):
        return Function.model_validate(function_dict(arg_type))

    def test_json_prefix_checker_valid(self):
        checker = JsonPrefixChecker(parameters={"value": "number"})
        self.assertEqual(checker.status('{"value":12.5}'), 0)
        self.assertEqual(checker.status('{"value":'), -1)
        self.assertEqual(checker.status('{"value":true}'), -2)

    def test_json_prefix_checker_rejects_duplicate(self):
        checker = JsonPrefixChecker(parameters={"value": "integer"})
        self.assertEqual(checker.status('{"value":2,"value":3}'), -2)

    def test_generator_skips_overflow_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            vocab = Path(root) / "vocab.json"
            vocab.write_text(json.dumps({"large": 0, "normal": 1}))
            logits = [float("-inf")] * 51
            logits[0], logits[1] = 20, 10
            model = FakeModel(
                {50: "{", 0: '"value":1e400}', 1: '"value":12}'},
                [logits], vocab,
            )
            result = ParametersGenerator().generate(self.function(), model, [99])
            self.assertEqual(result, {"value": 12.0})

    def test_selector_valid_name(self):
        scores1 = [float("-inf")] * 16
        scores1[5] = 10
        scores2 = [float("-inf")] * 16
        scores2[6] = 10
        model = FakeModel({5: "fn_test", 6: "\n"}, [scores1, scores2])
        selected = FunctionSelector().get_func_name(
            model, "context", [self.function()]
        )
        self.assertEqual(selected, "fn_test")

    @unittest.expectedFailure
    def test_selector_name_and_newline_in_one_token(self):
        """A tokenizer may fuse the function suffix and newline in one ID."""
        scores = [float("-inf")] * 16
        scores[5] = 10
        model = FakeModel({5: "fn_test\n"}, [scores], fused_newline=True)
        self.assertEqual(
            FunctionSelector().get_func_name(model, "context", [self.function()]),
            "fn_test",
        )

    @unittest.expectedFailure
    def test_selector_rejects_nan_before_decoding(self):
        """A NaN permitted logit should fail before consuming a forbidden ID."""
        scores = [float("-inf")] * 16
        scores[5] = float("nan")
        model = FakeModel({5: "fn_test", 6: "\n"}, [scores])
        with self.assertRaises(ValueError):
            FunctionSelector().get_func_name(model, "context", [self.function()])
        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    print(f"Target repository: {TARGET}")
    unittest.main(verbosity=2)
