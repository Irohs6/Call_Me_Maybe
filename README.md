*This project has been created as part of the 42 curriculum by <login1>, <login2>, <login3>.*

# Call Me Maybe

## Description

Call Me Maybe is a Python function-calling project based on the Qwen3-0.6B language model and the provided `llm_sdk`. It reads natural-language requests and function definitions, chooses a suitable function, and extracts its typed arguments. The application writes one JSON output file with the original prompt, function name and parameters.

**Project status:** Work in progress. Function-name selection uses constrained token choices; argument generation currently uses greedy token selection followed by JSON parsing and type validation. Full token-level JSON/schema constraints for arguments are **not yet connected**. The `test-json-structure-methods` branch completes the original `JsonChecker` class. Its single state machine checks JSON object syntax, required keys, and string, number, integer and boolean types. Pass a mapping such as `JsonChecker(parameters={name: definition.type for name, definition in function.parameters.items()})`. This checker is not yet connected to token selection in `ArgumentGenerator`.

## Instructions

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

Install the dependencies:

```bash
make install
# or: uv sync
```

Run with the default input files:

```bash
make run
# or: uv run python -m src
```

Provide alternative input/output paths:

```bash
uv run python -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

The output file defaults to `data/output/function_calls.json`. It is created by the program.

Other Makefile targets: `make debug`, `make clean`, `make lint` and optional `make lint-strict`. The lint target runs flake8 and mypy with the flags required by the subject. Lint dependencies are installed transiently with `uv run --with`; `uv sync` remains the runtime setup.

## Algorithm

1. Read function definitions and prompts as JSON, then validate input objects with Pydantic.
2. Build the Qwen/Hermes function-selection context listing the available function schemas.
3. Encode available function names and choose the next token among their allowed token-ID prefixes, using the logits returned by the SDK.
4. Build a separate context for the selected function's arguments; currently generate the JSON argument object greedily and verify the parsed types after generation.
5. Produce a `FunctionCall` object and serialize all results into one JSON array.

**Planned constrained decoding:** At each argument-generation step, obtain candidate token representations from the vocabulary, reject tokens that would violate the JSON prefix or the expected parameter names/types, and choose among the remaining valid tokens. The `JsonChecker.status()` method distinguishes complete, incomplete and invalid JSON prefixes. This is not yet integrated with `ArgumentGenerator`.

## Design decisions

- `ContextBuilder` assembles the language-model prompts.
- `FunctionSelector` selects only among the provided function names.
- `ArgumentGenerator` handles the parameters in one generation pass.
- `JsonChecker` keeps the original `for t in text` state machine and delegates primitive-prefix validation to its `check_string`, `check_number` and `check_boolean` methods. It enforces required parameter names and types.
- Pydantic models validate input and output structures. The `Small_LLM_Model` wrapper is used through its public methods.

## Example usage

Example natural-language request:

```text
What is the sum of 2 and 3?
```

Expected output shape (illustrative, not a recorded model result):

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {"a": 2.0, "b": 3.0}
  }
]
```

## Testing strategy

Run the JSON checker unit tests:

```bash
uv run python -m unittest discover -s tests
```

Run `make lint` for style and static type checks. Then execute the model against the supplied example inputs and additional, changed function sets. Inspect each output file for valid JSON, exact required keys, schema-compatible parameter types and correct extracted values. Include string escapes, Unicode, missing files, malformed input, empty strings, negative and decimal numbers, boolean values and ambiguous prompts in manual tests.

**Current limitation:** Passing parser unit tests does not establish end-to-end accuracy or full constrained-decoding compliance.

## Performance analysis

The subject targets at least 90% correct function selection and argument extraction, 100% parseable schema-compliant JSON, and under five minutes for the complete evaluation prompt set. These targets have **not been benchmarked for this branch**. Each generated token requires a model logits call; full-vocabulary prefix checks may add overhead. Timing and accuracy should be measured using the final integrated implementation.

## Challenges

- Qwen tokens can span multiple text characters, and UTF-8 characters may be split across tokens.
- JSON escapes such as `\u0041` must be accepted as incomplete prefixes until all four hexadecimal digits are present.
- JSON numbers require a separator to finish; `JsonChecker` handles this inside the original state machine.
- The current post-generation validation cannot substitute for excluding invalid token IDs during generation.

## Resources

- [42 Call Me Maybe subject](docs/subject_en.md)
- [Qwen3-0.6B model](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [uv documentation](https://docs.astral.sh/uv/)

**AI use:** ChatGPT assisted with design discussion, explanations of token-level constrained decoding and JSON escape handling, code review, and drafting parser test cases. Generated suggestions must be reviewed and verified by the project contributors.

**Before submission:** Replace the contributor login placeholders on the first line with the actual 42 logins.
