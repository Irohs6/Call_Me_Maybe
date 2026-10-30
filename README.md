*This project has been created as part of the 42 curriculum by gacattan.*

# Call Me Maybe

## Description

**Call Me Maybe** is a Python function-calling project using **Qwen/Qwen3-0.6B** through the provided `llm_sdk`. Given natural-language prompts and a JSON file describing available functions, the program selects the appropriate function and generates its arguments.

The goal is to produce **schema-compliant JSON through constrained decoding**, rather than relying on the model to spontaneously output valid JSON.

The program writes one JSON array containing the original prompt, the selected function name, and its arguments. The implementation supports `string`, `number`, `integer`, and `boolean` parameters.

## Instructions

### Requirements and installation

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/)
- The provided `llm_sdk` package and its model dependencies

From the repository root:

```bash
uv sync
# or
make install
```

The first model run may need to download Qwen model files.

### Run

```bash
uv run python -m src
# or
make run
```

Default files:

- Functions: `data/input/functions_definition.json`
- Prompts: `data/input/function_calling_tests.json`
- Output: `data/output/function_calls.json`

Use custom file paths:

```bash
uv run python -m src \
  --functions_definition path/to/functions.json \
  --input path/to/prompts.json \
  --output path/to/result.json
```

The application creates the parent directory for the output when needed.

Other Makefile targets are `make debug`, `make clean`, `make lint`, and `make lint-strict`.

## Algorithm

The implementation separates **function selection** from **argument generation**, using a shared LLM instance.

### 1. Read and validate inputs

`JsonParser` loads both JSON files, validates their contents using Pydantic, and rejects malformed JSON, duplicate object keys, duplicate function names, missing files, and unsupported argument types. Validation errors identify the affected function or prompt.

### 2. Select a function with constrained decoding

`ContextBuilder` constructs a Qwen/Hermes-style context containing the available function definitions and the user request.

`FunctionSelector` encodes the available function names (including a terminating newline) and uses the LLM logits to select only token IDs that can continue one of those names. After each selection, incompatible function candidates are removed. The completed token sequence identifies the chosen function.

Function selection is **model-driven**; it does not use hardcoded function-name rules or keyword matching.

### 3. Generate all arguments as one JSON object

`ContextBuilder.build_argument_context()` prepares a context for the selected function and all its parameters.

`ArgumentGenerator` reads the vocabulary file exposed by the SDK and caches the token IDs. On each generation step, it:

1. Requests logits from the LLM.
2. Considers vocabulary tokens in descending score order.
3. Decodes a candidate continuation and calls `JsonChecker.status()`.
4. Rejects a token when it would make the JSON prefix invalid.
5. Accepts a valid continuation, or parses and validates a complete JSON object.

The JSON checker returns:

| Status | Meaning |
|---|---|
| `-2` | Invalid prefix |
| `-1` | Valid but incomplete prefix |
| `0` | Complete object |

`JsonChecker` is a state-machine-based JSON prefix parser. It delegates string, number, and boolean parsing to `JsonStringParser`, `JsonNumberParser`, and `JsonBooleanParser`, all located in `src/json_checker.py`. It handles JSON structure, required parameter names, duplicate or unknown keys, escape sequences, Unicode `\uXXXX` escapes, numeric syntax, and the expected primitive types.

The generator validates the final object again: it must contain exactly the required keys and values of the expected types. Non-finite numbers are rejected; if a completed candidate is invalid, generation can try the next candidate. Argument generation is limited to **256 generated tokens** per non-empty argument object.

### 4. Write the result

`FunctionCall` models the output structure. The CLI writes a single JSON array, using strict JSON serialization (`allow_nan=False`). Errors are reported with a readable message and a nonzero exit status.

## Design decisions

- **No hardcoded function logic:** the same selection and generation pipeline handles function definitions supplied at runtime.
- **One argument generation pass:** numbers, strings, integers, and booleans are generated together in a schema-constrained JSON object.
- **Dedicated state machine:** the JSON checker handles incomplete prefixes during generation rather than checking syntax only at the end.
- **Pydantic for data models:** input structure and output fields are represented with explicit Python models.
- **SDK public methods only:** the code uses the provided model wrapper instead of accessing its internal implementation.
- **Cached vocabulary IDs:** avoid reloading the same vocabulary for every prompt.
- **Explicit error handling:** invalid inputs and generation failures produce diagnostic messages.

## Example usage

For the request:

```text
What is the sum of 2 and 3?
```

An expected output entry looks like this:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

This is an illustrative output format, not a claim about a separate benchmark.

## Testing strategy

Run all unit tests:

```bash
uv run python -m unittest discover -s tests -v
```

Run the repository's lint targets:

```bash
make lint
make lint-strict
```

**Latest reported local checks:**

- **46/46 unit tests passed** (`unittest`, 0.029 seconds).
- **Flake8 passed** on `src`.
- **Mypy strict passed** on all 10 source files under `src`, using the flags configured in the Makefile.

The tests cover the JSON prefix checker, argument generation with a simulated model, function-selection failure cases, duplicate JSON keys, input validation, error handling, and rejection of non-finite numeric arguments.

Unit tests with a mocked model validate structural behavior; they do not replace real Qwen tests. End-to-end evaluation should also cover changed function definitions, multiple parameters, booleans, Unicode, quoted strings, escaped characters, long values, and malformed input files.

## Performance analysis

The subject sets targets of **90% or better correct function selection and argument extraction**, **100% parseable schema-compliant JSON**, and **less than five minutes for the evaluated prompt set**.

In one reported local Qwen run, **11 prompts were processed in approximately 3 minutes 29 seconds**, with **11/11 expected outputs matched** for that particular test set. This measurement is specific to those prompts, functions, hardware, and model setup; it does not establish accuracy or timing on a different evaluation set.

The implementation caches vocabulary IDs and filters invalid token candidates before accepting them. Its cost still depends on model inference, the number and length of generated tokens, and how many candidate prefixes must be checked. No general speedup or accuracy result is claimed without comparative measurements.

## Challenges faced

- **Constrained function selection:** tokenized function names can share prefixes; the selector must preserve only continuations belonging to the available functions.
- **Incomplete JSON during generation:** a valid prefix is not necessarily a complete document; the checker distinguishes incomplete, invalid, and complete states.
- **Strings and escapes:** JSON quotes, backslashes, control characters, and `\uXXXX` sequences need dedicated parsing states.
- **Typed numbers:** JSON number syntax includes signs, decimals, and exponents. The generator also rejects values that cannot be represented as finite numbers.
- **Tokenization:** a decoded token can represent multiple characters, while some Unicode characters may require multiple tokens. Partial UTF-8 decoding remains a case requiring additional real-tokenizer testing.
- **Runtime and reliability:** vocabulary caching, candidate filtering, input validation, and error messages were added to reduce repeated work and make failures understandable.

## Resources

- [42 Call Me Maybe subject](docs/subject_en.md)
- [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Python JSON documentation](https://docs.python.org/3/library/json.html)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [uv documentation](https://docs.astral.sh/uv/)
- [Python unittest documentation](https://docs.python.org/3/library/unittest.html)

**AI usage:** ChatGPT was used for design discussions, explanations of constrained decoding and JSON parsing, review of edge cases, assistance with test cases, and documentation. Implementation choices and generated suggestions were reviewed and tested against the project requirements.
