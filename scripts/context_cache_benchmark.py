"""Compare original context encoding with cached, separated tokenization.

Run: uv run python -m scripts.context_cache_benchmark
Only the public SDK encode method is used; no inference is performed.
"""

from pathlib import Path
from time import perf_counter

from llm_sdk import Small_LLM_Model

from src.model.context_builder import ContextBuilder
from src.model.parser import JsonParser


USER_START = "<|im_start|>user\n"
USER_END = "<|im_end|>"


def split_context(context: str, prompt: str) -> tuple[str, str, str]:
    """Split at Qwen's special-token boundaries, preserving all text."""
    position = context.find(USER_START)
    if position == -1:
        raise ValueError("User message marker not found")

    prompt_start = position + len(USER_START)
    prompt_end = prompt_start + len(prompt)
    if context[prompt_start:prompt_end] != prompt:
        raise ValueError("Prompt does not match the context")

    prefix = context[:position] + "<|im_start|>"
    variable = "user\n" + prompt
    suffix = context[prompt_end:]
    if not suffix.startswith(USER_END):
        raise ValueError("Missing end-of-message marker")
    if prefix + variable + suffix != context:
        raise ValueError("Context reconstruction failed")
    return prefix, variable, suffix


def main() -> int:
    """Check ID equality and measure tokenization with reusable parts."""
    parser = JsonParser()
    functions = parser.load_functions(
        Path("data/input/functions_definition.json")
    )
    prompts = parser.load_prompts(
        Path("data/input/function_calling_tests.json")
    )
    builder = ContextBuilder(functions=functions)
    print("Loading Qwen tokenizer / model through the provided SDK...")
    model = Small_LLM_Model()

    cached_ids: dict[str, list[int]] = {}
    full_time = 0.0
    split_time = 0.0
    checked = 0
    mismatches: list[str] = []

    for prompt in prompts:
        contexts = [("selection", builder.build_function_selection_context(
            prompt
        ))]
        for function in functions:
            contexts.append((
                function.name,
                builder.build_argument_context(prompt, function),
            ))

        for label, context in contexts:
            prefix, variable, suffix = split_context(context, prompt.prompt)
            start = perf_counter()
            full_ids = model.encode(context)[0].tolist()
            full_time += perf_counter() - start

            start = perf_counter()
            for fixed in (prefix, suffix):
                if fixed not in cached_ids:
                    cached_ids[fixed] = model.encode(fixed)[0].tolist()
            variable_ids = model.encode(variable)[0].tolist()
            segmented_ids = (
                cached_ids[prefix] + variable_ids + cached_ids[suffix]
            )
            split_time += perf_counter() - start

            checked += 1
            if full_ids != segmented_ids:
                mismatch = next((
                    i for i, (a, b) in enumerate(
                        zip(full_ids, segmented_ids)
                    ) if a != b
                ), min(len(full_ids), len(segmented_ids)))
                mismatches.append(
                    f"{label}, prompt {prompt.prompt!r}: token {mismatch} "
                    f"(lengths {len(full_ids)} / {len(segmented_ids)})"
                )

    print(f"Contexts compared: {checked}")
    print(f"Matching token sequences: {checked - len(mismatches)}/{checked}")
    print(f"Full-context encode: {full_time:.4f} s")
    print(f"Cached segmented encode: {split_time:.4f} s")
    print(f"Cached fixed fragments: {len(cached_ids)}")
    if mismatches:
        print("MISMATCH: do not enable segmented encoding.")
        for detail in mismatches[:5]:
            print(" -", detail)
        return 1
    print("All token ID sequences are identical on these inputs.")
    print("These times measure encoding, NOT LLM inference or total runtime.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
