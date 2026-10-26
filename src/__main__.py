"""Command-line entry point for function calling."""

import json
from argparse import ArgumentParser
from pathlib import Path

from llm_sdk import Small_LLM_Model

from .application import Application
from .argument_generator import FunctionArgumentGenerator
from .constrained_decoder import ConstrainedDecoder
from .function_selector import FunctionSelector
from .models import FunctionCall


CYAN = "\033[96m"
RESET = "\033[0m"


def main() -> None:
    """Load inputs, generate function calls, and write JSON output."""
    parser = ArgumentParser()

    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path(
            "data/input/functions_definition.json"
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/input/function_calling_tests.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/output/function_calls.json"
        ),
    )

    args = parser.parse_args()

    application = Application(
        functions_path=args.functions_definition,
        input_path=args.input,
    )

    try:
        functions, prompts = application.load()
        model = Small_LLM_Model()
        decoder = ConstrainedDecoder(model=model)
        function_selector = FunctionSelector(
            decoder=decoder,
            functions=functions,
        )
        argument_generator = FunctionArgumentGenerator(
            decoder=decoder
        )

        results: list[FunctionCall] = []

        for prompt in prompts:
            selected_function = function_selector.select(
                prompt
            )
            argument_values = argument_generator.generate(
                prompt=prompt,
                function=selected_function,
            )
            results.append(
                FunctionCall(
                    prompt=prompt.prompt,
                    name=selected_function.name,
                    parameters=argument_values,
                )
            )

        output = [
            result.model_dump(mode="json")
            for result in results
        ]

        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with args.output.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                output,
                file,
                indent=4,
                ensure_ascii=False,
            )

        print(
            f"{CYAN}Dict answer final:{RESET} "
            f"{output}"
        )

    except (ValueError, OSError) as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
