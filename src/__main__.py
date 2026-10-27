from argparse import ArgumentParser
from pathlib import Path
from json.decoder import JSONDecodeError

from .argument_generator import ArgumentGenerator
from .model.app import Application
from .model.context_builder import ContextBuilder
from .function_selector import FunctionSelector
from .model.models import FunctionCall
from llm_sdk import Small_LLM_Model
import json

GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"


def main() -> None:
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

    app = Application(
        functions_path=args.functions_definition,
        input_path=args.input,
        output_path=args.output,
    )

    try:
        app.run()

        context_builder = ContextBuilder(
            functions=app.functions
        )

        model = Small_LLM_Model()

        function_selector = FunctionSelector(
            model=model,
            functions=app.functions,
        )
        argument_generator = ArgumentGenerator(
            model=model,
        )

        results = []
        for i, prompt in enumerate(app.prompts, start=1):
            context = context_builder.build_with_prompt(prompt)

            selected_function = function_selector.select(context)
            parameters = argument_generator.generate(
                prompt,
                selected_function,
            )
            function_call = FunctionCall(
                prompt=prompt.prompt,
                name=selected_function.name,
                parameters=parameters
            )
            results.append(function_call)
        output = [result.model_dump(mode="json") for result in results]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as file:
            json.dump(output, file, indent=4)
        print(f"{CYAN}Dict answer final:{RESET} {output}")
    except (ValueError, FileNotFoundError, JSONDecodeError) as error:
        print(f"Error: {error}")
        return


if __name__ == "__main__":
    main()
