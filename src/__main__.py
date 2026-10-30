"""Command-line entry point for function calling."""

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import TextIO


def colored(text: str, code: str, stream: TextIO) -> str:
    """Use terminal colors only when supported."""
    if stream.isatty() and "NO_COLOR" not in os.environ:
        return f"\033[{code}m{text}\033[0m"
    return text


def print_error(message: str) -> None:
    """Display one readable error without an automatic traceback."""
    print(f"{colored('Error:', '31', sys.stderr)} {message}",
          file=sys.stderr)


def show_progress(completed: int, total: int) -> None:
    """Show completed prompts only in interactive terminals."""
    if total == 0 or not sys.stderr.isatty():
        return
    width = 20
    filled = width * completed // total
    bar = "#" * filled + "-" * (width - filled)
    ending = "\n" if completed == total else ""
    print(
        f"\rGenerating [{bar}] {completed}/{total}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


def main() -> int:
    """Run the command-line function-calling pipeline."""
    parser = ArgumentParser()
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=Path("data/input/functions_definition.json"),
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/input/function_calling_tests.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/function_calls.json"),
    )
    args = parser.parse_args()

    try:
        # Keep SDK imports inside the handler: missing dependencies must
        # produce a readable error rather than a startup traceback.
        from .argument_generator import ArgumentGenerator
        from .function_selector import FunctionSelector
        from .model.app import Application
        from .model.context_builder import ContextBuilder
        from .model.models import FunctionCall
        from llm_sdk import Small_LLM_Model

        app = Application(
            functions_path=args.functions_definition,
            input_path=args.input,
            output_path=args.output,
        )
        app.run()

        context_builder = ContextBuilder(functions=app.functions)
        model = Small_LLM_Model()
        function_selector = FunctionSelector(
            model=model,
            functions=app.functions,
        )
        argument_generator = ArgumentGenerator(model=model)

        results: list[FunctionCall] = []
        show_progress(0, len(app.prompts))
        for i, prompt in enumerate(app.prompts, start=1):
            try:
                context = context_builder.build_function_selection_context(
                    prompt
                )
                selected_function = function_selector.select(context)
                parameters = argument_generator.generate(
                    prompt, selected_function,
                )
                results.append(FunctionCall(
                    prompt=prompt.prompt,
                    name=selected_function.name,
                    parameters=parameters,
                ))
            except Exception as error:
                if sys.stderr.isatty():
                    print(file=sys.stderr)
                raise RuntimeError(f"Prompt #{i}: {error}") from error
            show_progress(i, len(app.prompts))

        output = [result.model_dump(mode="python") for result in results]
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as file:
                json.dump(output, file, indent=4, allow_nan=False)
        except OSError as error:
            raise OSError(
                f"Cannot write output '{args.output}': {error}"
            ) from error

        return 0
    except ModuleNotFoundError as error:
        print_error(
            f"Missing Python module '{error.name}'. "
            "Check the SDK dependencies in pyproject.toml and run uv sync."
        )
    except ImportError as error:
        print_error(f"Cannot import a required module: {error}")
    except (ValueError, OSError, RuntimeError) as error:
        print_error(str(error))
    except KeyboardInterrupt:
        print_error("Execution interrupted")
        return 130
    except Exception as error:
        print_error(f"Unexpected {type(error).__name__}: {error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
