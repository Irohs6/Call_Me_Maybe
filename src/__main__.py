from argparse import ArgumentParser
from pathlib import Path

from .app import Application


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

    except ValueError as error:
        print(f"Error: {error}")


if __name__ == "__main__":
    main()
