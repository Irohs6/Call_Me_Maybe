from llm_sdk import Small_LLM_Model
from argparse import ArgumentParser
import json
from parser import JSONParser


def main() -> None:
    json_parse = JSONParser()
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--functions_definition",
        type=str,
        default="./data/input/functions_definition.json",
    )
    arg_parser.add_argument(
        "--input",
        type=str,
        default="./data/input/function_calling_tests.json",
    )

    arg_parser.add_argument(
        "--output",
        type=str,
        default="./data/output/function_calls.json",
    )

    args = arg_parser.parse_args()

    function_definition_path = args.functions_definition
    input_path = args.input
    output_path = args.output

    with open(function_definition_path, "r") as file:
        try:
            functions_definition = json.load(file)
            json_parse.parse(functions_definition)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {function_definition_path}: {e}")

    with open(input_path, "r") as file:
        try:
            function_calling_tests = json.load(file)
            json_parse.parse(function_calling_tests)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON from {input_path}: {e}")

    model = Small_LLM_Model()



    # You can now use `functions_definition` and `function_calling_tests` with the model
if __name__ == "__main__":
    main()
