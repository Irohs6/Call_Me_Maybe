"""Generate schema-compliant JSON arguments with constrained decoding."""

import heapq
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, field_validator

from .input_models import Function


class JsonPrefixChecker(BaseModel):
    """Validate complete JSON objects and their unfinished prefixes."""

    parameters: dict[str, str]

    @field_validator("parameters")
    @classmethod
    def check_types(cls, parameters: dict[str, str]) -> dict[str, str]:
        allowed = {"string", "number", "integer", "boolean"}
        for name, type_name in parameters.items():
            if not name or type_name not in allowed:
                raise ValueError(f"Invalid parameter: {name!r} ({type_name})")
        return parameters

    @staticmethod
    def _spaces(text: str, position: int) -> int:
        while position < len(text) and text[position] in " \t\r\n":
            position += 1
        return position

    @staticmethod
    def _string(text: str, position: int) -> tuple[int, int, str]:
        """Return (status, next position, decoded value)."""
        if position == len(text):
            return -1, position, ""
        if text[position] != '"':
            return -2, position, ""

        start = position
        position += 1
        while position < len(text):
            char = text[position]
            if char == '"':
                position += 1
                try:
                    value = json.loads(text[start:position])
                except json.JSONDecodeError:
                    return -2, position, ""
                return 0, position, value
            if ord(char) < 32:
                return -2, position, ""
            if char == "\\":
                position += 1
                if position == len(text):
                    return -1, position, ""
                if text[position] == "u":
                    for offset in range(1, 5):
                        index = position + offset
                        if index == len(text):
                            return -1, index, ""
                        if text[index] not in "0123456789abcdefABCDEF":
                            return -2, index, ""
                    position += 4
                elif text[position] not in '"\\/bfnrt':
                    return -2, position, ""
            position += 1
        return -1, position, ""

    @staticmethod
    def _number(text: str, position: int, integer: bool) -> tuple[int, int]:
        """Validate JSON number syntax, requiring a following separator."""
        size = len(text)
        if position == size:
            return -1, position
        if text[position] == "-":
            position += 1
            if position == size:
                return -1, position

        if text[position] == "0":
            position += 1
            if position < size and text[position].isdigit():
                return -2, position
        elif text[position] in "123456789":
            position += 1
            while position < size and text[position] in "0123456789":
                position += 1
        else:
            return -2, position

        if not integer and position < size and text[position] == ".":
            position += 1
            if position == size:
                return -1, position
            if text[position] not in "0123456789":
                return -2, position
            while position < size and text[position] in "0123456789":
                position += 1

        if not integer and position < size and text[position] in "eE":
            position += 1
            if position == size:
                return -1, position
            if text[position] in "+-":
                position += 1
                if position == size:
                    return -1, position
            if text[position] not in "0123456789":
                return -2, position
            while position < size and text[position] in "0123456789":
                position += 1

        if position == size:
            return -1, position
        if text[position] not in " \t\r\n,}":
            return -2, position
        return 0, position

    @staticmethod
    def _boolean(text: str, position: int) -> tuple[int, int]:
        """Check the prefix of a true/false literal."""
        expected = "true" if text[position:position + 1] == "t" else "false"
        for char in expected:
            if position == len(text):
                return -1, position
            if text[position] != char:
                return -2, position
            position += 1
        if position < len(text) and text[position] not in " \t\r\n,}":
            return -2, position
        return 0, position

    def _possible_key(self, text: str, used: set[str]) -> bool:
        """Reject impossible plain prefixes; defer unfinished escapes."""
        plain = text.split("\\", 1)[0]
        return any(
            name not in used and name.startswith(plain)
            for name in self.parameters
        )

    def status(self, text: str) -> int:
        """0: complete; -1: valid unfinished prefix; -2: impossible."""
        position = self._spaces(text, 0)
        if position == len(text):
            return -1
        if text[position] != "{":
            return -2
        position += 1
        used: set[str] = set()
        need_key = False

        while True:
            position = self._spaces(text, position)
            if position == len(text):
                return -1
            if text[position] == "}" and not need_key:
                position += 1
                position = self._spaces(text, position)
                if position != len(text):
                    return -2
                return 0 if used == self.parameters.keys() else -2

            if text[position] != '"' or len(used) == len(self.parameters):
                return -2
            start = position
            status, position, key = self._string(text, position)
            if status == -2:
                return -2
            if status == -1:
                return (-1 if self._possible_key(text[start + 1:], used)
                        else -2)
            if key not in self.parameters or key in used:
                return -2
            used.add(key)

            position = self._spaces(text, position)
            if position == len(text):
                return -1
            if text[position] != ":":
                return -2
            position = self._spaces(text, position + 1)
            if position == len(text):
                return -1

            type_name = self.parameters[key]
            if type_name == "string":
                result, position, _ = self._string(text, position)
            elif type_name == "boolean":
                result, position = self._boolean(text, position)
            else:
                result, position = self._number(
                    text, position, integer=type_name == "integer"
                )
            if result != 0:
                return result

            position = self._spaces(text, position)
            if position == len(text):
                return -1
            if text[position] == "}":
                position += 1
                position = self._spaces(text, position)
                if position != len(text):
                    return -2
                return 0 if used == self.parameters.keys() else -2
            if text[position] != "," or len(used) == len(self.parameters):
                return -2
            position += 1
            need_key = True


class ParametersGenerator(BaseModel):
    """Select the highest-scoring token that preserves the JSON schema."""

    MAX_TOKENS: ClassVar[int] = 512

    @staticmethod
    @lru_cache(maxsize=4)
    def _vocabulary_ids(path: str) -> tuple[int, ...]:
        try:
            with Path(path).open(encoding="utf-8") as file:
                vocabulary = json.load(file)
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot read vocabulary: {path}") from error
        if not isinstance(vocabulary, dict):
            raise ValueError("Vocabulary must be a JSON object")
        ids = {
            value for value in vocabulary.values()
            if type(value) is int and value >= 0
        }
        if not ids:
            raise ValueError("Vocabulary has no valid token IDs")
        return tuple(sorted(ids))

    @staticmethod
    def _validate(arguments: object, func: Function) -> dict[str, Any]:
        if not isinstance(arguments, dict):
            raise ValueError("Generated arguments must be an object")
        if set(arguments) != set(func.parameters):
            raise ValueError("Generated argument names do not match")

        result: dict[str, Any] = {}
        for name, definition in func.parameters.items():
            value = arguments[name]
            kind = definition.type
            if kind == "string":
                valid = isinstance(value, str)
            elif kind == "boolean":
                valid = type(value) is bool
            elif kind == "integer":
                valid = type(value) is int
            elif kind == "number":
                valid = type(value) in (int, float)
                if valid:
                    value = float(value)
                    valid = math.isfinite(value)
            else:
                raise ValueError(f"Unsupported type: {kind}")
            if not valid:
                raise ValueError(f"Invalid value for {name!r}: {kind}")
            result[name] = value
        return result

    def generate(
        self,
        func: Function,
        prompt: str,
        llm: Small_LLM_Model,
        tokenized_context: list[int],
    ) -> dict[str, Any]:
        """Generate one complete JSON object of the required parameters.

        The orchestrator already includes `prompt` in `tokenized_context`.
        """
        if not func.parameters:
            return {}

        checker = JsonPrefixChecker(parameters={
            name: definition.type
            for name, definition in func.parameters.items()
        })
        try:
            vocabulary_ids = self._vocabulary_ids(
                llm.get_path_to_vocab_file()
            )
        except Exception as error:
            raise ValueError("Could not load model vocabulary") from error

        # The orchestrator's context ends just before the arguments object.
        generated_ids = llm.encode("{")[0].tolist()
        if llm.decode(generated_ids) != "{":
            raise ValueError("Could not encode JSON opening brace")

        for _ in range(self.MAX_TOKENS):
            logits = llm.get_logits_from_input_ids(
                tokenized_context + generated_ids
            )
            options = [
                (-logits[token_id], token_id)
                for token_id in vocabulary_ids
                if token_id < len(logits)
                and math.isfinite(logits[token_id])
            ]
            heapq.heapify(options)
            previous_text = llm.decode(generated_ids)

            while options:
                _, token_id = heapq.heappop(options)
                next_ids = generated_ids + [token_id]
                text = llm.decode(next_ids)
                if text == previous_text and not text.endswith("\ufffd"):
                    continue

                state = checker.status(text)
                if state == -2 and text.endswith("\ufffd"):
                    # The tokenizer may expose a partial UTF-8 character.
                    if checker.status(text.rstrip("\ufffd")) == -1:
                        state = -1
                if state == -2:
                    continue

                if state == 0:
                    try:
                        arguments = json.loads(text)
                        return self._validate(arguments, func)
                    except (ValueError, OverflowError):
                        continue
                generated_ids.append(token_id)
                break
            else:
                raise ValueError("No token can continue the JSON")

        raise ValueError("JSON generation exceeded the token limit")
