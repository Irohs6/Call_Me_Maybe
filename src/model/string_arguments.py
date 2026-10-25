"""Incremental JSON grammar for an object of required string arguments."""

import json
from typing import Literal

from pydantic import BaseModel


class StringArgumentsGrammar(BaseModel):
    """Accept only known, unique argument keys with JSON string values."""

    parameters: list[str]

    @staticmethod
    def string_end(text: str, start: int) -> int:
        """Return the closing position, -1 for incomplete, -2 for invalid."""
        if text[start] != '"':
            return -2
        position = start + 1
        while position < len(text):
            character = text[position]
            if character == '"':
                return position + 1
            if ord(character) < 32:
                return -2
            if character == "\\":
                position += 1
                if position == len(text):
                    return -1
                escape = text[position]
                if escape == "u":
                    digits = text[position + 1:position + 5]
                    if any(
                        digit not in "0123456789abcdefABCDEF"
                        for digit in digits
                    ):
                        return -2
                    if len(digits) < 4:
                        return -1
                    position += 4
                elif escape not in '"\\/bfnrt':
                    return -2
            position += 1
        return -1

    def status(self, text: str) -> Literal["invalid", "incomplete", "complete"]:
        """Validate a prefix, allowing a token to extend past the final brace."""
        position = 0
        remaining = list(self.parameters)
        length = len(text)

        def whitespace() -> None:
            nonlocal position
            while position < length and text[position] in " \t\r\n":
                position += 1

        whitespace()
        if position == length:
            return "incomplete"
        if text[position] != "{":
            return "invalid"
        position += 1

        while True:
            whitespace()
            if position == length:
                return "incomplete"
            if not remaining:
                return "complete" if text[position] == "}" else "invalid"

            key = None
            tail = text[position:]
            for candidate in remaining:
                encoded = json.dumps(candidate, ensure_ascii=False)
                if tail.startswith(encoded):
                    key = candidate
                    position += len(encoded)
                    break
                if encoded.startswith(tail):
                    return "incomplete"
            if key is None:
                return "invalid"

            whitespace()
            if position == length:
                return "incomplete"
            if text[position] != ":":
                return "invalid"
            position += 1
            whitespace()
            if position == length:
                return "incomplete"

            end = self.string_end(text, position)
            if end < 0:
                return "incomplete" if end == -1 else "invalid"
            position = end
            remaining.remove(key)
            whitespace()
            if position == length:
                return "incomplete"
            if not remaining:
                return "complete" if text[position] == "}" else "invalid"
            if text[position] != ",":
                return "invalid"
            position += 1
