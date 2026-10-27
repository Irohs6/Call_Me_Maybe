"""Validate JSON prefixes using the Qwen byte-level vocabulary."""

import codecs
import json
from pathlib import Path

from pydantic import BaseModel


class JsonArgumentsGrammar(BaseModel):
    """Validate a flat JSON object against ordered function parameters."""

    parameters: dict[str, str]

    @staticmethod
    def load_vocabulary(path: str) -> dict[int, bytes]:
        """Map regular Qwen token IDs to their literal UTF-8 byte pieces."""
        vocabulary = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(vocabulary, dict):
            raise ValueError("Invalid Qwen vocabulary")

        base = list(range(33, 127))
        base += list(range(161, 173))
        base += list(range(174, 256))
        remaining = [value for value in range(256) if value not in base]
        chars = base + [256 + i for i in range(len(remaining))]
        byte_decoder = {
            chr(char): value
            for value, char in zip(base + remaining, chars)
        }

        try:
            return {
                int(token_id): bytes(byte_decoder[c] for c in piece)
                for piece, token_id in vocabulary.items()
            }
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("Invalid Qwen BPE vocabulary") from error

    @staticmethod
    def _spaces(data: bytes, pos: int) -> int:
        """Skip JSON whitespace."""
        while pos < len(data) and data[pos] in b" \t\r\n":
            pos += 1
        return pos

    @staticmethod
    def _literal(data: bytes, pos: int, literal: bytes) -> tuple[str, int]:
        """Match one fixed fragment while accepting unfinished prefixes."""
        available = data[pos:pos + len(literal)]
        if literal.startswith(available):
            if len(available) < len(literal):
                return "incomplete", pos
            return "ok", pos + len(literal)
        return "invalid", pos

    @staticmethod
    def _utf8_ok(value: bytes, final: bool) -> bool:
        """Reject malformed UTF-8, but allow incomplete multibyte prefixes."""
        try:
            decoder = codecs.getincrementaldecoder("utf-8")("strict")
            decoder.decode(value, final=final)
            return True
        except UnicodeDecodeError:
            return False

    @classmethod
    def _string(cls, data: bytes, pos: int) -> tuple[str, int]:
        """Parse a JSON string, including escapes and unicode prefixes."""
        if pos == len(data):
            return "incomplete", pos
        if data[pos] != 34:
            return "invalid", pos
        pos += 1
        chunk = bytearray()

        while pos < len(data):
            char = data[pos]
            if char in (34, 92):
                if not cls._utf8_ok(chunk, final=True):
                    return "invalid", pos
                chunk.clear()
                if char == 34:
                    return "ok", pos + 1
                pos += 1
                if pos == len(data):
                    return "incomplete", pos
                escaped = data[pos]
                if escaped == 117:  # \uXXXX
                    digits = data[pos + 1:pos + 5]
                    if any(c not in b"0123456789abcdefABCDEF" for c in digits):
                        return "invalid", pos
                    if len(digits) < 4:
                        return "incomplete", pos
                    pos += 5
                    continue
                if escaped not in b'"\\/bfnrt':
                    return "invalid", pos
            elif char < 32:
                return "invalid", pos
            else:
                chunk.append(char)
            pos += 1

        if not cls._utf8_ok(chunk, final=False):
            return "invalid", pos
        return "incomplete", pos

    @staticmethod
    def _number(data: bytes, pos: int, integer: bool) -> tuple[str, int]:
        """Parse JSON number syntax while permitting unfinished prefixes."""
        if pos == len(data):
            return "incomplete", pos
        if data[pos] == 45:
            pos += 1
            if pos == len(data):
                return "incomplete", pos
        if data[pos] == 48:
            pos += 1
            if pos < len(data) and 48 <= data[pos] <= 57:
                return "invalid", pos
        elif 49 <= data[pos] <= 57:
            while pos < len(data) and 48 <= data[pos] <= 57:
                pos += 1
        else:
            return "invalid", pos

        if not integer and pos < len(data) and data[pos] == 46:
            pos += 1
            if pos == len(data):
                return "incomplete", pos
            if not 48 <= data[pos] <= 57:
                return "invalid", pos
            while pos < len(data) and 48 <= data[pos] <= 57:
                pos += 1

        if not integer and pos < len(data) and data[pos] in (69, 101):
            pos += 1
            if pos < len(data) and data[pos] in (43, 45):
                pos += 1
            if pos == len(data):
                return "incomplete", pos
            if not 48 <= data[pos] <= 57:
                return "invalid", pos
            while pos < len(data) and 48 <= data[pos] <= 57:
                pos += 1

        return "ok", pos

    @classmethod
    def _value(cls, data: bytes, pos: int, kind: str) -> tuple[str, int]:
        """Parse the value type declared by the function schema."""
        if kind == "string":
            return cls._string(data, pos)
        if kind in ("number", "integer"):
            return cls._number(data, pos, kind == "integer")
        if kind == "boolean":
            true_status = cls._literal(data, pos, b"true")
            false_status = cls._literal(data, pos, b"false")
            if true_status[0] == "ok":
                return true_status
            if false_status[0] == "ok":
                return false_status
            if "incomplete" in (true_status[0], false_status[0]):
                return "incomplete", pos
        return "invalid", pos

    def status(self, data: bytes) -> str:
        """Return invalid, incomplete, or complete for a JSON byte prefix."""
        if not data.startswith(b"{"):
            return "invalid"
        pos = 1
        items = list(self.parameters.items())

        for i, (name, kind) in enumerate(items):
            pos = self._spaces(data, pos)
            result, pos = self._literal(
                data, pos, json.dumps(name, ensure_ascii=False).encode()
            )
            if result != "ok":
                return result
            pos = self._spaces(data, pos)
            result, pos = self._literal(data, pos, b":")
            if result != "ok":
                return result
            pos = self._spaces(data, pos)
            result, pos = self._value(data, pos, kind)
            if result != "ok":
                return result
            pos = self._spaces(data, pos)
            ending = b"}" if i == len(items) - 1 else b","
            result, pos = self._literal(data, pos, ending)
            if result != "ok":
                return result

        if not items:
            pos = self._spaces(data, pos)
            result, pos = self._literal(data, pos, b"}")
            if result != "ok":
                return result
        pos = self._spaces(data, pos)
        return "complete" if pos == len(data) else "invalid"
