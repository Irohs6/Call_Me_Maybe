"""Constrain JSON string values with Qwen's vocabulary."""

import codecs
import json
from pathlib import Path

from pydantic import BaseModel


class StringGrammar(BaseModel):
    """Check whether bytes form a valid JSON string prefix."""

    @staticmethod
    def load_vocabulary(path: str) -> dict[int, bytes]:
        """Convert Qwen byte-level BPE entries into token ID -> bytes."""
        vocabulary = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        base = list(range(33, 127))
        base += list(range(161, 173))
        base += list(range(174, 256))
        remaining = [n for n in range(256) if n not in base]
        chars = base + [256 + i for i in range(len(remaining))]
        byte_decoder = {
            chr(char): value
            for value, char in zip(base + remaining, chars)
        }

        return {
            int(token_id): bytes(byte_decoder[char] for char in piece)
            for piece, token_id in vocabulary.items()
        }

    @staticmethod
    def valid_utf8(data: bytes, final: bool) -> bool:
        """Allow UTF-8 split between tokens but refuse invalid bytes."""
        decoder = codecs.getincrementaldecoder("utf-8")("strict")
        try:
            decoder.decode(data, final=final)
            return True
        except UnicodeDecodeError:
            return False

    def status(self, data: bytes) -> int:
        """Return 0 complete, -1 incomplete, or -2 invalid."""
        if not data.startswith(b'"'):
            return -2

        state = "string"
        unicode_count = 0
        chunk = bytearray()

        for pos, char in enumerate(data[1:], start=1):
            if state == "unicode":
                if char not in b"0123456789abcdefABCDEF":
                    return -2
                unicode_count += 1
                if unicode_count == 4:
                    state = "string"
            elif state == "escape":
                if char == 117:  # u
                    state = "unicode"
                    unicode_count = 0
                elif char in b'"\\/bfnrt':
                    state = "string"
                else:
                    return -2
            elif char == 34:  # closing quote
                if not self.valid_utf8(chunk, final=True):
                    return -2
                return 0 if pos == len(data) - 1 else -2
            elif char == 92:  # backslash
                if not self.valid_utf8(chunk, final=True):
                    return -2
                chunk.clear()
                state = "escape"
            elif char < 32:
                return -2
            else:
                chunk.append(char)

        if state == "string" and not self.valid_utf8(chunk, final=False):
            return -2
        return -1
