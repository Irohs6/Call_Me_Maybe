"""Read JSON object structure one character at a time."""

from pydantic import BaseModel


class JsonGrammar(BaseModel):
    """Check JSON object prefixes (value parsing is not implemented yet)."""

    @staticmethod
    def is_whitespace(char: str) -> bool:
        """Recognize only the whitespace characters allowed by JSON."""
        return char in (" ", "\t", "\r", "\n")

    @staticmethod
    def is_quote(char: str) -> bool:
        """Check for a double quote."""
        return char == '"'

    @staticmethod
    def is_escape(char: str) -> bool:
        """Check for a backslash starting a JSON escape."""
        return char == "\\"

    @staticmethod
    def is_valid_escape(char: str) -> bool:
        """Recognize the simple escapes permitted after a backslash."""
        return char in ('"', "\\", "/", "b", "f", "n", "r", "t")

    @staticmethod
    def is_unicode_escape(char: str) -> bool:
        """Check for the u introducing four hexadecimal digits."""
        return char == "u"

    @staticmethod
    def is_hex_digit(char: str) -> bool:
        """Recognize a hexadecimal digit in a Unicode escape."""
        return char in "0123456789abcdefABCDEF"

    @staticmethod
    def is_control_character(char: str) -> bool:
        """Reject unescaped control characters inside strings."""
        return ord(char) < 32

    def can_skip_whitespace(self, state: str, char: str) -> bool:
        """Ignore whitespace outside JSON strings."""
        return (
            state in (
                "start", "key_or_end", "key_required", "colon",
                "value", "after_value", "done",
            )
            and self.is_whitespace(char)
        )

    @staticmethod
    def read_start(char: str) -> str:
        """Expect the opening brace."""
        return "key_or_end" if char == "{" else "invalid"

    def read_key_or_end(self, char: str) -> str:
        """Allow a key or an empty object immediately after '{'."""
        if char == "}":
            return "done"
        if self.is_quote(char):
            return "key"
        return "invalid"

    def read_key_required(self, char: str) -> str:
        """Require another key after a comma."""
        return "key" if self.is_quote(char) else "invalid"

    def read_key(self, char: str) -> str:
        """Read a key until an unescaped closing quote."""
        if self.is_quote(char):
            return "colon"
        if self.is_escape(char):
            return "escape_key"
        if self.is_control_character(char):
            return "invalid"
        return "key"

    def read_escape_key(self, char: str) -> str:
        """Read one escape sequence inside a key."""
        if self.is_valid_escape(char):
            return "key"
        if self.is_unicode_escape(char):
            return "unicode_key"
        return "invalid"

    @staticmethod
    def read_colon(char: str) -> str:
        """Require a colon between a key and its value."""
        return "value" if char == ":" else "invalid"

    @staticmethod
    def read_after_value(char: str) -> str:
        """Accept a comma or the end of an object after a value."""
        if char == ",":
            return "key_required"
        if char == "}":
            return "done"
        return "invalid"

    @staticmethod
    def read_done(char: str) -> str:
        """Forbid non-whitespace text after the closing brace."""
        return "invalid"

    def next_state(self, state: str, char: str) -> str:
        """Delegate structural transitions to small reader methods."""
        if state == "start":
            return self.read_start(char)
        if state == "key_or_end":
            return self.read_key_or_end(char)
        if state == "key_required":
            return self.read_key_required(char)
        if state == "key":
            return self.read_key(char)
        if state == "escape_key":
            return self.read_escape_key(char)
        if state == "colon":
            return self.read_colon(char)
        if state == "after_value":
            return self.read_after_value(char)
        if state == "done":
            return self.read_done(char)
        return "invalid"

    def status(self, text: str) -> int:
        """Return 0 complete, -1 incomplete, -2 invalid."""
        state = "start"
        unicode_count = 0

        for char in text:
            if self.can_skip_whitespace(state, char):
                continue

            if state == "value":
                raise NotImplementedError("Value parsing is not implemented")

            if state == "unicode_key":
                if not self.is_hex_digit(char):
                    return -2
                unicode_count += 1
                if unicode_count == 4:
                    state = "key"
                continue

            state = self.next_state(state, char)
            if state == "invalid":
                return -2
            if state == "unicode_key":
                unicode_count = 0

        return 0 if state == "done" else -1
