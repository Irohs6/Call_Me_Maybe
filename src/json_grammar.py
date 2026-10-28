"""Read JSON object structure and strings one character at a time."""

from pydantic import BaseModel


class JsonStringParser(BaseModel):
    """Parse one JSON string, keeping escape and Unicode state."""

    state: str = "start"
    unicode_count: int = 0

    @staticmethod
    def is_quote(char: str) -> bool:
        """Recognize JSON double quotes."""
        return char == '"'

    @staticmethod
    def is_escape(char: str) -> bool:
        """Recognize a backslash."""
        return char == "\\"

    @staticmethod
    def is_valid_escape(char: str) -> bool:
        """Recognize valid escapes after a backslash."""
        return char in ('"', "\\", "/", "b", "f", "n", "r", "t")

    @staticmethod
    def is_unicode_escape(char: str) -> bool:
        """Recognize a Unicode escape marker."""
        return char == "u"

    @staticmethod
    def is_hex_digit(char: str) -> bool:
        """Recognize a hexadecimal digit."""
        return char in "0123456789abcdefABCDEF"

    @staticmethod
    def is_control_character(char: str) -> bool:
        """Recognize forbidden unescaped control characters."""
        return ord(char) < 32

    def read_start(self, char: str) -> int:
        """Begin a string with an opening quote."""
        if not self.is_quote(char):
            return -2
        self.state = "string"
        return -1

    def read_string(self, char: str) -> int:
        """Read normal string content until its closing quote."""
        if self.is_quote(char):
            self.state = "done"
            return 0
        if self.is_escape(char):
            self.state = "escape"
        elif self.is_control_character(char):
            return -2
        return -1

    def read_escape(self, char: str) -> int:
        """Validate one escaped character or begin Unicode."""
        if self.is_valid_escape(char):
            self.state = "string"
        elif self.is_unicode_escape(char):
            self.state = "unicode"
            self.unicode_count = 0
        else:
            return -2
        return -1

    def read_unicode(self, char: str) -> int:
        """Require four hexadecimal digits after the Unicode marker."""
        if not self.is_hex_digit(char):
            return -2
        self.unicode_count += 1
        if self.unicode_count == 4:
            self.state = "string"
        return -1

    def read(self, char: str) -> int:
        """Return 0 complete, -1 incomplete, or -2 invalid."""
        if self.state == "start":
            return self.read_start(char)
        if self.state == "string":
            return self.read_string(char)
        if self.state == "escape":
            return self.read_escape(char)
        if self.state == "unicode":
            return self.read_unicode(char)
        return -2


class JsonGrammar(BaseModel):
    """Check JSON object structure with string keys and values."""

    @staticmethod
    def is_whitespace(char: str) -> bool:
        """Recognize JSON whitespace."""
        return char in (" ", "\t", "\r", "\n")

    def can_skip_whitespace(self, state: str, char: str) -> bool:
        """Skip whitespace only outside strings."""
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

    @staticmethod
    def read_key_or_end(char: str) -> str:
        """Allow the first key or an empty object."""
        if char == "}":
            return "done"
        if JsonStringParser.is_quote(char):
            return "key"
        return "invalid"

    @staticmethod
    def read_key_required(char: str) -> str:
        """Require a key after a comma."""
        if JsonStringParser.is_quote(char):
            return "key"
        return "invalid"

    @staticmethod
    def read_colon(char: str) -> str:
        """Require a colon before a value."""
        return "value" if char == ":" else "invalid"

    @staticmethod
    def read_value(char: str) -> str:
        """Begin a string value; other types will come later."""
        if JsonStringParser.is_quote(char):
            return "string_value"
        return "invalid"

    @staticmethod
    def read_after_value(char: str) -> str:
        """Accept a comma or close the object."""
        if char == ",":
            return "key_required"
        if char == "}":
            return "done"
        return "invalid"

    @staticmethod
    def read_done(char: str) -> str:
        """Forbid any non-whitespace after the closing brace."""
        return "invalid"

    def next_state(self, state: str, char: str) -> str:
        """Dispatch only the JSON object structure."""
        if state == "start":
            return self.read_start(char)
        if state == "key_or_end":
            return self.read_key_or_end(char)
        if state == "key_required":
            return self.read_key_required(char)
        if state == "colon":
            return self.read_colon(char)
        if state == "value":
            return self.read_value(char)
        if state == "after_value":
            return self.read_after_value(char)
        if state == "done":
            return self.read_done(char)
        return "invalid"

    def status(self, text: str) -> int:
        """Return 0 complete, -1 incomplete, -2 invalid."""
        state = "start"
        string_parser = JsonStringParser()

        for char in text:
            if self.can_skip_whitespace(state, char):
                continue

            if state in ("key", "string_value"):
                result = string_parser.read(char)
                if result == -2:
                    return -2
                if result == 0:
                    state = (
                        "colon" if state == "key" else "after_value"
                    )
                continue

            state = self.next_state(state, char)
            if state == "invalid":
                return -2
            if state in ("key", "string_value"):
                string_parser = JsonStringParser()
                string_parser.read(char)

        return 0 if state == "done" else -1
