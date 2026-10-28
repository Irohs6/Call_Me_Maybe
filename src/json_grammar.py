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


class JsonNumberParser(BaseModel):
    """Parse one JSON number, keeping its numeric state."""

    state: str = "start"

    @staticmethod
    def is_negative_sign(char: str) -> bool:
        """Check for the minus sign."""
        return char == "-"

    @staticmethod
    def is_digit(char: str) -> bool:
        """Recognize digits from zero to nine."""
        return char in "0123456789"

    @staticmethod
    def is_non_zero_digit(char: str) -> bool:
        """Recognize digits from one to nine."""
        return char in "123456789"

    @staticmethod
    def is_decimal_point(char: str) -> bool:
        """Check for the decimal point."""
        return char == "."

    @staticmethod
    def is_exponent(char: str) -> bool:
        """Check for the exponent marker."""
        return char in ("e", "E")

    @staticmethod
    def is_exponent_sign(char: str) -> bool:
        """Check for an optional exponent sign."""
        return char in ("+", "-")

    @staticmethod
    def is_terminator(char: str) -> bool:
        """Recognize separators following a completed number."""
        return char in (" ", "\t", "\r", "\n", ",", "}")

    @staticmethod
    def is_number_start(char: str) -> bool:
        """Recognize the possible first characters of a number."""
        return char == "-" or char in "0123456789"

    def read_start(self, char: str) -> int:
        """Start an integer, possibly with a minus sign."""
        if self.is_negative_sign(char):
            self.state = "sign"
        elif char == "0":
            self.state = "zero"
        elif self.is_non_zero_digit(char):
            self.state = "integer"
        else:
            return -2
        return -1

    def read_sign(self, char: str) -> int:
        """Require a digit after the minus sign."""
        if char == "0":
            self.state = "zero"
        elif self.is_non_zero_digit(char):
            self.state = "integer"
        else:
            return -2
        return -1

    def read_zero(self, char: str) -> int:
        """Prevent leading zeros and accept decimals or exponents."""
        if self.is_decimal_point(char):
            self.state = "fraction_start"
        elif self.is_exponent(char):
            self.state = "exponent_start"
        else:
            return -2
        return -1

    def read_integer(self, char: str) -> int:
        """Read the integer digits."""
        if self.is_digit(char):
            return -1
        if self.is_decimal_point(char):
            self.state = "fraction_start"
        elif self.is_exponent(char):
            self.state = "exponent_start"
        else:
            return -2
        return -1

    def read_fraction_start(self, char: str) -> int:
        """Require a digit immediately after a decimal point."""
        if not self.is_digit(char):
            return -2
        self.state = "fraction"
        return -1

    def read_fraction(self, char: str) -> int:
        """Read fraction digits and an optional exponent."""
        if self.is_digit(char):
            return -1
        if not self.is_exponent(char):
            return -2
        self.state = "exponent_start"
        return -1

    def read_exponent_start(self, char: str) -> int:
        """Accept an exponent sign or its first digit."""
        if self.is_exponent_sign(char):
            self.state = "exponent_sign"
        elif self.is_digit(char):
            self.state = "exponent_digits"
        else:
            return -2
        return -1

    def read_exponent_sign(self, char: str) -> int:
        """Require a digit after the optional exponent sign."""
        if not self.is_digit(char):
            return -2
        self.state = "exponent_digits"
        return -1

    def read_exponent_digits(self, char: str) -> int:
        """Continue reading exponent digits."""
        return -1 if self.is_digit(char) else -2

    def can_finish(self) -> bool:
        """A number must have a complete integer, fraction or exponent."""
        return self.state in (
            "zero", "integer", "fraction", "exponent_digits",
        )

    def read(self, char: str) -> int:
        """Return 0 on a separator, -1 incomplete, -2 invalid."""
        if self.is_terminator(char):
            if not self.can_finish():
                return -2
            self.state = "done"
            return 0
        if self.state == "start":
            return self.read_start(char)
        if self.state == "sign":
            return self.read_sign(char)
        if self.state == "zero":
            return self.read_zero(char)
        if self.state == "integer":
            return self.read_integer(char)
        if self.state == "fraction_start":
            return self.read_fraction_start(char)
        if self.state == "fraction":
            return self.read_fraction(char)
        if self.state == "exponent_start":
            return self.read_exponent_start(char)
        if self.state == "exponent_sign":
            return self.read_exponent_sign(char)
        if self.state == "exponent_digits":
            return self.read_exponent_digits(char)
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
        """Begin a string or number value."""
        if JsonStringParser.is_quote(char):
            return "string_value"
        if JsonNumberParser.is_number_start(char):
            return "number_value"
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
        number_parser = JsonNumberParser()

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

            if state == "number_value":
                result = number_parser.read(char)
                if result == -2:
                    return -2
                if result == 0:
                    state = "after_value"
                    if not self.is_whitespace(char):
                        state = self.read_after_value(char)
                if state == "invalid":
                    return -2
                continue

            state = self.next_state(state, char)
            if state == "invalid":
                return -2
            if state in ("key", "string_value"):
                string_parser = JsonStringParser()
                string_parser.read(char)
            elif state == "number_value":
                number_parser = JsonNumberParser()
                number_parser.read(char)

        return 0 if state == "done" else -1
