"""Check JSON object prefixes with small state-based parsers."""

import json

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
    integer_only: bool = False

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
        """Prevent leading zeros and accept allowed numeric suffixes."""
        if self.integer_only:
            return -2
        if self.is_decimal_point(char):
            self.state = "fraction_start"
        elif self.is_exponent(char):
            self.state = "exponent_start"
        else:
            return -2
        return -1

    def read_integer(self, char: str) -> int:
        """Read digits and allow fractions only for number fields."""
        if self.is_digit(char):
            return -1
        if self.integer_only:
            return -2
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


class JsonBooleanParser(BaseModel):
    """Parse one JSON boolean, keeping its position in true or false."""

    state: str = "start"
    expected: str = ""
    position: int = 0

    @staticmethod
    def is_boolean_start(char: str) -> bool:
        """Recognize the first character of true or false."""
        return char in ("t", "f")

    @staticmethod
    def is_terminator(char: str) -> bool:
        """Recognize separators following a complete boolean."""
        return char in (" ", "\t", "\r", "\n", ",", "}")

    def read_start(self, char: str) -> int:
        """Choose the boolean literal from its first character."""
        if not self.is_boolean_start(char):
            return -2
        self.expected = "true" if char == "t" else "false"
        self.position = 1
        self.state = "literal"
        return -1

    def read_literal(self, char: str) -> int:
        """Require the next character of the expected literal."""
        if self.position >= len(self.expected):
            return -2
        if char != self.expected[self.position]:
            return -2
        self.position += 1
        return -1

    def read_end(self) -> int:
        """A boolean is complete only after every expected letter."""
        if self.state != "literal":
            return -2
        if self.position != len(self.expected):
            return -2
        self.state = "done"
        return 0

    def read(self, char: str) -> int:
        """Return 0 on a separator, -1 incomplete, -2 invalid."""
        if self.state == "done":
            return -2
        if self.is_terminator(char):
            return self.read_end()
        if self.state == "start":
            return self.read_start(char)
        if self.state == "literal":
            return self.read_literal(char)
        return -2


class JsonChecker(BaseModel):
    """Validate a JSON object prefix using required names and types."""

    parameters: dict[str, str]

    @staticmethod
    def is_whitespace(char: str) -> bool:
        """Recognize JSON whitespace."""
        return char in (" ", "\t", "\r", "\n")

    def can_skip_whitespace(self, state: str, char: str) -> bool:
        """Skip whitespace only outside strings and numeric tokens."""
        return (
            state in (
                "start", "key_or_end", "key_required", "colon",
                "value", "after_value", "done",
            )
            and self.is_whitespace(char)
        )

    def has_remaining_keys(self, used_keys: set[str]) -> bool:
        """Check whether another required key can be written."""
        return len(used_keys) < len(self.parameters)

    def can_close_object(self, used_keys: set[str]) -> bool:
        """Close only once every required key was written."""
        return used_keys == set(self.parameters)

    def is_expected_key(self, key: str, used_keys: set[str]) -> bool:
        """Reject unknown or repeated parameter names."""
        return key in self.parameters and key not in used_keys

    def is_key_prefix(self, key: str, used_keys: set[str]) -> bool:
        """Check whether a remaining parameter can begin with this key."""
        prefix = key.split("\\", 1)[0]

        for name in self.parameters:
            if name in used_keys:
                continue
            if name.startswith(prefix):
                return True
        return False

    def is_structure_allowed(
        self,
        state: str,
        char: str,
        used_keys: set[str],
    ) -> bool:
        """Check whether commas, keys and closing braces are allowed."""
        if state in ("key_or_end", "after_value"):
            if self.is_close_brace(char):
                return self.can_close_object(used_keys)
        if state == "after_value" and self.is_comma(char):
            return self.has_remaining_keys(used_keys)
        if state in ("key_or_end", "key_required"):
            if JsonStringParser.is_quote(char):
                return self.has_remaining_keys(used_keys)
        return True

    @staticmethod
    def is_open_brace(char: str) -> bool:
        """Recognize an opening brace."""
        return char == "{"

    @staticmethod
    def is_close_brace(char: str) -> bool:
        """Recognize a closing brace."""
        return char == "}"

    @staticmethod
    def is_comma(char: str) -> bool:
        """Recognize an object separator."""
        return char == ","

    @staticmethod
    def is_colon(char: str) -> bool:
        """Recognize the separator before a value."""
        return char == ":"

    @staticmethod
    def read_start(char: str) -> str:
        """Require an opening brace."""
        return (
            "key_or_end" if JsonChecker.is_open_brace(char)
            else "invalid"
        )

    @staticmethod
    def read_key_or_end(char: str) -> str:
        """Read the first key or an empty object."""
        if JsonChecker.is_close_brace(char):
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
        """Require the colon between a key and a value."""
        return "value" if JsonChecker.is_colon(char) else "invalid"

    @staticmethod
    def read_value(char: str, expected_type: str | None = None) -> str:
        """Choose a primitive parser compatible with the expected type."""
        if JsonStringParser.is_quote(char):
            return (
                "string_value" if expected_type in (None, "string")
                else "invalid"
            )
        if JsonNumberParser.is_number_start(char):
            return (
                "number_value" if expected_type in (None, "number", "integer")
                else "invalid"
            )
        if JsonBooleanParser.is_boolean_start(char):
            return (
                "boolean_value" if expected_type in (None, "boolean")
                else "invalid"
            )
        return "invalid"

    @staticmethod
    def read_after_value(char: str) -> str:
        """Accept either a comma or a closing brace."""
        if JsonChecker.is_comma(char):
            return "key_required"
        if JsonChecker.is_close_brace(char):
            return "done"
        return "invalid"

    @staticmethod
    def read_done(char: str) -> str:
        """Reject additional text after the closing brace."""
        return "invalid"

    def next_state(
        self, state: str, char: str, expected_type: str | None = None,
    ) -> str:
        """Dispatch structure states and typed value openings."""
        if state == "start":
            return self.read_start(char)
        if state == "key_or_end":
            return self.read_key_or_end(char)
        if state == "key_required":
            return self.read_key_required(char)
        if state == "colon":
            return self.read_colon(char)
        if state == "value":
            return self.read_value(char, expected_type)
        if state == "after_value":
            return self.read_after_value(char)
        if state == "done":
            return self.read_done(char)
        return "invalid"

    def status(self, text: str) -> int:
        """Return 0 complete, -1 valid prefix, -2 invalid."""
        state = "start"
        string_parser = JsonStringParser()
        number_parser = JsonNumberParser()
        boolean_parser = JsonBooleanParser()
        key_buffer = ""
        current_key = ""
        used_keys: set[str] = set()

        for t in text:
            if self.can_skip_whitespace(state, t):
                continue

            if state in ("key", "string_value"):
                result = string_parser.read(t)
                if result == -2:
                    return -2
                if state == "key":
                    if result == -1:
                        key_buffer += t
                        if not self.is_key_prefix(key_buffer, used_keys):
                            return -2
                    else:
                        try:
                            key = json.loads('"' + key_buffer + '"')
                        except json.JSONDecodeError:
                            return -2
                        if not self.is_expected_key(key, used_keys):
                            return -2
                        current_key = key
                        used_keys.add(key)
                        state = "colon"
                elif result == 0:
                    state = "after_value"
                continue

            if state in ("number_value", "boolean_value"):
                parser = (
                    number_parser if state == "number_value"
                    else boolean_parser
                )
                result = parser.read(t)
                if result == -2:
                    return -2
                if result == 0:
                    state = "after_value"
                    if not self.is_whitespace(t):
                        if not self.is_structure_allowed(
                            state, t, used_keys,
                        ):
                            return -2
                        state = self.read_after_value(t)
                if state == "invalid":
                    return -2
                continue

            if not self.is_structure_allowed(state, t, used_keys):
                return -2
            expected_type = None
            if state == "value":
                expected_type = self.parameters[current_key]
            state = self.next_state(state, t, expected_type)
            if state == "invalid":
                return -2
            if state in ("key", "string_value"):
                string_parser = JsonStringParser()
                string_parser.read(t)
                if state == "key":
                    key_buffer = ""
            elif state == "number_value":
                number_parser = JsonNumberParser(
                    integer_only=expected_type == "integer",
                )
                number_parser.read(t)
            elif state == "boolean_value":
                boolean_parser = JsonBooleanParser()
                boolean_parser.read(t)

        return 0 if state == "done" else -1
