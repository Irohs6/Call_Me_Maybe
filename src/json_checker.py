"""Check JSON object prefixes against the selected function parameters."""

import json

from pydantic import BaseModel


class JsonChecker(BaseModel):
    """Validate JSON syntax, required keys and parameter types."""

    parameters: dict[str, str]

    def check_key_prefix(
        self, value: str, used_keys: set[str],
    ) -> bool:
        """Keep only prefixes that can become a remaining key."""
        try:
            prefix = json.loads(value + '"')
        except json.JSONDecodeError:
            return True
        if any(0xD800 <= ord(t) <= 0xDFFF for t in prefix):
            return True
        return any(
            name.startswith(prefix)
            for name in self.parameters
            if name not in used_keys
        )

    def check_value(self, value: str, type_name: str) -> int:
        """Check a value using the type defined by the function."""
        if type_name == "string":
            return self.check_string(value)
        if type_name in ("number", "integer"):
            if type_name == "integer" and any(
                t in value for t in ".eE"
            ):
                return -2
            return self.check_number(value)
        if type_name == "boolean":
            return self.check_boolean(value)
        return -2

    def status(self, text: str) -> int:
        """Return 0 if valid, -1 if incomplete, -2 if invalid."""
        state = "start"
        value = ""
        current_key = ""
        used_keys: set[str] = set()
        required_keys = set(self.parameters)

        for t in text:
            if state in (
                "start", "key_or_end", "key_required",
                "colon", "value", "after_value", "done",
            ) and t in " \t\r\n":
                continue

            if state == "start":
                if t == "{":
                    state = "key_or_end"
                else:
                    return -2

            elif state == "key_or_end":
                if t == "}":
                    if used_keys != required_keys:
                        return -2
                    state = "done"
                elif t == '"' and used_keys != required_keys:
                    value = '"'
                    state = "key"
                else:
                    return -2

            elif state == "key_required":
                if t == '"' and used_keys != required_keys:
                    value = '"'
                    state = "key"
                else:
                    return -2

            elif state == "key":
                value += t
                result = self.check_string(value)
                if result == -2:
                    return -2
                if result == 0:
                    current_key = json.loads(value)
                    if current_key not in self.parameters:
                        return -2
                    if current_key in used_keys:
                        return -2
                    used_keys.add(current_key)
                    state = "colon"
                elif not self.check_key_prefix(value, used_keys):
                    return -2

            elif state == "colon":
                if t == ":":
                    state = "value"
                else:
                    return -2

            elif state == "value":
                value = t
                type_name = self.parameters[current_key]
                if type_name == "string" and t == '"':
                    state = "string_value"
                elif type_name in ("number", "integer") and (
                    t == "-" or t in "0123456789"
                ):
                    state = "number_value"
                elif type_name == "boolean" and t in "tf":
                    state = "boolean_value"
                else:
                    return -2

            elif state == "string_value":
                value += t
                result = self.check_string(value)
                if result == -2:
                    return -2
                if result == 0:
                    state = "after_value"

            elif state in ("number_value", "boolean_value"):
                if t in " \t\r\n,}":
                    result = self.check_value(
                        value, self.parameters[current_key],
                    )
                    if result != 0:
                        return -2
                    if t == "}":
                        if used_keys != required_keys:
                            return -2
                        state = "done"
                    elif t == ",":
                        if used_keys == required_keys:
                            return -2
                        state = "key_required"
                    else:
                        state = "after_value"
                else:
                    value += t
                    if self.check_value(
                        value, self.parameters[current_key],
                    ) == -2:
                        return -2

            elif state == "after_value":
                if t == "}":
                    if used_keys != required_keys:
                        return -2
                    state = "done"
                elif t == "," and used_keys != required_keys:
                    state = "key_required"
                else:
                    return -2

            else:
                return -2

        return 0 if state == "done" else -1

    def check_string(self, value: str) -> int:
        """Check a JSON string prefix including quotes and escapes."""
        state = "start"
        unicode_count = 0

        for t in value:
            if state == "start":
                if t != '"':
                    return -2
                state = "string"

            elif state == "string":
                if t == '"':
                    state = "done"
                elif t == "\\":
                    state = "escape"
                elif ord(t) < 32:
                    return -2

            elif state == "escape":
                if t in '"\\/bfnrt':
                    state = "string"
                elif t == "u":
                    unicode_count = 0
                    state = "unicode"
                else:
                    return -2

            elif state == "unicode":
                if t not in "0123456789abcdefABCDEF":
                    return -2
                unicode_count += 1
                if unicode_count == 4:
                    state = "string"

            else:
                return -2

        return 0 if state == "done" else -1

    def check_number(self, value: str) -> int:
        """Check a JSON number prefix including decimals and exponents."""
        state = "start"

        for t in value:
            if state == "start":
                if t == "-":
                    state = "sign"
                elif t == "0":
                    state = "zero"
                elif t in "123456789":
                    state = "integer"
                else:
                    return -2

            elif state == "sign":
                if t == "0":
                    state = "zero"
                elif t in "123456789":
                    state = "integer"
                else:
                    return -2

            elif state in ("zero", "integer"):
                if state == "integer" and t in "0123456789":
                    continue
                if t == ".":
                    state = "fraction_start"
                elif t in "eE":
                    state = "exponent_start"
                else:
                    return -2

            elif state == "fraction_start":
                if t not in "0123456789":
                    return -2
                state = "fraction"

            elif state == "fraction":
                if t in "0123456789":
                    continue
                if t in "eE":
                    state = "exponent_start"
                else:
                    return -2

            elif state == "exponent_start":
                if t in "+-":
                    state = "exponent_sign"
                elif t in "0123456789":
                    state = "exponent_digits"
                else:
                    return -2

            elif state == "exponent_sign":
                if t not in "0123456789":
                    return -2
                state = "exponent_digits"

            elif state == "exponent_digits":
                if t not in "0123456789":
                    return -2

        if state in ("zero", "integer", "fraction", "exponent_digits"):
            return 0
        return -1

    def check_boolean(self, value: str) -> int:
        """Check true and false prefixes."""
        if value in ("true", "false"):
            return 0
        if "true".startswith(value) or "false".startswith(value):
            return -1
        return -2
