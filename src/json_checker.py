from pydantic import BaseModel


class JsonChecker(BaseModel):
    parameters: dict[str, str]

    def status(self, text: str) -> int:
        """Return 0 if valid, -1 if incomplete, -2 if invalid."""
        state = "start"

        for t in text:
            if state in (
                "start",
                "key_or_end",
                "key_required",
                "colon",
                "value",
                "after_value",
                "done",
            ) and t in " \t\r\n":
                continue
            if state == "start":
                if t == "{":
                    state = "key_or_end"
                else:
                    return -2

            elif state == "key_or_end":
                if t == "}":
                    state = "done"
                elif t == '"':
                    state = "key"
                else:
                    return -2
            elif state == "key_required":
                if t == '"':
                    state = "key"
                else:
                    return -2
            elif state == "key":
                if t == '"':
                    state = "colon"
                elif t == "\\":
                    state = "escape"
                elif ord(t) < 32:
                    return -2
                else:
                    continue
            elif state == "escape":
                if t in '"\\/bfnrt':
                    state = "key"
                elif t == "u":
                    state = "unicode_string"
                else:
                    return -2
            elif state == "colon":
                if t == ":":
                    state = "value"
                else:
                    return -2
            elif state == "value":
                if t == '"':
                    state = "string_value"
                else:
                    return -2
            elif state == "string_value":
                if t == '"':
                    state = "after_value"
                elif t == "\\":
                    state = "escape_string"
                elif ord(t) < 32:
                    return -2
                else:
                    continue
            elif state == "escape_string":
                state = "string_value"
            elif state == "unicode_string":
                if t in "0123456789abcdefABCDEF":
                    state = "unicode_string"
                else:
                    return -2
            elif state == "after_value":
                if t == ",":
                    state = "key_required"
                elif t == "}":
                    state = "done"
                else:
                    return -2
            else:
                return -2
        if state == "done":
            return 0

        return -1

    def check_string(self, value: str) -> str:
        """Check a JSON string prefix."""
        ...

    def check_number(self, value: str) -> str:
        """Check a JSON number prefix."""
        ...

    def check_boolean(self, value: str) -> str:
        """Check true or false prefixes."""
        ...