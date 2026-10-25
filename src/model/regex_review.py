"""Generate alternative interpretations of a broad regular expression."""

import json
import re

from pydantic import BaseModel

from .models import Function, Type


class RegexInterpretations(BaseModel):
    """Offer regex, literal and character-class choices for LLM review."""

    pattern: str

    def syntax(self) -> tuple[bool, list[str]]:
        """Locate unescaped wildcards/anchors and complete character classes."""
        broad = False
        classes: list[str] = []
        class_start: int | None = None
        position = 0
        while position < len(self.pattern):
            character = self.pattern[position]
            if character == "\\":
                position += 2
                continue
            if class_start is not None:
                first = class_start + 1
                if self.pattern[first:first + 1] == "^":
                    first += 1
                if character == "]" and position > first:
                    classes.append(self.pattern[class_start:position + 1])
                    class_start = None
            elif character == "[":
                class_start = position
            elif character in ".$^":
                broad = True
            position += 1
        return broad, classes

    def candidates(self) -> list[tuple[str, Function]]:
        """Build valid choices without deciding which interpretation is intended."""
        try:
            compiled = re.compile(self.pattern)
        except re.error:
            return []
        broad, classes = self.syntax()
        if not broad and compiled.match("") is None:
            return []

        candidates: list[tuple[str, Function]] = []

        def add(pattern: str, name: str, description: str) -> None:
            if any(existing == pattern for existing, _ in candidates):
                return
            try:
                re.compile(pattern)
            except re.error:
                return
            candidates.append((pattern, Function(
                name=name,
                description=description,
                parameters={},
                returns=Type(type="string"),
            )))

        add(
            self.pattern,
            "fn_replace_with_regex",
            "Replace matches of the regular expression "
            + json.dumps(self.pattern)
            + ". Interpret special regex syntax, including wildcards and "
            "repetition, as operators.",
        )
        add(
            re.escape(self.pattern),
            "fn_replace_literal_text",
            "Replace occurrences of the exact literal text "
            + json.dumps(self.pattern)
            + ". Punctuation is matched literally, with no wildcard "
            "interpretation.",
        )
        for index, character_class in enumerate(classes):
            body = character_class[1:-1]
            if any(operator in body for operator in "\\-^[]"):
                description = (
                    "Replace each individual character matched by the regex "
                    "character class " + json.dumps(character_class)
                )
            else:
                description = (
                    "Replace each individual character belonging to this set: "
                    + ", ".join(body)
                )
            description += (
                ". Match one such character at a time and preserve all other "
                "characters."
            )
            name = "fn_replace_individual_characters"
            if index:
                name += f"_{index}"
            add(character_class, name, description)
        return candidates
