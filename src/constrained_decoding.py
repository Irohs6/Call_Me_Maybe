"""Reusable constrained-decoding primitives."""

import json
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict, Field


CandidateKey = TypeVar("CandidateKey")


def allowed_ids_for_position(
    token_sequences: dict[CandidateKey, list[int]],
    position: int,
) -> list[int]:
    """Return the distinct token IDs allowed at one candidate position."""
    allowed_ids: list[int] = []

    for tokens in token_sequences.values():
        if len(tokens) <= position:
            continue
        token_id = tokens[position]
        if token_id not in allowed_ids:
            allowed_ids.append(token_id)

    return allowed_ids


class ConstrainedDecoder(BaseModel):
    """Centralize SDK access and token-level constrained selection."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    model: Small_LLM_Model
    vocabulary_ids: frozenset[int] = Field(default_factory=frozenset)

    def model_post_init(self, __context: Any) -> None:
        """Load the public vocabulary once after Pydantic validation."""
        self.vocabulary_ids = self._load_vocabulary_ids()

    def encode(self, text: str) -> list[int]:
        """Encode text with the public SDK and return flat token IDs."""
        return self.model.encode(text)[0].tolist()

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs with the public SDK."""
        return self.model.decode(token_ids)

    def logits(self, input_ids: list[int]) -> list[float]:
        """Return next-token logits through the public SDK."""
        return self.model.get_logits_from_input_ids(input_ids)

    def select_candidate(
        self,
        context_ids: list[int],
        token_sequences: dict[CandidateKey, list[int]],
    ) -> CandidateKey:
        """Select one finite candidate using only candidate token sequences."""
        if not token_sequences:
            raise ValueError("No constrained candidates available")

        remaining = token_sequences.copy()
        generated_ids: list[int] = []

        if len(remaining) == 1:
            return next(iter(remaining))

        while remaining:
            position = len(generated_ids)
            allowed_ids = allowed_ids_for_position(
                remaining,
                position,
            )
            allowed_ids = [
                token_id
                for token_id in allowed_ids
                if token_id in self.vocabulary_ids
            ]

            if not allowed_ids:
                raise ValueError(
                    "No vocabulary token can continue constrained decoding"
                )

            logits = self.logits(context_ids + generated_ids)
            next_token_id = max(
                allowed_ids,
                key=lambda token_id: logits[token_id],
            )
            generated_ids.append(next_token_id)

            remaining = {
                key: tokens
                for key, tokens in remaining.items()
                if tokens[:len(generated_ids)] == generated_ids
            }

            completed = [
                key
                for key, tokens in remaining.items()
                if tokens == generated_ids
            ]
            if completed:
                return completed[0]

            if len(remaining) == 1:
                return next(iter(remaining))

        raise ValueError("Constrained decoding eliminated every candidate")

    def highest_valid_token(
        self,
        logits: list[float],
        is_valid: Callable[[int], bool],
    ) -> int:
        """Return the highest-logit token accepted by a validator."""
        for raw_token_id in np.argsort(logits)[::-1]:
            token_id = int(raw_token_id)

            if token_id not in self.vocabulary_ids:
                continue
            if not np.isfinite(logits[token_id]):
                continue
            if is_valid(token_id):
                return token_id

        raise ValueError("No token can continue constrained decoding")

    def _load_vocabulary_ids(self) -> frozenset[int]:
        """Load token IDs from the vocabulary path exposed by the SDK."""
        vocab_path = Path(self.model.get_path_to_vocab_file())

        try:
            with vocab_path.open("r", encoding="utf-8") as file:
                vocabulary = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Unable to load vocabulary file: {vocab_path}"
            ) from error

        if not isinstance(vocabulary, dict):
            raise ValueError("Vocabulary file must contain a JSON object")

        token_ids = frozenset(
            token_id
            for token_id in vocabulary.values()
            if isinstance(token_id, int)
            and not isinstance(token_id, bool)
        )

        if not token_ids:
            raise ValueError("Vocabulary file contains no token IDs")

        return token_ids
