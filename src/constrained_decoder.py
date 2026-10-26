"""Reusable constrained-decoding primitives."""

from typing import Callable, TypeVar

import numpy as np
from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ConfigDict


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
        """Select one finite candidate using candidate token sequences."""
        if not token_sequences:
            raise ValueError(
                "No constrained candidates available"
            )

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

            if not allowed_ids:
                raise ValueError(
                    "No token can continue constrained decoding"
                )

            logits = self.logits(
                context_ids + generated_ids
            )
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

        raise ValueError(
            "Constrained decoding eliminated every candidate"
        )

    def highest_valid_token(
        self,
        logits: list[float],
        is_valid: Callable[[int], bool],
    ) -> int:
        """Return the highest-logit token accepted by a validator."""
        for raw_token_id in np.argsort(logits)[::-1]:
            token_id = int(raw_token_id)

            if not np.isfinite(logits[token_id]):
                continue
            if is_valid(token_id):
                return token_id

        raise ValueError(
            "No token can continue constrained decoding"
        )
