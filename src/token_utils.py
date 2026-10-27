"""Shared utilities for constrained token selection."""

from collections.abc import Iterable, Sequence


def allowed_ids_for_position(
    sequences: Iterable[Sequence[int]],
    position: int,
) -> list[int]:
    """Return unique token IDs available at one candidate position."""
    allowed_ids: list[int] = []

    for tokens in sequences:
        if position >= len(tokens):
            continue

        token_id = tokens[position]
        if token_id not in allowed_ids:
            allowed_ids.append(token_id)

    return allowed_ids
