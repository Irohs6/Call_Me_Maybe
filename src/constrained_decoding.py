from typing import TypeVar


CandidateKey = TypeVar("CandidateKey")


def allowed_ids_for_position(
    token_sequences: dict[CandidateKey, list[int]],
    position: int,
) -> list[int]:
    allowed_ids: list[int] = []

    for tokens in token_sequences.values():
        if len(tokens) > position:
            token_id = tokens[position]

            if token_id not in allowed_ids:
                allowed_ids.append(token_id)

    return allowed_ids