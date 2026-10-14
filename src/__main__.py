from llm_sdk import Small_LLM_Model


def main() -> None:
    model = Small_LLM_Model()

    text = "My name is Nathan"
    input_ids = model.encode(text)
    token_ids = input_ids[0].tolist()

    for _ in range(10):
        logits = model.get_logits_from_input_ids(token_ids)
        next_token_id = max(
            range(len(logits)),
            key=lambda token_id: logits[token_id],
        )

        token_ids.append(next_token_id)

        token_text = model.decode([next_token_id])
        print(
            f"Token ID: {next_token_id:<6} "
            f"Text: {token_text!r}"
        )

    print(len(logits))
    print(f"Generated: {model.decode(token_ids)}")

    print(input_ids)
    print(input_ids[0])
    print(input_ids[0].tolist())


if __name__ == "__main__":
    main()