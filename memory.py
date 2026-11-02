
import os
from math import ceil


HISTORY_TOKEN_BUDGET = int(
    os.getenv("HISTORY_TOKEN_BUDGET", "2000")
)


def estimate_tokens(text: str) -> int:
    """Estimate tokens using roughly four characters per token."""
    if not text:
        return 0

    return ceil(len(text) / 4)


def estimate_message_tokens(message) -> int:
    """Estimate tokens for a message, including role overhead."""
    return (
        estimate_tokens(message.role)
        + estimate_tokens(message.content)
        + 4
    )


def validate_history(history) -> None:
    """Require complete, alternating user-assistant turns."""
    if len(history) % 2 != 0:
        raise ValueError(
            "History must contain complete user-assistant turns."
        )

    for index in range(0, len(history), 2):
        user_message = history[index]
        assistant_message = history[index + 1]

        if (
            user_message.role != "user"
            or assistant_message.role != "assistant"
        ):
            raise ValueError(
                "History must contain alternating user-assistant turns."
            )

        if (
            not user_message.content.strip()
            or not assistant_message.content.strip()
        ):
            raise ValueError("History messages cannot be blank.")


def trim_history(history, token_budget: int = HISTORY_TOKEN_BUDGET) -> list:
    """Keep the newest complete turns within the estimated token budget."""
    if token_budget < 0:
        raise ValueError("Token budget cannot be negative.")

    validate_history(history)

    selected_turns = []
    tokens_used = 0

    for index in range(len(history) - 2, -1, -2):
        user_message = history[index]
        assistant_message = history[index + 1]

        turn_tokens = (
            estimate_message_tokens(user_message)
            + estimate_message_tokens(assistant_message)
        )

        if turn_tokens > token_budget - tokens_used:
            break

        selected_turns.append((user_message, assistant_message))
        tokens_used += turn_tokens

    selected_turns.reverse()

    return [
        message
        for turn in selected_turns
        for message in turn
    ]
