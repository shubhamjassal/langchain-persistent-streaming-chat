
import pytest
from pydantic import BaseModel

from memory import estimate_tokens, trim_history


class Message(BaseModel):
    role: str
    content: str


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("Hello") == 2
    assert estimate_tokens("a" * 100) == 25


def test_trim_history_preserves_order():
    history = [
        Message(role="user", content="Question 1"),
        Message(role="assistant", content="Answer 1"),
        Message(role="user", content="Question 2"),
        Message(role="assistant", content="Answer 2"),
    ]

    result = trim_history(history, token_budget=1000)

    assert [message.content for message in result] == [
        "Question 1",
        "Answer 1",
        "Question 2",
        "Answer 2",
    ]


def test_trim_history_keeps_newest_complete_turn():
    history = [
        Message(role="user", content="a" * 100),
        Message(role="assistant", content="b" * 100),
        Message(role="user", content="Recent question"),
        Message(role="assistant", content="Recent answer"),
    ]

    result = trim_history(history, token_budget=50)

    assert [message.content for message in result] == [
        "Recent question",
        "Recent answer",
    ]


def test_trim_history_rejects_invalid_turn_order():
    history = [
        Message(role="user", content="Question"),
        Message(role="user", content="Another question"),
    ]

    with pytest.raises(ValueError):
        trim_history(history, token_budget=1000)


def test_trim_history_rejects_negative_budget():
    with pytest.raises(ValueError):
        trim_history([], token_budget=-1)


def test_trim_history_rejects_incomplete_turn():
    history = [
        Message(role="user", content="Question"),
    ]

    with pytest.raises(ValueError, match="complete user-assistant turns"):
        trim_history(history, token_budget=1000)


def test_trim_history_with_empty_history():
    assert trim_history([], token_budget=100) == []


def test_trim_history_with_zero_budget():
    history = [
        Message(role="user", content="Question"),
        Message(role="assistant", content="Answer"),
    ]

    assert trim_history(history, token_budget=0) == []


def test_trim_history_rejects_blank_content():
    history = [
        Message(role="user", content="   "),
        Message(role="assistant", content="Answer"),
    ]

    with pytest.raises(ValueError, match="cannot be blank"):
        trim_history(history, token_budget=1000)
