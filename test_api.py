
from unittest.mock import MagicMock, patch

import database
import pytest
from fastapi.testclient import TestClient

from api import app
from database import save_message

@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db_path = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db_path)

    with TestClient(app) as test_client:
        yield test_client


def create_test_conversation(client):
    response = client.post("/conversations")
    assert response.status_code == 201
    return response.json()["conversation_id"]


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_returns_router_result(client):
    expected = {
        "answer": "RAG means Retrieval-Augmented Generation.",
        "source": "gemini",
        "status": "success",
    }

    with patch("api.router") as mock_router:
        mock_router.invoke.return_value = expected

        response = client.post(
            "/ask",
            json={
                "type": "general",
                "question": "What does RAG stand for?",
            },
        )

    assert response.status_code == 200
    assert response.json() == expected
    mock_router.invoke.assert_called_once()


def test_invalid_route_is_rejected_before_router_call(client):
    with patch("api.router") as mock_invoke:
        response = client.post(
            "/ask",
            json={
                "type": "unknown",
                "question": "What does RAG stand for?",
            },
        )

    assert response.status_code == 422
    mock_invoke.assert_not_called()


def test_router_failure_returns_safe_error(client):
    with patch("api.router") as mock_router:
        mock_router.invoke.side_effect = RuntimeError(
            "Internal service details"
        )

        response = client.post(
            "/ask",
            json={
                "type": "general",
                "question": "What does RAG stand for?",
            },
        )

    assert response.status_code == 500
    assert response.json() == {
        "detail": "Unable to process the question right now."
    }
    assert "Internal service details" not in response.text


def test_stream_returns_token_and_done_events(client):
    mock_chunks = [
        MagicMock(content="Hello "),
        MagicMock(content="world!"),
    ]

    conversation_id = create_test_conversation(client)

    with patch("api.model") as mock_model:
        mock_model.stream.return_value = iter(mock_chunks)

        response = client.post(
            "/ask/stream",
            json={
                "conversation_id": conversation_id,
                "type": "general",
                "question": "Say hello",
            },
        )

    assert response.status_code == 200
    assert "event: token" in response.text
    assert '"Hello "' in response.text
    assert '"world!"' in response.text
    assert "event: done" in response.text


def test_stream_saves_completed_exchange(client):
    conversation_id = create_test_conversation(client)

    mock_chunks = [
        MagicMock(content="Robotics "),
        MagicMock(content="is fascinating."),
    ]

    with patch("api.model") as mock_model:
        mock_model.stream.return_value = iter(mock_chunks)

        response = client.post(
            "/ask/stream",
            json={
                "conversation_id": conversation_id,
                "type": "general",
                "question": "Tell me about robotics.",
            },
        )

    assert response.status_code == 200
    assert "event: done" in response.text

    history_response = client.get(
        f"/conversations/{conversation_id}/messages"
    )

    assert history_response.status_code == 200

    messages = history_response.json()["messages"]

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Tell me about robotics."
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Robotics is fascinating."


def test_stream_returns_safe_error_event(client):
    conversation_id = create_test_conversation(client)

    with patch("api.model") as mock_model:
        mock_model.stream.side_effect = RuntimeError(
            "Internal service details"
        )

        response = client.post(
            "/ask/stream",
            json={
                "conversation_id": conversation_id,
                "type": "general",
                "question": "Say hello",
            },
        )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "Unable to complete and save the response." in response.text
    assert "Internal service details" not in response.text


def test_stream_rejects_document_route(client):
    conversation_id = create_test_conversation(client)

    with patch("api.model") as mock_model:
        response = client.post(
            "/ask/stream",
            json={
                "conversation_id": conversation_id,
                "type": "document",
                "question": "Summarize the PDF",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Streaming currently supports general questions only."
    )
    mock_model.stream.assert_not_called()


def test_get_messages_for_unknown_conversation_returns_404(client):
    response = client.get(
        "/conversations/nonexistent-conversation/messages"
    )

    assert response.status_code == 404
    assert response.json() == {
        "detail": "Conversation does not exist."
    }



def test_delete_conversation_removes_history(client):
    conversation_id = create_test_conversation(client)

    # Save a real message in the isolated test database.
    save_message(
        conversation_id,
        "user",
        "Test message",
    )

    # Delete the conversation and its messages.
    response = client.delete(
        f"/conversations/{conversation_id}"
    )

    assert response.status_code == 200
    assert response.json() == {
        "deleted": True,
        "conversation_id": conversation_id,
    }

    # The deleted conversation's history should no longer be accessible.
    history_response = client.get(
        f"/conversations/{conversation_id}/messages"
    )

    assert history_response.status_code == 404
    assert history_response.json() == {
        "detail": "Conversation does not exist."
    }


def test_conversation_list_counts_messages_and_excludes_deleted(
    client,
):
    # Create two conversations.
    empty_conversation_id = create_test_conversation(client)
    conversation_id = create_test_conversation(client)

    # Add two messages to the second conversation.
    save_message(conversation_id, "user", "What is robotics?")
    save_message(
        conversation_id,
        "assistant",
        "Robotics is the study of robots.",
    )

    # Check that the list reports accurate message counts.
    response = client.get("/conversations")

    assert response.status_code == 200

    conversations = response.json()["conversations"]
    conversations_by_id = {
        conversation["id"]: conversation
        for conversation in conversations
    }

    assert conversations_by_id[empty_conversation_id]["message_count"] == 0
    assert conversations_by_id[conversation_id]["message_count"] == 2

    # Delete the conversation containing messages.
    delete_response = client.delete(
        f"/conversations/{conversation_id}"
    )
    assert delete_response.status_code == 200

    # Confirm the deleted conversation no longer appears.
    response = client.get("/conversations")

    assert response.status_code == 200

    remaining_ids = {
        conversation["id"]
        for conversation in response.json()["conversations"]
    }

    assert empty_conversation_id in remaining_ids
    assert conversation_id not in remaining_ids


def test_rename_conversation_endpoint(client):
    conversation_id = "api-rename-test"
    database.create_conversation(conversation_id)

    response = client.patch(
        f"/conversations/{conversation_id}",
        json={"title": "  Robotics   Notes  "},
    )

    assert response.status_code == 200
    assert response.json() == {
        "conversation_id": conversation_id,
        "title": "Robotics Notes",
    }

    conversation = next(
        item
        for item in database.list_conversations()
        if item["id"] == conversation_id
    )
    assert conversation["title"] == "Robotics Notes"


def test_rename_conversation_endpoint_rejects_blank_title(client):
    conversation_id = "api-blank-title-test"
    database.create_conversation(conversation_id)

    response = client.patch(
        f"/conversations/{conversation_id}",
        json={"title": "   "},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Title cannot be blank."


def test_rename_conversation_endpoint_rejects_long_title(client):
    conversation_id = "api-long-title-test"
    database.create_conversation(conversation_id)

    response = client.patch(
        f"/conversations/{conversation_id}",
        json={"title": "A" * 61},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Title cannot exceed 60 characters."
    )


def test_rename_unknown_conversation_returns_404(client):
    response = client.patch(
        "/conversations/missing-conversation",
        json={"title": "New title"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation does not exist."