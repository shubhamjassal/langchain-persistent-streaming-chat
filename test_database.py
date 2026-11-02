import pytest
import database


def test_conversation_persistence(tmp_path, monkeypatch):
    # Use an isolated test database, not your real database.
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()

    conversation_id = "test-conversation-1"
    database.create_conversation(conversation_id)

    database.save_message(
        conversation_id, "user", "What is robotics?"
    )
    database.save_message(
        conversation_id, "assistant", "Robotics is the study of robots."
    )

    # Retrieve the saved messages.
    messages = database.get_conversation_messages(conversation_id)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is robotics?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "Robotics is the study of robots."


def test_conversations_are_isolated(tmp_path, monkeypatch):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()

    database.create_conversation("chat-1")
    database.create_conversation("chat-2")

    database.save_message("chat-1", "user", "Question for chat one.")
    database.save_message("chat-2", "user", "Question for chat two.")

    chat_one = database.get_conversation_messages("chat-1")
    chat_two = database.get_conversation_messages("chat-2")

    assert len(chat_one) == 1
    assert len(chat_two) == 1
    assert chat_one[0]["content"] == "Question for chat one."
    assert chat_two[0]["content"] == "Question for chat two."


def test_unknown_conversation_raises_error(tmp_path, monkeypatch):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()

    try:
        database.get_conversation_messages("unknown-chat")
    except ValueError as exc:
        assert str(exc) == "Conversation does not exist."
    else:
        raise AssertionError("Expected ValueError for unknown conversation.")


def test_save_exchange_rolls_back_if_assistant_message_fails(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    conversation_id = "rollback-test"
    database.create_conversation(conversation_id)

    # Make the assistant-message insert fail at the database level.
    with database.get_connection() as connection:
        connection.execute(
            """
            CREATE TRIGGER fail_assistant_insert
            BEFORE INSERT ON messages
            WHEN NEW.role = 'assistant'
            BEGIN
                SELECT RAISE(ABORT, 'Simulated assistant insert failure');
            END
            """
        )

    try:
        database.save_exchange(
            conversation_id,
            "What is robotics?",
            "Robotics is the study of robots.",
        )
    except Exception as exc:
        assert "Simulated assistant insert failure" in str(exc)
    else:
        raise AssertionError("Expected the exchange to fail.")

    messages = database.get_conversation_messages(conversation_id)
    assert messages == []



@pytest.mark.parametrize(
    ("question", "answer", "expected_error"),
    [
        ("   ", "A valid answer.", "Question cannot be blank."),
        ("A valid question.", "\t\n", "Answer cannot be blank."),
    ],
)
def test_save_exchange_rejects_blank_content(
    tmp_path,
    monkeypatch,
    question,
    answer,
    expected_error,
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    conversation_id = "blank-content-test"
    database.create_conversation(conversation_id)

    with pytest.raises(ValueError, match=expected_error):
        database.save_exchange(conversation_id, question, answer)

    assert database.get_conversation_messages(conversation_id) == []


def test_save_exchange_sets_title_from_first_question(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    conversation_id = "title-test"
    database.create_conversation(conversation_id)

    database.save_exchange(
        conversation_id,
        "  What   is robotics?  ",
        "Robotics is the study of robots.",
    )

    conversations = database.list_conversations()
    conversation = next(
        item for item in conversations
        if item["id"] == conversation_id
    )

    assert conversation["title"] == "What is robotics?"


def test_save_exchange_limits_title_to_60_characters(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    conversation_id = "long-title-test"
    database.create_conversation(conversation_id)

    question = "What is robotics and how does it work in modern industries today?"

    database.save_exchange(
        conversation_id,
        question,
        "Robotics combines engineering and computer science.",
    )

    conversations = database.list_conversations()
    conversation = next(
        item for item in conversations
        if item["id"] == conversation_id
    )

    assert 0 < len(conversation["title"]) <= 60
    expected_title = " ".join(question.split())[:60].rsplit(" ", 1)[0]

    assert conversation["title"] == expected_title


def test_save_exchange_preserves_custom_title(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    conversation_id = "custom-title-test"
    database.create_conversation(conversation_id)

    with database.get_connection() as connection:
        connection.execute(
            "UPDATE conversations SET title = ? WHERE id = ?",
            ("My Robotics Project", conversation_id),
        )

    database.save_exchange(
        conversation_id,
        "What is robotics?",
        "Robotics is the study of robots.",
    )

    conversations = database.list_conversations()
    conversation = next(
        item for item in conversations
        if item["id"] == conversation_id
    )

    assert conversation["title"] == "My Robotics Project"


def test_save_exchange_handles_single_word_longer_than_60_characters(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    conversation_id = "single-long-word-test"
    database.create_conversation(conversation_id)

    question = "R" * 80

    database.save_exchange(
        conversation_id,
        question,
        "Here is the answer.",
    )

    conversation = next(
        item
        for item in database.list_conversations()
        if item["id"] == conversation_id
    )

    assert conversation["title"] == "R" * 60
    assert len(conversation["title"]) == 60


def test_save_exchange_does_not_overwrite_first_question_title(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    conversation_id = "stable-title-test"
    database.create_conversation(conversation_id)

    database.save_exchange(
        conversation_id,
        "What is robotics?",
        "Robotics is the study of robots.",
    )

    database.save_exchange(
        conversation_id,
        "Explain machine learning in detail.",
        "Machine learning learns patterns from data.",
    )

    conversation = next(
        item
        for item in database.list_conversations()
        if item["id"] == conversation_id
    )

    assert conversation["title"] == "What is robotics?"
    assert conversation["message_count"] == 4


def test_rename_conversation_updates_title(tmp_path, monkeypatch):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    database.create_conversation("rename-test")

    database.rename_conversation(
        "rename-test",
        "  Robotics   Learning Notes  ",
    )

    conversation = next(
        item
        for item in database.list_conversations()
        if item["id"] == "rename-test"
    )

    assert conversation["title"] == "Robotics Learning Notes"


@pytest.mark.parametrize(
    "invalid_title",
    ["", "   ", "\t\n"],
)
def test_rename_conversation_rejects_blank_title(
    tmp_path, monkeypatch, invalid_title
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    database.create_conversation("rename-test")

    with pytest.raises(ValueError, match="Title cannot be blank."):
        database.rename_conversation("rename-test", invalid_title)


def test_rename_conversation_rejects_title_over_60_characters(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()
    database.create_conversation("rename-test")

    with pytest.raises(
        ValueError,
        match="Title cannot exceed 60 characters.",
    ):
        database.rename_conversation("rename-test", "A" * 61)


def test_rename_unknown_conversation_raises_error(
    tmp_path, monkeypatch
):
    test_db = tmp_path / "test_chat_history.db"
    monkeypatch.setattr(database, "DATABASE_PATH", test_db)

    database.initialize_database()

    with pytest.raises(
        ValueError,
        match="Conversation does not exist.",
    ):
        database.rename_conversation("missing-chat", "New title")
