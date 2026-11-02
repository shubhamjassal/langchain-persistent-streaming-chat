

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

DATABASE_PATH = Path(
    os.getenv(
        "CHAT_HISTORY_DB",
        str(Path(__file__).resolve().parent / "chat_history.db"),
    )
)


@contextmanager
def get_connection():
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        with connection:
            yield connection
    finally:
        connection.close()



def initialize_database():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                title TEXT NOT NULL DEFAULT 'New conversation'
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL
                    CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT
                    (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                FOREIGN KEY (conversation_id)
                    REFERENCES conversations(id)
            )
        """)


def create_conversation(conversation_id: str):
    with get_connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id) VALUES (?)",
            (conversation_id,),
        )

def rename_conversation(conversation_id: str, new_title: str):
    """Rename an existing conversation."""
    title = " ".join(new_title.split())

    if not title:
        raise ValueError("Title cannot be blank.")

    if len(title) > 60:
        raise ValueError("Title cannot exceed 60 characters.")

    with get_connection() as connection:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?
            WHERE id = ?
            """,
            (title, conversation_id),
        )

        if cursor.rowcount == 0:
            raise ValueError("Conversation does not exist.")


def save_message(
    conversation_id: str,
    role: str,
    content: str,
):
    if role not in {"user", "assistant"}:
        raise ValueError("Invalid message role.")

    with get_connection() as connection:
        conversation = connection.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            raise ValueError("Conversation does not exist.")

        connection.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            (conversation_id, role, content),
        )


def get_conversation_messages(conversation_id: str) -> list[dict]:
    with get_connection() as connection:
        conversation = connection.execute(
            "SELECT id FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            raise ValueError("Conversation does not exist.")

        rows = connection.execute(
            """
            SELECT role, content, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
            """,
            (conversation_id,),
        ).fetchall()

        return [dict(row) for row in rows]



def list_conversations(
    limit: int = 20,
    offset: int = 0,
    search_query: str | None = None,
) -> list[dict]:
    with get_connection() as connection:
        if search_query and search_query.strip():
            # Treat %, _ and \ as literal search characters.
            escaped_query = (
                search_query.strip()
                .replace("\\", "\\\\")
                .replace("%", "\\%")
                .replace("_", "\\_")
            )
            search_pattern = f"%{escaped_query}%"

            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    COUNT(m.id) AS message_count
                FROM conversations AS c
                LEFT JOIN messages AS m
                    ON m.conversation_id = c.id
                WHERE
                    c.title LIKE ? ESCAPE '\\'
                    OR EXISTS (
                        SELECT 1
                        FROM messages AS matching_message
                        WHERE matching_message.conversation_id = c.id
                          AND matching_message.content LIKE ? ESCAPE '\\'
                    )
                GROUP BY c.id
                ORDER BY c.created_at DESC, c.id DESC
                LIMIT ? OFFSET ?
                """,
                (
                    search_pattern,
                    search_pattern,
                    limit,
                    offset,
                ),
            ).fetchall()

        else:
            rows = connection.execute(
                """
                SELECT
                    c.id,
                    c.title,
                    c.created_at,
                    COUNT(m.id) AS message_count
                FROM conversations AS c
                LEFT JOIN messages AS m
                    ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.created_at DESC, c.id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()

        return [dict(row) for row in rows]

def get_conversation_details(conversation_id: str) -> dict | None:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                COUNT(m.id) AS message_count
            FROM conversations AS c
            LEFT JOIN messages AS m
                ON m.conversation_id = c.id
            WHERE c.id = ?
            GROUP BY c.id
            """,
            (conversation_id,),
        ).fetchone()

        return dict(row) if row else None

def delete_conversation(conversation_id: str):
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        )
        connection.execute(
            "DELETE FROM conversations WHERE id = ?",
            (conversation_id,),
        )





def save_exchange(
    conversation_id: str,
    question: str,
    answer: str,
):
    """Save a valid user-assistant exchange atomically."""
    if not question.strip():
        raise ValueError("Question cannot be blank.")

    if not answer.strip():
        raise ValueError("Answer cannot be blank.")

    with get_connection() as connection:
        conversation = connection.execute(
            "SELECT id, title FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()

        if conversation is None:
            raise ValueError("Conversation does not exist.")

        connection.executemany(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (?, ?, ?)
            """,
            [
                (conversation_id, "user", question),
                (conversation_id, "assistant", answer),
            ],
        )

        if conversation["title"] == "New conversation":
            normalized_question = " ".join(question.split())
            title = normalized_question[:60]

            if len(normalized_question) > 60 and " " in title:
                title = title.rsplit(" ", 1)[0]

            connection.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE id = ?
                """,
                (title, conversation_id),
            )