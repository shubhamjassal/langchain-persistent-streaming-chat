from langchain_core.runnables import RunnableBranch, RunnableLambda
from routing_logic import build_router


def test_response_structure():
    result = {
        "answer": "RAG stands for Retrieval-Augmented Generation.",
        "source": "chroma",
        "status": "success",
        "sources": ["rag-guide"],
    }

    assert "answer" in result
    assert "source" in result
    assert "status" in result
    assert "sources" in result

    assert result["source"] == "chroma"
    assert result["status"] == "success"
    assert result["sources"] == ["rag-guide"]


def test_rejection_response():
    result = {
        "answer": (
            "I couldn't find sufficiently relevant information "
            "in the available documents to answer this question."
        ),
        "source": "chroma",
        "status": "no_relevant_context",
        "sources": [],
    }

    assert result["status"] == "no_relevant_context"
    assert result["sources"] == []
    assert result["answer"] != ""

def test_routing_decision():
    is_document_question = lambda data: data["type"] == "document"

    assert is_document_question(
        {"type": "document", "question": "What is RAG?"}
    ) is True

    assert is_document_question(
        {"type": "general", "question": "Why is Python popular?"}
    ) is False

def test_runnable_branch_routes_correctly():
    document_chain = RunnableLambda(
        lambda data: {
            "answer": "Fake document answer",
            "source": "chroma",
            "status": "success",
            "sources": ["rag-guide"],
        }
    )

    general_chain = RunnableLambda(
        lambda data: {
            "answer": "Fake general answer",
            "source": "gemini",
            "status": "success",
            "sources": [],
        }
    )

    router = build_router(document_chain, general_chain)

    document_result = router.invoke({
    "type": "document",
    "question": "What is RAG?",
})

    general_result = router.invoke({
    "type": "general",
    "question": "Why is Python popular?",
})

    assert document_result["source"] == "chroma"
    assert document_result["sources"] == ["rag-guide"]

    assert general_result["source"] == "gemini"
    assert general_result["sources"] == []



def test_unknown_type_returns_standardized_error():
    document_chain = RunnableLambda(
        lambda data: {"source": "chroma"}
    )
    general_chain = RunnableLambda(
        lambda data: {"source": "gemini"}
    )

    router = build_router(document_chain, general_chain)

    result = router.invoke({
        "type": "unknown",
        "question": "What is RAG?",
    })

    assert result["source"] == "router"
    assert result["status"] == "invalid_input"
    assert result["sources"] == []
    assert result["answer"].startswith("Invalid input:")



def test_missing_type_returns_standardized_error():
    document_chain = RunnableLambda(
        lambda data: {"source": "chroma"}
    )
    general_chain = RunnableLambda(
        lambda data: {"source": "gemini"}
    )

    router = build_router(document_chain, general_chain)

    result = router.invoke({"question": "What is RAG?"})

    assert result["source"] == "router"
    assert result["status"] == "invalid_input"
    assert result["sources"] == []
    assert result["answer"].startswith("Invalid input:")

def test_empty_question_returns_standardized_error():
    document_chain = RunnableLambda(
        lambda data: {"source": "chroma"}
    )
    general_chain = RunnableLambda(
        lambda data: {"source": "gemini"}
    )

    router = build_router(document_chain, general_chain)

    result = router.invoke({
        "type": "document",
        "question": "",
    })

    assert result["source"] == "router"
    assert result["status"] == "invalid_input"
    assert result["sources"] == []
    assert result["answer"].startswith("Invalid input:")


def test_whitespace_question_returns_standardized_error():
    document_chain = RunnableLambda(
        lambda data: {"source": "chroma"}
    )
    general_chain = RunnableLambda(
        lambda data: {"source": "gemini"}
    )

    router = build_router(document_chain, general_chain)

    result = router.invoke({
        "type": "document",
        "question": "   ",
    })

    assert result["source"] == "router"
    assert result["status"] == "invalid_input"
    assert result["sources"] == []
    assert result["answer"].startswith("Invalid input:")


def test_valid_question_is_trimmed():
    from routing_models import RouteInput

    result = RouteInput(
        type="document",
        question="   What is RAG?   ",
    )

    assert result.question == "What is RAG?"


def test_invalid_input_does_not_call_either_chain():
    calls = {"document": 0, "general": 0}

    def document_handler(data):
        calls["document"] += 1
        return {"source": "chroma"}

    def general_handler(data):
        calls["general"] += 1
        return {"source": "gemini"}

    document_chain = RunnableLambda(document_handler)
    general_chain = RunnableLambda(general_handler)

    router = build_router(document_chain, general_chain)

    result = router.invoke({
        "type": "document",
        "question": "",
    })

    assert result["status"] == "invalid_input"
    assert calls == {"document": 0, "general": 0}


def test_retry_succeeds_after_transient_failures():
    from langchain_core.runnables import RunnableLambda

    calls = {"count": 0}

    def flaky_operation(data):
        calls["count"] += 1

        if calls["count"] < 3:
            raise ValueError("Temporary failure")

        return {"answer": "Success"}

    retrying_chain = RunnableLambda(flaky_operation).with_retry(
        retry_if_exception_type=(ValueError,),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )

    result = retrying_chain.invoke({"question": "Test"})

    assert result["answer"] == "Success"
    assert calls["count"] == 3


def test_retry_raises_after_all_attempts_fail():
    from langchain_core.runnables import RunnableLambda

    calls = {"count": 0}

    def always_fails(data):
        calls["count"] += 1
        raise ValueError("Permanent failure")

    retrying_chain = RunnableLambda(always_fails).with_retry(
        retry_if_exception_type=(ValueError,),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )

    import pytest

    with pytest.raises(ValueError, match="Permanent failure"):
        retrying_chain.invoke({"question": "Test"})

    assert calls["count"] == 3

def test_retry_does_not_retry_permanent_error():
    from langchain_core.runnables import RunnableLambda
    from langchain_core.exceptions import ModelAuthenticationError
    import pytest

    calls = {"count": 0}

    def authentication_failure(data):
        calls["count"] += 1
        raise ModelAuthenticationError("Invalid API credentials")

    retrying_chain = RunnableLambda(authentication_failure).with_retry(
        retry_if_exception_type=(ConnectionError, TimeoutError),
        stop_after_attempt=3,
        wait_exponential_jitter=False,
    )

    with pytest.raises(ModelAuthenticationError, match="Invalid API credentials"):
        retrying_chain.invoke({"question": "Test"})

    assert calls["count"] == 1

def process_questions(questions, router):
    for question in questions:
        print(f"\nQuestion: {question['question']}")

        try:
            result = router.invoke(question)

            print("Answer:", result["answer"])
            print("Source:", result["source"])
            print("Status:", result["status"])
            print("Sources:", result.get("sources", []))

        except Exception as error:
            print("Status: error")
            print(f"Error: {error}")
            continue

def test_process_questions_continues_after_failure(capsys):
    from routing_with_chroma import process_questions

    class FakeRouter:
        def invoke(self, question):
            if question["question"] == "First question":
                raise ValueError("Simulated failure")

            return {
                "answer": "Second question answered",
                "source": "gemini",
                "status": "success",
                "sources": [],
            }

    questions = [
        {"type": "general", "question": "First question"},
        {"type": "general", "question": "Second question"},
    ]

    process_questions(questions, FakeRouter())

    output = capsys.readouterr().out

    assert "Status: error" in output
    assert "Simulated failure" in output
    assert "Second question answered" in output
    assert "Status: success" in output

if __name__ == "__main__":
    questions = [
        {
            "type": "document",
            "question": "What does RAG stand for?",
        },
        {
            "type": "document",
            "question": "What is the capital of France?",
        },
        {
            "type": "general",
            "question": "Why is Python popular for AI development?",
        },
    ]

    process_questions(questions, router)