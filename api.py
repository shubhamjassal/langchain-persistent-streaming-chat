import json
import logging

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from fastapi.responses import StreamingResponse

from routing_with_chroma import router, model
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, AIMessage

from typing import Literal

from pydantic import BaseModel, Field, model_validator
from memory import trim_history

from uuid import uuid4

from database import (
    create_conversation,
    delete_conversation,
    get_conversation_messages,
    initialize_database,
    list_conversations,
    save_exchange,
    rename_conversation,
)

from database import (
    create_conversation,
    delete_conversation,
    get_conversation_details,
    get_conversation_messages,
    initialize_database,
    list_conversations,
    save_exchange,
    rename_conversation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    initialize_database()
    yield


app = FastAPI(
    title="LangChain Question Answering API",
    description="API for routing questions to Gemini or ChromaDB.",
    version="1.0.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "null",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type"],
)

class QuestionRequest(BaseModel):
    type: Literal["general", "document"] = Field(
        default="general",
        description="Question route: general or document",
    )
    question: str = Field(
        min_length=1,
        description="The question to answer",
    )



class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)




class ChatRequest(BaseModel):
    conversation_id: str = Field(min_length=1, max_length=100)
    type: Literal["general", "document"] = "general"
    question: str = Field(min_length=1, max_length=4000)

    @model_validator(mode="after")
    def validate_question(self):
        if not self.question.strip():
            raise ValueError("Question cannot be blank.")
        return self

class RenameConversationRequest(BaseModel):
    title: str


@app.post("/conversations", status_code=201)
def create_new_conversation():
    conversation_id = str(uuid4())
    create_conversation(conversation_id)

    return {"conversation_id": conversation_id}



@app.get("/conversations")
def get_conversations(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    search: str | None = Query(default=None, max_length=200),
):
    conversations = list_conversations(
        limit=limit + 1,
        offset=offset,
        search_query=search,
    )

    has_more = len(conversations) > limit

    return {
        "conversations": conversations[:limit],
        "limit": limit,
        "offset": offset,
        "has_more": has_more,
    }


@app.patch("/conversations/{conversation_id}")
def rename_conversation_endpoint(
    conversation_id: str,
    request: RenameConversationRequest,
):
    try:
        rename_conversation(conversation_id, request.title)
    except ValueError as exc:
        message = str(exc)

        if message == "Conversation does not exist.":
            raise HTTPException(status_code=404, detail=message)

        raise HTTPException(status_code=422, detail=message)

    return {
        "conversation_id": conversation_id,
        "title": " ".join(request.title.split()),
    }


@app.get("/conversations/{conversation_id}/messages")
def get_conversation_history(conversation_id: str):
    try:
        messages = get_conversation_messages(conversation_id)
        return {"conversation_id": conversation_id, "messages": messages}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.delete("/conversations/{conversation_id}")
def remove_conversation(conversation_id: str):
    try:
        # Verify it exists before deleting.
        get_conversation_messages(conversation_id)
        delete_conversation(conversation_id)
        return {"deleted": True, "conversation_id": conversation_id}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/ask")
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if request.type != "general":
        raise HTTPException(
            status_code=400,
            detail="Streaming currently supports general questions only.",
    )

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty.",
        )


    try:
        result = router.invoke(
            {
                "type": request.type,
                "question": question,
            }
        )
        return result

    except Exception:
        logger.exception("Failed to process question")
        raise HTTPException(
            status_code=500,
            detail="Unable to process the question right now.",
        )


@app.post("/ask/stream")


@app.post("/ask/stream")
def ask_stream(request: ChatRequest):
    question = request.question.strip()

    if request.type != "general":
        raise HTTPException(
            status_code=400,
            detail="Streaming currently supports general questions only.",
        )

    try:
        stored_messages = get_conversation_messages(
            request.conversation_id
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Convert saved database records into validated chat messages.
    try:
        history = [
            ChatMessage(
                role=item["role"],
                content=item["content"],
            )
            for item in stored_messages
        ]

        history = trim_history(history)

    except ValueError as exc:
        logger.exception("Stored conversation history is invalid")
        raise HTTPException(
            status_code=500,
            detail="Unable to load conversation history.",
        ) from exc

    messages = []

    for item in history:
        content = item.content.strip()

        if item.role == "user":
            messages.append(HumanMessage(content=content))
        else:
            messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=question))

    def generate():
        answer_parts = []

        try:
            for chunk in model.stream(messages):
                if isinstance(chunk.content, str):
                    text = chunk.content

                    if text:
                        answer_parts.append(text)
                        yield (
                            "event: token\n"
                            f"data: {json.dumps(text)}\n\n"
                        )

                else:
                    for block in chunk.content:
                        if (
                            isinstance(block, dict)
                            and block.get("type") == "text"
                            and block.get("text")
                        ):
                            text = block["text"]
                            answer_parts.append(text)

                            yield (
                                "event: token\n"
                                f"data: {json.dumps(text)}\n\n"
                            )

            answer = "".join(answer_parts)

            # Persist only after generation finishes successfully.
            save_exchange(
                request.conversation_id,
                question,
                answer,
            )

            yield "event: done\ndata: [DONE]\n\n"

        except Exception:
            logger.exception("Streaming response or persistence failed")

            error_data = json.dumps(
                {"message": "Unable to complete and save the response."}
            )
            yield f"event: error\ndata: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/conversations/{conversation_id}")
def get_conversation_details_endpoint(conversation_id: str):
    conversation = get_conversation_details(conversation_id)

    if conversation is None:
        raise HTTPException(
            status_code=404,
            detail="Conversation does not exist.",
        )

    return conversation
