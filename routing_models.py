
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class RouteInput(BaseModel):
    type: Literal["document", "general"]
    question: str = Field(min_length=1)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question must not be empty or whitespace.")
        return value.strip()
