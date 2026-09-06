from datetime import datetime, timezone
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


DEFAULT_WORKSPACE_ID = "default"
MAX_CAPTURE_CHARACTERS = 20_000


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(BaseModel):
    """A locally persisted conversation container; chat is not implemented yet."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    workspace_id: str = DEFAULT_WORKSPACE_ID
    title: str = Field(min_length=1, max_length=120)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("title")
    @classmethod
    def title_is_not_whitespace(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("title must not be blank")
        return value


class Capture(BaseModel):
    """Source text entered manually in Phase 1; it has no generated result."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    conversation_id: str
    source_text: str = Field(min_length=1, max_length=MAX_CAPTURE_CHARACTERS)
    created_at: datetime = Field(default_factory=utc_now)

    @field_validator("source_text")
    @classmethod
    def source_text_is_not_whitespace(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("source_text must not be blank")
        return value
