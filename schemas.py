"""Request/response contracts with validation."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class GenerationRequest(BaseModel):
    """Validated input for the /generate endpoint."""

    prompt: str = Field(..., min_length=1, examples=["Explain quantum physics simply."])
    max_tokens: int = Field(256, ge=10, le=1024, examples=[128])
    # gt=0.0: the model samples with do_sample=True, where temperature/top_p must be > 0.
    temperature: float = Field(0.7, gt=0.0, le=1.0, examples=[0.7])
    top_k: int = Field(50, ge=1, le=200)
    top_p: float = Field(0.95, gt=0.0, le=1.0)

    @field_validator("prompt")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("prompt must not be empty or whitespace-only")
        return value


class GenerationResponse(BaseModel):
    """Response returned by /generate."""

    result: str
    token_usage: int
