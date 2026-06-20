"""FastAPI service that serves a local LLM.

Run with:  fastapi dev main.py
Docs at:   http://127.0.0.1:8000/docs
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from ml_engine import llm_engine
from schemas import GenerationRequest, GenerationResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup."""
    llm_engine.load_model()
    yield
    print("Shutting down model engine.")


app = FastAPI(title="Local LLM API", lifespan=lifespan)


@app.get("/")
def read_root() -> dict[str, str]:
    return {"status": "online"}


@app.get("/health")
def health() -> dict[str, object]:
    return {"status": "ok", "model_loaded": llm_engine.pipe is not None}


@app.post("/generate", response_model=GenerationResponse)
def generate_text(request: GenerationRequest) -> GenerationResponse:
    try:
        result = llm_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
        )
    except Exception as exc:  # noqa: BLE001  surface a clean 500
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return GenerationResponse(result=result, token_usage=len(result.split()))
