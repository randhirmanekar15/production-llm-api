# Production-Ready LLM API

Serve an open-source small model (TinyLlama) behind a clean FastAPI service: module separation, Pydantic validation, and a load-once-on-startup lifecycle. The "make it shippable" project — engineering over prompting.

## Stack

| Piece | Choice |
|-------|--------|
| API | FastAPI + Uvicorn |
| Inference | PyTorch + Transformers |
| Model | TinyLlama-1.1B-Chat |
| Validation | Pydantic |

## Layout

- `ml_engine.py` — loads the model once, generates
- `schemas.py` — request/response contracts with bounds
- `main.py` — FastAPI app, lifespan, endpoints

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
fastapi dev main.py
# then open http://127.0.0.1:8000/docs  (Swagger UI)
```

Endpoints: `GET /` (status), `GET /health` (model loaded?), `POST /generate`.

## Test

```bash
pip install pytest
pytest        # validation tests, no model load
```

## Limitations

- No auth or rate limit by default — add an API key + throttle before exposing it.
- Single process won't scale; use replicas or a dedicated inference server (e.g. vLLM) for real load.
- `/generate` is sync `def`; offload blocking inference to a worker for concurrency.

---

Inspired by Aman Kharwal's tutorial, [Build a Production-Ready LLM API](https://amanxai.com/2026/02/11/build-a-production-ready-llm-api/). Rebuilt and extended (token usage, `/health`, validation bounds).

MIT licensed.
