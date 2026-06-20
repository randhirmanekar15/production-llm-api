# Production-Ready LLM API

![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue) ![License: MIT](https://img.shields.io/badge/License-MIT-green) ![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)

**Serve an open-source small model (TinyLlama) behind a clean, validated FastAPI service — the LLMOps "make it shippable" project.**

## Overview

In 2026, the skill bar moved. Two years ago, "can you prompt a model?" was enough to put on a resume. Today that's table stakes. The question that actually gets you hired is different: *can you serve a model reliably?* Prompting lives in a notebook. Serving lives in production — with validation, health checks, predictable latency, and a contract other systems can depend on. That gap is LLMOps, and this project lives squarely inside it.

Self-hosting used to mean a GPU bill that only made sense for a funded team. That's no longer true. Small open-weight models like `TinyLlama/TinyLlama-1.1B-Chat-v1.0` run on modest hardware, which means you can own the whole stack — no per-token API bill, no vendor lock-in, no data leaving your box. The trade-off isn't capability anymore; it's whether you can wrap that model in something a real client can call without it falling over.

This service is that wrapper. It loads the model **once** on startup, exposes a typed `POST /generate` endpoint with hard bounds on every input, reports token usage on every response, and ships a `/health` endpoint so an orchestrator can tell whether the thing is actually ready. Three files, one job each. Nothing clever — just the boring reliability work that separates a demo from a deployment.

## Features

- **Load-once model lifecycle** — the model loads a single time on startup via FastAPI's `lifespan`, not per-request.
- **Strict input validation** — Pydantic enforces bounds as cost/latency guardrails: non-empty prompts, `max_tokens` capped, `temperature` clamped.
- **Token usage in every response** — see the token count so you can reason about cost and context limits.
- **Health endpoint** — `GET /health` for liveness/readiness checks behind a load balancer or container probe.
- **Self-documenting** — interactive Swagger UI at `/docs`, generated from the Pydantic schemas.
- **Clean three-file architecture** — engine, schemas, and app separated so each is testable and swappable.

## How it works

The codebase is deliberately split into three files, each with one responsibility:

- **`ml_engine.py`** — owns the model. Loads `TinyLlama` once on startup and exposes `generate()`. No HTTP concerns leak in here.
- **`schemas.py`** — owns the contract. Pydantic models with hard bounds: `prompt` (`min_length=1`), `max_tokens` (`10–1024`), `temperature` (`0.0–1.0`). Invalid requests are rejected before any compute is spent.
- **`main.py`** — owns the wiring. The FastAPI app, an `asynccontextmanager` lifespan that loads the model at boot, and the routes (`GET /`, `GET /health`, `POST /generate`).

```text
HTTP request
     │
     ▼
Pydantic validate   ──►  reject (422) if out of bounds
     │
     ▼
engine.generate()        (model already resident from lifespan)
     │
     ▼
Response  { result, token_usage }
```

## Tech stack

| Layer | Choice |
|-------|--------|
| API framework | FastAPI |
| ASGI server | Uvicorn |
| Inference | PyTorch |
| Model loading | Transformers |
| Validation | Pydantic |
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |

## Project structure

```text
production-llm-api/
├── ml_engine.py        # loads the model ONCE on startup; generate()
├── schemas.py          # Pydantic request/response models + validation bounds
├── main.py             # FastAPI app, lifespan, routes (/, /health, /generate)
├── test_schemas.py     # tests for the validation contract
├── requirements.txt
├── ARTICLE.md
├── LICENSE
└── README.md
```

## Installation

```bash
git clone https://github.com/randhirmanekar15/production-llm-api.git
cd production-llm-api
pip install -r requirements.txt
```

The model weights download automatically from the Hugging Face Hub on first startup.

## Usage

Start the dev server:

```bash
fastapi dev main.py
```

Open the interactive Swagger UI at `http://127.0.0.1:8000/docs`, or call the endpoint directly:

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain LLMOps in one sentence.", "max_tokens": 128, "temperature": 0.7}'
```

Example response:

```json
{ "result": "LLMOps is the practice of deploying, serving, and monitoring LLMs reliably in production.", "token_usage": 16 }
```

## API

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service status |
| GET | `/health` | Liveness/readiness + `model_loaded` |
| POST | `/generate` | Generate text from a prompt |

**`POST /generate` request fields**

| Field | Type | Bounds | Description |
|-------|------|--------|-------------|
| `prompt` | string | non-empty (not whitespace-only) | Input prompt |
| `max_tokens` | int | `10`–`1024` | Max tokens to generate |
| `temperature` | float | `>0.0`–`1.0` | Sampling temperature |
| `top_k` | int | `1`–`200` | Top-k sampling cutoff |
| `top_p` | float | `>0.0`–`1.0` | Nucleus sampling cutoff |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Hugging Face model identifier |

## Testing

Validation is the contract, so it's what gets tested. `test_schemas.py` covers the Pydantic bounds — empty prompts, out-of-range `max_tokens`, and `temperature` outside `0.0–1.0` are all rejected. No model load required.

```bash
pip install pytest
pytest
```

## Limitations

- **No auth or rate limiting by default** — add both before exposing this to the open internet.
- **Single process won't scale** — for real load, run replicas or a dedicated serving stack like vLLM.
- **Throughput is serialized.** FastAPI runs the sync `/generate` endpoint in a threadpool, so one request doesn't block the event loop — but the threadpool caps concurrency and a single resident model still serializes heavy inference. Scale out with replicas or a dedicated serving stack.
- **Memory ceiling** — TinyLlama fits modest hardware; larger models will hit your RAM/VRAM limits.

## Roadmap

- [ ] Auth + rate limiting
- [ ] Dockerfile for containerized deploys
- [ ] vLLM backend for throughput
- [ ] Async worker so inference doesn't block the event loop

## Credits

📖 Full write-up: [ARTICLE.md](ARTICLE.md).

Based on Aman Kharwal's tutorial, ["Build a Production-Ready LLM API"](https://amanxai.com/2026/02/11/build-a-production-ready-llm-api/).

**What I changed vs the source tutorial:**

- Added **token-usage reporting** in the response so cost and context limits are visible per call.
- Added a **`/health` endpoint** for container/load-balancer probes.
- Added **validation bounds** (`prompt`, `max_tokens`, `temperature`) as explicit cost/latency guardrails.

## Author

Built by **Randhir Manekar** — [randhirmanekar.com](https://randhirmanekar.com) · [github.com/randhirmanekar15](https://github.com/randhirmanekar15)

## License

MIT — see [LICENSE](LICENSE).
