# I shipped an LLM API in three files — and that's the skill that actually gets you hired in 2026

*Anyone can paste a prompt into ChatGPT. Far fewer can serve a model behind a clean API that doesn't fall over. Here's the second thing, built small enough to run on your laptop.*

## Why this, why now

The bar moved. In 2024, "can you prompt well" was a resume line. In 2026, it's table stakes — and nobody's paying for it.

What employers actually want now is the boring part: can you *serve* a model reliably? Validate inputs? Load weights once instead of on every request? Add observability so you know what's happening in production? That's LLMOps, and it's where the salaries are.

Two things made this project realistic. First, small open models grew up. TinyLlama is 1.1B parameters — it fits in memory you actually have, no $40k GPU lease, no per-token API bill. Self-hosting stopped being a fantasy.

Second, the industry mood shifted. Gartner's Hype Cycle for Agentic AI now plants cost, governance, and reliability right next to raw capability. Translation: "it works in a notebook" is no longer the finish line. "It works when 50 people hit it" is.

So I built the smallest thing that demonstrates I understand that difference.

## What it does

It's a FastAPI service that serves TinyLlama behind a clean HTTP endpoint.

You POST a prompt, it runs the model through a proper chat template, samples a response with the generation knobs you'd expect (temperature, top_k, top_p), and returns the text plus a token-usage count. There's a status route at `/`, a dedicated `/health` readiness check, and auto-generated Swagger docs at `/docs`.

The model loads once, on startup. Not per request. That single decision is the difference between a toy and something shippable.

## The stack

| Layer | Choice | Why |
|---|---|---|
| API framework | FastAPI | Async, Pydantic-native, free OpenAPI docs |
| Server | Uvicorn | ASGI, the FastAPI default |
| Inference | PyTorch + Transformers | The Hugging Face standard |
| Model | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | Small enough to self-host, chat-tuned |
| Validation | Pydantic | Type-safe request/response contracts |

## How it works

The whole thing is three files, and the separation is the point.

`ml_engine.py` owns the model. `schemas.py` owns the data contracts. `main.py` owns the HTTP layer. Each file has one job, so I can swap the model without touching routing, or tighten validation without touching inference.

The engine loads weights once and holds them:

```python
class LLMEngine:
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate(self, prompt, max_tokens, temperature, top_k, top_p):
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        ).to(self.model.device)
        out = self.model.generate(
            inputs, max_new_tokens=max_tokens, do_sample=True,
            temperature=temperature, top_k=top_k, top_p=top_p,
        )
        text = self.tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)
        return text, out.shape[-1]
```

The API layer wires it together with a lifespan handler — model loads when the app boots, not on the first poor user's request:

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    engine.load_model()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/generate", response_model=GenerationResponse)
def generate(req: GenerationRequest):
    text, tokens = engine.generate(
        req.prompt, req.max_tokens, req.temperature, req.top_k, req.top_p
    )
    return GenerationResponse(result=text, token_usage=tokens)
```

Pydantic does the gatekeeping before any of that runs. `prompt` needs `min_length=1`, `max_tokens` is bounded `10–1024`, `temperature` is clamped `0.0–1.0`. Garbage requests get a clean 422 instead of a stack trace.

## What I changed

I didn't copy the tutorial. I extended it where a real deployment would force me to:

1. **Real token usage in the response.** I return `token_usage` computed with the model's own tokenizer (not a word count), so a caller can track cost and I can spot a runaway prompt before it eats my GPU.
2. **Hard validation limits.** The `max_tokens` ceiling at 1024 isn't cosmetic — it's a cost and latency guardrail baked into the contract, not buried in a comment. Prompts must also be non-empty (not whitespace-only), and `temperature`/`top_p` must be > 0 because the model samples.
3. **A real `/health` endpoint.** Returns status and whether the model is loaded — separate from the `/` status route. That's the first thing any orchestrator or load balancer asks before sending traffic.
4. **Safe error responses.** Failures are logged server-side but return a generic message to the client, so raw stack traces and internals never leak through the API. (Dockerization and a vLLM backend are on the roadmap, not done yet.)

## Where it breaks

Honesty matters more than a polished demo, so here's what I'd fix before calling this production:

- **No auth, no rate limit.** Anyone who finds the URL can drain my compute. You need an API key and a throttle before this faces the internet.
- **Single process won't scale.** One model, one worker. Concurrent requests queue. Real load needs multiple replicas behind a queue or a proper inference server like vLLM.
- **The endpoint is `def`, not `async def`.** Inference is blocking, so a long generation stalls the event loop. Offloading to a threadpool or worker fixes it.
- **Memory is the real ceiling.** TinyLlama is friendly, but step up to a 7B model and your laptop quits. Plan your hardware before your model.

## Takeaway

Prompting is a feature. Engineering is the product.

The people clearing interviews in 2026 aren't the ones with the cleverest prompt — they're the ones who can put a model behind a validated, observable, restartable service and tell you exactly where it falls over. This project is small on purpose. The skill it proves is not.

*Built by adapting Aman Kharwal's walkthrough, ["Build a Production-Ready LLM API"](https://amanxai.com/2026/02/11/build-a-production-ready-llm-api/) — credit where it's due.*

### Sources
- [Aman Kharwal — Build a Production-Ready LLM API](https://amanxai.com/2026/02/11/build-a-production-ready-llm-api/)
- [Gartner — Hype Cycle for Agentic AI](https://www.gartner.com/en/articles/hype-cycle-for-agentic-ai)
- [Codersera — Open-Source LLM Landscape 2026](https://codersera.com/blog/open-source-llms-landscape-2026/)
