# RAG∞ Ultimate: Groq for text, Qwen / Gemma only for images

Ask questions about datasheets, papers and technical PDFs and get **cited, verified** answers, at the lowest
token cost:

* **Text questions → Groq** (free tier, very fast): `openai/gpt-oss-120b`, falling back to `llama-3.3-70b-versatile`.
* **Only when images are involved → Qwen 3.8 / Gemma 4 on OpenRouter**:
  * questions about a figure, graph, chart, diagram or schematic: the figure images are attached and the model reads them;
  * scanned (image-only) pages: read once while indexing.
* **Search runs locally** (embeddings + reranker on your CPU): 0 tokens.

```text
PDF ─► parse text · tables · equations · figures (local) ─► scanned pages ─► Gemma/Qwen OCR (once)
     ─► local embeddings + BM25 index
QUESTION ─► rule-based router (1 ms, 0 tokens) ─► specialists ─► local reranker ─► evidence fusion (~1,250 tokens)
     ├─ text question ────────────► Groq gpt-oss-120b (tools only for calculations / comparisons)
     └─ figure question + figure ─► Gemma 4 / Qwen 3.8 with the figure image(s)
     ─► verifier (deterministic, 0 tokens) ─► answer with [REF-n] citations, verdict, model, tokens
```

**Installation: see [STEP_BY_STEP.md](STEP_BY_STEP.md)** (Windows, macOS and Linux, from zero to deployed).

## Models

| Role | Used for | Chain (tried in order) |
|---|---|---|
| master | text answers | `groq:openai/gpt-oss-120b` → `groq:llama-3.3-70b-versatile` → `openrouter:google/gemma-4-31b-it:free` |
| fast | tiny helper jobs | `groq:openai/gpt-oss-20b` → `groq:llama-3.1-8b-instant` |
| verifier | optional LLM check (off by default) | `groq:llama-3.3-70b-versatile` → `groq:openai/gpt-oss-20b` |
| vision | figure questions, scanned pages | **free:** `gemma-4-31b-it:free` → `gemma-4-26b-a4b-it:free` → `qwen3.8-27b`; **quality:** `qwen3.8-flash` → `gemma-4-31b-it` → `gemma-4-31b-it:free` |

`MODEL_PROFILE=free` (default) costs $0. `MODEL_PROFILE=quality` puts paid Qwen 3.8 first for images (cents).
Any chain can be overridden in `.env`, e.g. `VISION_MODELS=openrouter:qwen/qwen3.8-flash`.

## How tokens are kept low

| Measure | Effect |
|---|---|
| Rule-based router and specialists | No LLM call to plan or retrieve |
| Local embeddings and reranker | Search costs 0 tokens |
| Evidence budget `MAX_CONTEXT_CHARS=5000` | About 1,250 evidence tokens per answer (fits Groq's free per-minute limits) |
| One call for simple questions | Tools (search / read page / calculator) only for calculations, comparisons, equations |
| Hidden reasoning `low` and never returned | Short gpt-oss reasoning, no reasoning text in the response |
| Images only on figure questions, downscaled to 1024 px, max 2 | Vision tokens only where they are needed |
| Figures are not described in advance (`VISION_INDEX=false`) | No vision calls for figures nobody asks about |
| Deterministic verifier, LLM check off | Verification costs 0 tokens |
| Verified-answer cache, memory of 2 turns | Repeated questions are free; short history |

Every answer shows the provider, model and token count (and cost, for paid OpenRouter models).

## Quick start

```bash
python -m venv venv && source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # Windows: copy .env.example .env
#   GROQ_API_KEY=gsk_...          (required, free)
#   OPENROUTER_API_KEY=sk-or-...  (for images, free)
python -m scripts.check_setup                          # checks both keys, every model and role ($0)
streamlit run app.py
```

## Using it

1. **Chat:** upload PDFs, click a suggested question or type your own. Each answer shows a verdict
   (✓ Verified / ≈ Partly verified / ⚠ Check sources), citation chips, `Groq: gpt-oss-120b` or
   `OpenRouter: gemma-4-31b-it:free`, and the token count. Values read from an image are marked
   ≈ Partly verified, because they cannot be checked against text.
2. **Library:** your documents; toggle **Use in chat** or **Ask →**.
3. **Validate:** auto-generated questions with known answers; retrieval and answer accuracy on your document.
4. **Insights:** quality metrics, agent activity, feedback, system telemetry.
5. **Settings:** profile, model chains, Groq model picker, reasoning effort, image routing, OCR, reranking.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests -q        # 96 offline tests with a fake Groq + OpenRouter server; no keys needed
```

## Limits (honest)

* Groq's free tier has per-minute and per-day limits per model. When gpt-oss-120b is exhausted, the chain moves to
  Llama 3.3 70B, then to free Gemma on OpenRouter.
* OpenRouter's free endpoints allow about 20 requests/min and 50/day (1,000/day after buying $10 of credits).
* Figure questions are detected by keywords (figure, graph, plot, curve, chart, diagram, waveform, schematic,
  image, picture). Other phrasings are answered from captions by Groq.
* Document excerpts go to Groq; figure images and scanned pages go to OpenRouter and its upstream provider.

See [STEP_BY_STEP.md](STEP_BY_STEP.md) and [ARCHITECTURE.md](ARCHITECTURE.md).
