"""
Model catalog: which model plays which role.

The rule of this app: **text goes to Groq (free, fast), images go to Qwen / Gemma (OpenRouter).**

    master     writes the cited answer for TEXT questions          -> Groq gpt-oss-120b
    fast       tiny jobs (abbreviation resolution)                 -> Groq gpt-oss-20b
    verifier   optional LLM cross-check (off by default: tokens)    -> Groq Llama 3.3 70B (other family)
    vision     anything that needs to SEE an image:                -> OpenRouter Gemma 4 / Qwen 3.8
               * questions about figures, graphs, diagrams (the figure images are attached)
               * OCR of scanned pages while indexing
               * optional figure descriptions while indexing (VISION_INDEX, off by default)

Every chain entry is "provider:model". Entries whose provider has no key are
skipped, and the router walks the chain in order on recoverable errors
(rate limit, model down, no credits). The last text fallback is a free Gemma
on OpenRouter, so answers keep coming when Groq's daily free quota runs out.

Profiles (MODEL_PROFILE):
    free     Groq free tier for text + Gemma 4 ":free" for images -> $0
    quality  Groq for text + Qwen 3.8 (paid, cents) first for images -> better image reading

Override any chain with MASTER_MODELS / FAST_MODELS / VERIFIER_MODELS / VISION_MODELS,
e.g. VISION_MODELS=openrouter:qwen/qwen3.8-flash,openrouter:google/gemma-4-31b-it:free
"""
from __future__ import annotations

from typing import Dict, Tuple

ROLES = ("master", "fast", "verifier", "vision")
PROVIDERS = ("groq", "openrouter")

# Groq production models (console.groq.com/docs/models, November 2026)
GROQ_120B = "groq:openai/gpt-oss-120b"
GROQ_20B = "groq:openai/gpt-oss-20b"
GROQ_LLAMA70 = "groq:llama-3.3-70b-versatile"
GROQ_LLAMA8 = "groq:llama-3.1-8b-instant"
# OpenRouter multimodal models (openrouter.ai, November 2026)
GEMMA_31B_FREE = "openrouter:google/gemma-4-31b-it:free"
GEMMA_26B_FREE = "openrouter:google/gemma-4-26b-a4b-it:free"
GEMMA_31B = "openrouter:google/gemma-4-31b-it"
QWEN_FLASH = "openrouter:qwen/qwen3.8-flash"
QWEN_27B = "openrouter:qwen/qwen3.8-27b"

_TEXT = {
    "master": (GROQ_120B, GROQ_LLAMA70, GEMMA_31B_FREE),
    "fast": (GROQ_20B, GROQ_LLAMA8),
    "verifier": (GROQ_LLAMA70, GROQ_20B),
}
PROFILES: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "free": {**_TEXT, "vision": (GEMMA_31B_FREE, GEMMA_26B_FREE, QWEN_27B)},
    "quality": {**_TEXT, "vision": (QWEN_FLASH, GEMMA_31B, GEMMA_31B_FREE)},
}
PROFILE_LABELS = {
    "free": "🆓 Free - Groq for text, Gemma 4 free for images ($0)",
    "quality": "🖼️ Quality images - Groq for text, Qwen 3.8 for images (cents)",
}
DEFAULT_PROFILE = "free"

# Optional OpenRouter retrieval models (EMBED_BACKEND / RERANK_BACKEND = openrouter). Default is local = 0 tokens.
EMBED_MODEL = "qwen/qwen3-embedding-8b"
RERANK_MODEL = "qwen/qwen3-reranker-8b"
EMBED_QUERY_INSTRUCTION = ("Instruct: Given a question about a technical document, retrieve the passages, "
                           "table rows, equations or figure descriptions that answer it\nQuery: ")


def chain_for(profile: str, role: str) -> Tuple[str, ...]:
    return PROFILES.get(profile, PROFILES[DEFAULT_PROFILE]).get(role, ())


def split(entry: str) -> Tuple[str, str]:
    """'groq:openai/gpt-oss-120b' -> ('groq', 'openai/gpt-oss-120b'). Unprefixed ids default to OpenRouter."""
    head, sep, rest = entry.partition(":")
    if sep and head in PROVIDERS:
        return head, rest
    return "openrouter", entry


def is_free(entry: str) -> bool:
    provider, model = split(entry)
    return provider == "groq" or model.endswith(":free")


def short_name(entry: str) -> str:
    """'groq:openai/gpt-oss-120b' -> 'gpt-oss-120b' (compact UI labels)."""
    return split(entry)[1].split("/", 1)[-1]
