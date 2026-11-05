"""
Central configuration for RAG Ultimate.

Every value can come from an environment variable, a `.env` file, or
Streamlit secrets (Streamlit Community Cloud). Secrets (GROQ_API_KEY,
OPENROUTER_API_KEY, HF_TOKEN, APP_PASSWORD) are never displayed or logged.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

try:  # optional: read a local .env file
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Lookup helpers: environment first, then Streamlit secrets
# ---------------------------------------------------------------------------
def _secret(name: str):
    try:
        import streamlit as st

        value = st.secrets.get(name)  # raises if no secrets file exists
    except Exception:
        return None
    return None if value is None else str(value)


def _lookup(name: str):
    value = os.getenv(name)
    if value is not None and value.strip():
        return value
    return _secret(name)


def _env_str(name: str, default: str) -> str:
    return (_lookup(name) or "").strip() or default


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(_lookup(name) or default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, lo: float, hi: float) -> float:
    try:
        return max(lo, min(hi, float(_lookup(name) or default)))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = _lookup(name)
    if value is None or not str(value).strip():
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


# ---------------------------------------------------------------------------
# Retrieval constants (unchanged from the original RAG)
# ---------------------------------------------------------------------------
SCORE_THRESHOLDS = {"text": 52, "table": 52, "row": 58, "equation": 48, "figure": 42}
AVG_CHARS_PER_TOK = 4
MAX_PREVIEW = 400


@dataclass(frozen=True)
class Settings:
    # --- Providers: Groq for text, OpenRouter (Qwen / Gemma) for images -----------
    groq_api_key: str = field(default="", repr=False)        # secret: never shown or logged
    groq_base_url: str = "https://api.groq.com/openai/v1"
    openrouter_api_key: str = field(default="", repr=False)  # secret: never shown or logged
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    app_url: str = "https://github.com/rag-ultimate"  # sent as HTTP-Referer (OpenRouter app attribution)
    app_title: str = "RAG Ultimate"                   # sent as X-Title
    model_profile: str = "free"                       # free | quality  (see llm/catalog.py)
    master_models: Tuple[str, ...] = ()               # empty = profile default
    fast_models: Tuple[str, ...] = ()
    verifier_models: Tuple[str, ...] = ()
    vision_models: Tuple[str, ...] = ()
    reasoning_effort: str = "low"                     # none | low | medium | high (hidden reasoning, never shown)
    data_collection: str = "allow"                    # allow | deny (deny = only providers that don't train on data)
    llm_temperature: float = 0.1
    max_output_tokens: int = 1500                     # includes hidden reasoning tokens
    llm_timeout: float = 90.0
    image_answers: bool = True                        # figure questions: attach the figure images for Qwen/Gemma
    max_answer_images: int = 2                        # images attached per figure question
    image_max_px: int = 1024                          # images are downscaled to this (fewer image tokens)

    # --- Persistent storage + access ---------------------------------------
    hf_token: str = field(default="", repr=False)  # Hugging Face WRITE token (secret)
    hf_dataset_repo: str = ""  # e.g. "your-name/rag-data" (created private automatically)
    sync_interval: float = 45.0  # seconds between background uploads
    app_password: str = field(default="", repr=False)  # optional password for the whole app

    # --- Pipeline -------------------------------------------------------------
    rerank: bool = True
    rerank_backend: str = "local"     # local (0 tokens) | openrouter | auto
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # local cross-encoder (also used for claim support)
    rerank_model_remote: str = "qwen/qwen3-reranker-8b"
    verify_with_llm: bool = False     # extra LLM cross-check (deterministic checks always run, 0 tokens)
    max_regenerations: int = 1
    memory_turns: int = 2  # previous Q&A pairs given to the master agent
    health_weights: Tuple[float, float, float, float, float] = (0.30, 0.20, 0.20, 0.20, 0.10)
    support_threshold: float = 0.0    # local cross-encoder logit below which a citation is weakly supported
    support_threshold_remote: float = 0.25  # same, for OpenRouter reranker relevance (0..1)

    # --- Agent --------------------------------------------------------------
    agent_mode: str = "agent"  # "agent" (ReAct with tools) or "quick" (one call)
    agent_max_steps: int = 3  # tool rounds before the agent must answer
    observation_chars: int = 1500  # max characters per tool observation

    # --- Embeddings ---------------------------------------------------------
    embed_backend: str = "local"      # local (0 tokens) | openrouter | auto
    embed_model: str = "nomic-ai/nomic-embed-text-v1.5-Q"   # local fastembed model
    embed_model_remote: str = "qwen/qwen3-embedding-8b"
    embed_dims: int = 1024            # Matryoshka truncation of the remote embedding

    # --- Vision at indexing time ---------------------------------------------
    vision_index: bool = False        # describe ALL figures while indexing (off: figures are seen on demand)
    ocr_scanned: bool = True          # OCR image-only pages with the vision model
    max_vision_figures: int = 40      # per document (largest figures first)
    max_ocr_pages: int = 40           # per document
    vision_workers: int = 4           # parallel vision requests while indexing

    # --- Retrieval / chunking ---------------------------------------------
    top_k: int = 3
    max_chunk_tokens: int = 400
    overlap_tokens: int = 60
    max_context_chunks: int = 6
    max_context_chars: int = 5000     # ~1,250 tokens of evidence per answer (Groq free tier: 8K tokens/min)
    max_chars_per_chunk: int = 900

    # --- Uploads / figures ------------------------------------------------
    max_upload_mb: int = 50
    max_pages: int = 500
    min_figure_px: int = 48

    # --- Storage / misc ---------------------------------------------------
    data_dir: Path = PROJECT_ROOT / "data"
    enable_feedback: bool = False
    log_level: str = "INFO"

    # --- Production guards (rate limit, caches, budget) -----------------------
    rate_limit_per_min: int = 8          # per browser session
    global_rate_limit_per_min: int = 25  # whole server (Groq free tier: ~30 requests/min)
    answer_cache_size: int = 256
    answer_cache_ttl: int = 3600         # seconds; 0 disables the answer cache
    daily_budget_usd: float = 0.0        # 0 = no limit; stops AI answers once today's spend reaches it

    # --- Cross-modal concept linking ------------------------------------------
    concept_linking: bool = True
    concept_expand_max: int = 4          # extra linked items per question (before rerank/fusion)
    concept_min_conf: float = 0.6        # links below this are never expanded
    concept_llm_resolve: bool = False    # small LLM only for ambiguous abbreviations (cached per doc)

    @property
    def parse_signature(self) -> Tuple:
        return (self.max_chunk_tokens, self.overlap_tokens, self.min_figure_px)

    @property
    def has_openrouter(self) -> bool:
        return bool(self.openrouter_api_key)

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    def provider_configured(self, provider: str) -> bool:
        return {"groq": self.has_groq, "openrouter": self.has_openrouter}.get(provider, False)

    def models_for(self, role: str) -> Tuple[str, ...]:
        """Ordered "provider:model" chain for a role: explicit override > profile default."""
        from llm.catalog import chain_for

        override = getattr(self, f"{role}_models", ())
        return tuple(override) or chain_for(self.model_profile, role)

    @property
    def _auto_remote(self) -> bool:
        # "auto" uses OpenRouter's (paid, very cheap) Qwen3 retrieval models, except on the $0 free profile.
        return self.has_openrouter and self.model_profile != "free"

    @property
    def remote_embeddings(self) -> bool:
        return self.embed_backend == "openrouter" or (self.embed_backend == "auto" and self._auto_remote)

    @property
    def remote_rerank(self) -> bool:
        return self.rerank_backend == "openrouter" or (self.rerank_backend == "auto" and self._auto_remote)


def _writable_dir(preferred: Path) -> Path:
    """Use `preferred` if writable, else a temp folder (read-only hosts)."""
    try:
        preferred.mkdir(parents=True, exist_ok=True)
        probe = preferred / ".write_test"
        probe.write_text("ok")
        probe.unlink()
        return preferred
    except OSError:
        import tempfile

        fallback = Path(tempfile.gettempdir()) / "rag-ultimate-data"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _weights() -> Tuple[float, float, float, float, float]:
    raw = _env_str("HEALTH_WEIGHTS", "0.30,0.20,0.20,0.20,0.10").split(",")
    try:
        vals = [max(0.0, float(x)) for x in raw][:5]
        if len(vals) == 5 and sum(vals) > 0:
            total = sum(vals)
            return tuple(v / total for v in vals)  # type: ignore[return-value]
    except ValueError:
        pass
    return (0.30, 0.20, 0.20, 0.20, 0.10)


def _models(name: str) -> Tuple[str, ...]:
    return tuple(m.strip() for m in _env_str(name, "").split(",") if m.strip())


def _choice(name: str, default: str, allowed: Tuple[str, ...]) -> str:
    value = _env_str(name, default).lower()
    return value if value in allowed else default


def load_settings() -> Settings:
    return Settings(
        groq_api_key=(_lookup("GROQ_API_KEY") or "").strip(),
        groq_base_url=_env_str("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/"),
        openrouter_api_key=(_lookup("OPENROUTER_API_KEY") or "").strip(),
        openrouter_base_url=_env_str("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/"),
        app_url=_env_str("APP_URL", "https://github.com/rag-ultimate"),
        app_title=_env_str("APP_TITLE", "RAG Ultimate"),
        model_profile=_choice("MODEL_PROFILE", "free", ("free", "quality")),
        master_models=_models("MASTER_MODELS"),
        fast_models=_models("FAST_MODELS"),
        verifier_models=_models("VERIFIER_MODELS"),
        vision_models=_models("VISION_MODELS"),
        reasoning_effort=_choice("REASONING_EFFORT", "low", ("none", "low", "medium", "high")),
        data_collection=_choice("DATA_COLLECTION", "allow", ("allow", "deny")),
        llm_temperature=_env_float("LLM_TEMPERATURE", 0.1, 0.0, 2.0),
        max_output_tokens=_env_int("MAX_OUTPUT_TOKENS", 1500, 200, 32000),
        llm_timeout=_env_float("LLM_TIMEOUT", 90.0, 5.0, 600.0),
        image_answers=_env_bool("IMAGE_ANSWERS", True),
        max_answer_images=_env_int("MAX_ANSWER_IMAGES", 2, 1, 6),
        image_max_px=_env_int("IMAGE_MAX_PX", 1024, 256, 4096),
        hf_token=(_lookup("HF_TOKEN") or "").strip(),
        hf_dataset_repo=_env_str("HF_DATASET_REPO", ""),
        sync_interval=_env_float("SYNC_INTERVAL", 45.0, 15.0, 3600.0),
        app_password=(_lookup("APP_PASSWORD") or "").strip(),
        rerank=_env_bool("RERANK", True),
        rerank_backend=_choice("RERANK_BACKEND", "local", ("auto", "openrouter", "local")),
        rerank_model=_env_str("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2"),
        rerank_model_remote=_env_str("RERANK_MODEL_REMOTE", "qwen/qwen3-reranker-8b"),
        verify_with_llm=_env_bool("VERIFY_WITH_LLM", False),
        max_regenerations=_env_int("MAX_REGENERATIONS", 1, 0, 2),
        memory_turns=_env_int("MEMORY_TURNS", 2, 0, 10),
        health_weights=_weights(),
        support_threshold=_env_float("SUPPORT_THRESHOLD", 0.0, -20.0, 20.0),
        support_threshold_remote=_env_float("SUPPORT_THRESHOLD_REMOTE", 0.25, 0.0, 1.0),
        agent_mode=_choice("AGENT_MODE", "agent", ("agent", "quick")),
        agent_max_steps=_env_int("AGENT_MAX_STEPS", 3, 1, 6),
        observation_chars=_env_int("OBSERVATION_CHARS", 1500, 400, 20000),
        embed_backend=_choice("EMBED_BACKEND", "local", ("auto", "openrouter", "local")),
        embed_model=_env_str("EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5-Q"),
        embed_model_remote=_env_str("EMBED_MODEL_REMOTE", "qwen/qwen3-embedding-8b"),
        embed_dims=_env_int("EMBED_DIMS", 1024, 32, 4096),
        vision_index=_env_bool("VISION_INDEX", False),
        ocr_scanned=_env_bool("OCR_SCANNED", True),
        max_vision_figures=_env_int("MAX_VISION_FIGURES", 40, 0, 500),
        max_ocr_pages=_env_int("MAX_OCR_PAGES", 40, 0, 1000),
        vision_workers=_env_int("VISION_WORKERS", 4, 1, 16),
        top_k=_env_int("TOP_K", 3, 1, 10),
        max_chunk_tokens=_env_int("CHUNK_SIZE", 400, 100, 800),
        overlap_tokens=_env_int("CHUNK_OVERLAP", 60, 0, 150),
        max_context_chunks=_env_int("MAX_CONTEXT_CHUNKS", 6, 1, 30),
        max_context_chars=_env_int("MAX_CONTEXT_CHARS", 5000, 1000, 200000),
        max_chars_per_chunk=_env_int("MAX_CHARS_PER_CHUNK", 900, 200, 10000),
        max_upload_mb=_env_int("MAX_UPLOAD_MB", 50, 1, 500),
        max_pages=_env_int("MAX_PAGES", 500, 1, 5000),
        min_figure_px=_env_int("MIN_FIGURE_PX", 48, 0, 2000),
        data_dir=_writable_dir(Path(_env_str("DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()),
        enable_feedback=_env_bool("ENABLE_FEEDBACK", False),
        log_level=_env_str("LOG_LEVEL", "INFO").upper(),
        rate_limit_per_min=_env_int("RATE_LIMIT_PER_MIN", 8, 0, 1000),
        global_rate_limit_per_min=_env_int("GLOBAL_RATE_LIMIT_PER_MIN", 25, 0, 10000),
        answer_cache_size=_env_int("ANSWER_CACHE_SIZE", 256, 0, 10000),
        answer_cache_ttl=_env_int("ANSWER_CACHE_TTL", 3600, 0, 86400 * 7),
        daily_budget_usd=_env_float("DAILY_BUDGET_USD", 0.0, 0.0, 100000.0),
        concept_linking=_env_bool("CONCEPT_LINKING", True),
        concept_expand_max=_env_int("CONCEPT_EXPAND_MAX", 4, 0, 10),
        concept_min_conf=_env_float("CONCEPT_MIN_CONF", 0.6, 0.0, 1.0),
        concept_llm_resolve=_env_bool("CONCEPT_LLM_RESOLVE", False),
    )


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER_NAME = "rag_ultimate"


def setup_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(name)s  %(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(getattr(logging, level, logging.INFO))
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")
