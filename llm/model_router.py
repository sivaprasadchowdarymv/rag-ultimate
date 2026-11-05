"""
Role-based routing across Groq (text) and OpenRouter (images).

    router.chat("master", messages, tools=...)   text answer  -> groq:openai/gpt-oss-120b, ...
    router.chat("vision", messages, images=[..]) image answer -> openrouter:google/gemma-4-31b-it:free, ...

Each role has an ordered chain of "provider:model" entries (llm/catalog.py).
Entries whose provider has no API key are skipped. The router moves to the
next entry only for recoverable errors (rate limit, model down, no credits,
timeout, empty answer); auth errors, moderation refusals and malformed
requests stop immediately.

Every result records the provider and model that answered, the attempts made,
the tokens and (OpenRouter only) the real cost in USD.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from config.settings import Settings, get_logger
from llm.catalog import PROVIDERS, ROLES, split
from llm.providers import FALLBACK_KINDS, ChatResult, KeyInfo, LLMError, make_provider

log = get_logger("router")

__all__ = ["ModelRouter", "RoutedResult", "ProviderStatus", "ROLES"]

# Hidden reasoning budget per role: small jobs and image reading need none.
_ROLE_REASONING = {"fast": "none", "verifier": "none", "vision": "none"}


@dataclass
class ProviderStatus:
    name: str
    label: str
    configured: bool
    connected: bool = False
    models: List[str] = field(default_factory=list)
    message: str = ""
    key: Optional[KeyInfo] = None
    used_for: str = ""


@dataclass
class RoutedResult:
    result: ChatResult
    attempts: List[Tuple[str, str, str]]  # (provider, model, outcome)
    latency_ms: float

    @property
    def cost(self) -> float:
        return self.result.cost


class ModelRouter:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.by_name = {name: make_provider(name, settings) for name in PROVIDERS}
        self.providers = [p for p in self.by_name.values() if p.configured()]

    # ------------------------------------------------------------------ models
    def chain(self, role: str):
        """[(provider, model)] for the role, keeping only providers that have a key."""
        out = []
        for entry in self.settings.models_for(role):
            name, model = split(entry)
            p = self.by_name.get(name)
            if p is not None and p.configured():
                out.append((p, model))
        return out

    def has_role(self, role: str) -> bool:
        return bool(self.chain(role))

    @property
    def enabled(self) -> bool:
        return bool(self.providers)

    # -------------------------------------------------------------------- chat
    def chat(self, role: str, messages: List[Dict[str, Any]], tools: Optional[List[Dict]] = None,
             tool_choice: str = "auto", images: Optional[Sequence[bytes]] = None,
             max_tokens: Optional[int] = None, reasoning: Optional[str] = None,
             json_mode: bool = False, temperature: Optional[float] = None) -> RoutedResult:
        t0 = time.perf_counter()
        attempts: List[Tuple[str, str, str]] = []
        chain = self.chain(role)
        if not chain:
            need = "OPENROUTER_API_KEY (images are read by Qwen / Gemma)" if role == "vision" else \
                "GROQ_API_KEY (free at console.groq.com/keys)"
            raise LLMError(f"No AI model is configured for this task. Add {need} to the app's secrets.",
                           "no_provider")
        effort = reasoning or _ROLE_REASONING.get(role) or self.settings.reasoning_effort
        last: Optional[LLMError] = None
        for provider, model in chain:
            try:
                result = provider.chat(model, messages, tools=tools, tool_choice=tool_choice,
                                       temperature=self.settings.llm_temperature if temperature is None else temperature,
                                       max_tokens=max_tokens or self.settings.max_output_tokens, images=images,
                                       reasoning=effort, json_mode=json_mode)
                attempts.append((provider.name, model, "ok"))
                return RoutedResult(result, attempts, (time.perf_counter() - t0) * 1000)
            except LLMError as exc:
                attempts.append((provider.name, model, exc.kind))
                last = exc
                if exc.kind not in FALLBACK_KINDS:
                    raise
                log.warning("%s/%s failed (%s); trying the next model for '%s'", provider.name, model, exc.kind, role)
        assert last is not None
        last.user_message += " No other model in the chain could answer."
        raise last

    # ------------------------------------------------------------------ status
    def status(self) -> List[ProviderStatus]:
        out = []
        for name, p in self.by_name.items():
            st = ProviderStatus(p.name, p.label, p.configured(),
                                used_for="text answers" if name == "groq" else "images (figures, scanned pages)")
            if st.configured:
                try:
                    st.key = p.key_info()
                    st.connected = True
                except LLMError as exc:
                    st.message = exc.user_message
                if st.connected:
                    try:
                        st.models = p.list_models()
                    except LLMError:
                        st.models = []
            out.append(st)
        return out
