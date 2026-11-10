"""MOVA AI sentence layer.

Stages:

    tags collected → ordered readings → one OpenAI-compatible LLM call

- Each gesture carries its possible meanings; spell-mode words are pinned
  in place. The totality is shown as a "readings" count, never enumerated.
- The LLM call composes the sentence directly (see llm_sentence.py);
  credentials via AI_LLM_BASE_URL / AI_LLM_MODEL / AI_LLM_API_KEY in .env.
- A single-reading sequence skips the network entirely (fast path).
"""
from src.ai.builder import AiSentenceBuilder, get_ai_builder

__all__ = ["AiSentenceBuilder", "get_ai_builder"]
