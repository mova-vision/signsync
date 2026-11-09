"""MOVA AI sentence layer.

Stages, each swappable without code changes:

    tags collected → meanings → permutations → [MiniLM filter] → decision model

- The MiniLM stage is optional (env-switchable, lazy-loaded, ONNX local).
  Turned off = same data in/out, only the option count changes.
- The decision stage is a Cloudflare clef-flash call (see cloudflare_clef.py).
- A failed/absent stage degrades gracefully: the pipeline passes the full
  candidate set through instead of erroring out.
"""
from src.ai.builder import AiSentenceBuilder, get_ai_builder

__all__ = ["AiSentenceBuilder", "get_ai_builder"]
