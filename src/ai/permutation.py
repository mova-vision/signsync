"""Ordered-tags → candidate sentences.

Builds every possible sentence from the collected tags:

- Spell-mode space tokens (already-flushed words like "HELLO: MARK") pin a
  fixed word at their raw sequence position — they are not permuted.
- Normal gesture tags expand through `resolve_meanings` and get permuted.

Data format is stage-agnostic: `List[str] sentences in → List[str] out`,
so MiniLM can be commented out and nothing downstream changes shape.
"""
import itertools
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from src.ai.meanings import resolve_meanings

# Hard ceiling for the cartesian product — keeps a runaway candidate set
# from freezing the pipeline before MiniLM ever gets to see it.
MAX_OPTIONS = 3000


def slot_parts(
    tags: List[str],
    tokens: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[str]]]:
    """Interleave spell tokens (fixed words) with permutable gesture slots.

    Returns (slots, meanings_by_tag):
      slots: ordered list of {"type": "fixed"|"value", "word"|"tag", …}
    """
    sorted_tokens = sorted(tokens or [], key=lambda t: t.get("pos") or 0)
    slots: List[Dict[str, Any]] = []
    meanings: Dict[str, List[str]] = {}
    ti = 0
    for idx, name in enumerate(tags or []):
        while ti < len(sorted_tokens) and (sorted_tokens[ti].get("pos") or 0) <= idx:
            tok = sorted_tokens[ti]
            letters = tok.get("letters") or [""]
            word = "".join(letters) if tok.get("mode") == "spell" else (letters[0] or "")
            if word:
                slots.append({"type": "fixed", "word": word, "space": tok.get("space")})
            ti += 1
        tag = name or "?"
        options = resolve_meanings(tag)
        meanings[tag] = options
        slots.append(
            {
                "type": "value",
                "tag": tag,
                "options": options,
            }
        )
    while ti < len(sorted_tokens):
        tok = sorted_tokens[ti]
        letters = tok.get("letters") or [""]
        word = "".join(letters) if tok.get("mode") == "spell" else (letters[0] or "")
        if word:
            slots.append({"type": "fixed", "word": word, "space": tok.get("space")})
        ti += 1
    return slots, meanings


@lru_cache(maxsize=1)
def _slot_options_cached(tag: str) -> Tuple[str, ...]:
    return tuple(resolve_meanings(tag))


def build_sentences(
    tags: List[str],
    tokens: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Cartesian product → candidate sentences (capped at MAX_OPTIONS)."""
    slots, meaning_map = slot_parts(tags, tokens)
    if not slots:
        return {"sentences": [], "options_before": 0, "meaning_map": meaning_map}

    pools: List[Tuple[str, Tuple[str, ...]]] = []
    for slot in slots:
        if slot["type"] == "fixed":
            pools.append((slot["word"],))
        else:
            pools.append(tuple(_slot_options_cached(slot["tag"])))

    total = 1
    for pool in pools:
        total *= len(pool)

    if total > MAX_OPTIONS:
        pool = next(p for p in pools if len(p) > 1)
        # Trim widest pools first, deterministically, until under cap.
        trimmed_pools = [list(p) for p in pools]
        guard = 0
        while total > MAX_OPTIONS and guard < 100:
            widest = max(trimmed_pools, key=len)
            if len(widest) <= 1:
                break
            widest.pop()
            total = 1
            for p in trimmed_pools:
                total *= max(1, len(p))
        pools = [tuple(p) for p in trimmed_pools]

    sentences = [" ".join(words) for words in itertools.product(*pools)]
    return {
        "sentences": sentences,
        "options_before": len(sentences),
        "meaning_map": meaning_map,
    }
