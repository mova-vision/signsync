"""MiniLM candidate filter — the optional reducer stage.

Ported from experiments/gesture_permutations_pick/script.py (the cross
encoder lives there too). Same data format as every other stage:

    List[str] candidates in → (kept List[str], top1 str, elapsed_ms) out

Behavior notes:
- Lazy singleton; first call pulls/downloads the ONNX model once and then
  every later call is ~tens of ms.
- `is_configured()` tells the orchestrator whether the stage can run at
  all (deps present); a missing config means the stage reports
  `available=False` and the FULL candidate list flows through untouched.
"""
import os
import time
from typing import Any, Dict, List, Optional, Tuple

MODEL_ID = "Xenova/ms-marco-MiniLM-L-6-v2"

_filter: Optional[Tuple[Any, Any, Any]] = None  # (tokenizer, ort session, numpy)
_load_failed = False


def top_k_from_env(default: int = 8) -> int:
    try:
        return max(1, int(os.environ.get("AI_MINILM_TOP_K", str(default)).strip() or default))
    except ValueError:
        return default


def _find_onnx(snap: str) -> Optional[str]:
    found = []
    for root, _, files in os.walk(snap):
        for name in files:
            if name in ("model_quantized.onnx", "model.onnx"):
                found.append(os.path.join(root, name))
    found.sort(key=lambda p: (os.path.basename(p) != "model_quantized.onnx", len(p)))
    return found[0] if found else None


def _load():
    """Load (tokenizer, session, numpy) once; returns None when unavailable."""
    global _filter, _load_failed
    if _filter is not None:
        return _filter
    if _load_failed:
        return None
    try:
        import numpy as np
        import onnxruntime as ort
        from huggingface_hub import snapshot_download
        from transformers import AutoTokenizer
    except ImportError:
        _load_failed = True
        return None

    snap = None
    for patterns in (["*.json", "*model_quantized.onnx"], ["*.json", "*model.onnx"], None):
        try:
            snap = snapshot_download(repo_id=MODEL_ID, allow_patterns=patterns)
            found = _find_onnx(snap)
            if found:
                break
        except Exception:
            snap = None
    try:
        tok = AutoTokenizer.from_pretrained(snap)
        fpath = _find_onnx(snap)
        if not fpath:
            _load_failed = True
            return None
        sess = ort.InferenceSession(fpath, providers=["CPUExecutionProvider"])
    except Exception:
        _load_failed = True
        return None
    _filter = (tok, sess, np)
    return _filter


def is_available() -> bool:
    """True when the ONNX stack is importable and loadable."""
    return _load() is not None


def build_query(tags: List[str]) -> str:
    return "gestures: " + " + ".join(tags)


def rank(
    tags: List[str],
    sentences: List[str],
    top_k: int,
) -> Tuple[List[str], str, float]:
    """Score (sequence-context, candidate) pairs, keep the top top_k.

    Returns (kept in original order, top1 candidate, elapsed_ms).
    """
    bundle = _load()
    if bundle is None:
        return list(sentences), (sentences[0] if sentences else ""), 0.0
    tok, sess, np = bundle
    query = build_query(tags)
    enc = tok(
        [query] * len(sentences),
        sentences,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="np",
    )
    wanted = {i.name for i in sess.get_inputs()}
    ort_inputs = {
        k: np.asarray(enc[k]).astype(np.int64)
        for k in ("input_ids", "attention_mask", "token_type_ids")
        if k in wanted and k in enc
    }
    start = time.perf_counter()
    logits = sess.run(None, ort_inputs)[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    scores = logits.reshape(-1)
    order = sorted(range(len(sentences)), key=lambda i: float(scores[i]), reverse=True)
    top1 = sentences[order[0]]
    keep_idx = set(order[: max(1, top_k)])
    kept = [s for i, s in enumerate(sentences) if i in keep_idx]
    return kept, top1, elapsed_ms


def maybe_reduce(
    tags: List[str],
    sentences: List[str],
    trigger: int,
) -> Tuple[List[str], str, float]:
    """Run only when the candidate set is big enough to need it.

    Small sets pass through unchanged (0 ms) — that IS the latency
    measurement for the off-state, nothing is faked.
    """
    if len(sentences) <= max(1, trigger):
        return list(sentences), (sentences[0] if sentences else ""), 0.0
    kept, top1, elapsed_ms = rank(tags, sentences, top_k_from_env())
    return list(kept) if kept else list(sentences), top1, elapsed_ms
