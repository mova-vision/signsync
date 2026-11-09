"""AI sentence builder — the stage orchestrator.

Pipeline (every stage swappable / removable without code changes):

    tags → meanings → permutations → [MiniLM reduce] → decision (clef) → TTS

The returned events carry per-stage latencies straight into the frontend,
nothing hardcoded: miniml_ms is 0 when the stage passes through and the
frontend shows "off" only when the stage itself reports it. A decision
error is surfaced as an error event, never dropped silently.

How to switch stages from .env (no code touch):
    AI_ENABLED=0           → layer fully off (no events)
    AI_MINILM_ON=0         → MiniLM stage reports available=False,
                             full candidate set flows to the decision stage
    AI_MINILM_TRIGGER=50   → candidates above this count get reduced
    AI_DECISION=clef       → decision adaptor (only clef for now)
"""
import itertools
import json
import os
import pathlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.ai import cloudflare_clef
from src.ai import minilm_filter
from src.ai.permutation import build_sentences, slot_parts

_BASE = pathlib.Path(__file__).resolve().parent.parent.parent

# ---------- env helpers ----------

def _env(scene: str, key: str, default: str) -> str:
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        raw = ""
        env_path = _BASE / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{key}="):
                    raw = line.split("=", 1)[1].split("#", 1)[0].strip()
                    break
    default_b = str(default)
    return raw if raw else default_b


def ai_enabled() -> bool:
    return _env("ai", "AI_ENABLED", "1").lower() not in ("0", "false", "no", "off")


def minilm_enabled() -> bool:
    return _env("minilm", "AI_MINILM_ON", "1").lower() not in ("0", "false", "no", "off")


def minilm_trigger() -> int:
    return max(2, int(_env("minilm trigger", "AI_MINILM_TRIGGER", "50") or 50))


def decision_name() -> str:
    return _env("decision", "AI_DECISION", "clef").lower()


# ---------- builder ----------

class AiSentenceBuilder:
    """Runs the stage chain in a worker thread and emits finished events."""

    def __init__(
        self,
        speech_adaptor_getter: Callable[[], Any],
        history_maxlen: int = 6,
    ):
        self.speech_adaptor_getter = speech_adaptor_getter
        self.history: List[Dict[str, str]] = []
        self.history_maxlen = history_maxlen
        self.audio_path = _BASE / "data" / "audio" / "ai_last.wav"
        self.audio_path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._lock = threading.Lock()

    # ---- conversation memory (persists across runs within a session) ----
    def remember(self, sentence: str) -> None:
        with self._lock:
            self.history = (self.history + [{"role": "U", "text": sentence}])[-self.history_maxlen:]

    def reset(self) -> None:
        with self._lock:
            self.history = []

    def _state(self, **fields: Any) -> Dict[str, Any]:
        base = {
            "seq": self._seq,
            "ts": round(time.time(), 3),
        }
        base.update(fields)
        return base

    # ---- the stage chain itself (synchronous, runs inside the thread) ----
    def _run_stages(
        self,
        tags: List[str],
        tokens: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        self._seq += 1
        t0 = time.perf_counter()

        # 1) tags → permutations (meanings resolved inside)
        perm = build_sentences(list(tags), list(tokens or []))
        sentences: List[str] = perm["sentences"]
        perm_ms = (time.perf_counter() - t0) * 1000.0

        if not sentences:
            return self._state(
                result="error",
                error="No candidate sentences for this tag set",
                perm_ms=perm_ms,
                miniml_enabled=minilm_enabled(),
                miniml_ms=0.0,
                option_count=0,
                option_count_after=0,
                decision_available=cloudflare_clef.is_configured(),
                tags=list(tags),
                tokens=list(tokens or []),
            )

        tags_clean = [
            slot["tag"] for slot in slot_parts(list(tags), list(tokens or []))[0]
            if slot["type"] == "value"
        ]

        # 2) optional MiniLM reduce — candidates above the trigger count get
        # scored and cut to AI_MINILM_TOP_K; smaller sets pass through with
        # miniml_ms = 0 (nothing is faked).
        miniml_up_to = len(sentences) > minilm_trigger()
        if minilm_enabled() and miniml_up_to:
            t1 = time.perf_counter()
            sentences, _top1, _ = minilm_filter.maybe_reduce(
                tags_clean, sentences, minilm_trigger()
            )
            miniml_ms = (time.perf_counter() - t1) * 1000.0
        else:
            miniml_ms = 0.0
        option_count_after = len(sentences)

        # 3) decision stage (cloudflare clef-flash).
        # A single candidate needs no model — clef rejects 1-criteria choices
        # outright, so the pipeline picks it deterministically instead.
        decision_available = cloudflare_clef.is_configured()
        decision = None
        decision_error = None
        decision_note = None
        if len(sentences) == 1:
            decision = {
                "selected": sentences[0],
                "confidence": None,
                "probabilities": None,
                "latency_ms": 0.0,
            }
            decision_note = "single candidate (add meanings in data/meanings.json for more permutations)"
        elif decision_available:
            try:
                decision = cloudflare_clef.decide(
                    tags_clean,
                    sentences,
                    history=list(self.history),
                )
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
                decision_error = str(exc)
        else:
            decision_error = (
                "CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN missing in .env"
            )

        if not decision:
            return self._state(
                result="error",
                error=decision_error,
                perm_ms=round(perm_ms, 2),
                miniml_enabled=minilm_enabled(),
                miniml_ms=round(miniml_ms, 2),
                option_count=perm["options_before"],
                option_count_after=option_count_after,
                decision_available=decision_available,
                tags=list(tags),
                tokens=list(tokens or []),
            )

        sentence = decision["selected"]
        total_ms = (time.perf_counter() - t0) * 1000.0
        self.remember(sentence)

        event = self._state(
            result="ok",
            sentence=sentence,
            confidence=decision.get("confidence"),
            probabilities=decision.get("probabilities"),
            decision_ms=round(decision.get("latency_ms", 0.0), 2),
            decision_note=decision_note,
            total_ms=round(total_ms, 2),
            perm_ms=round(perm_ms, 2),
            miniml_enabled=minilm_enabled(),
            miniml_ms=round(miniml_ms, 2),
            option_count=perm["options_before"],
            option_count_after=option_count_after,
            decision_available=decision_available,
            tags=list(tags),
            tokens=list(tokens or []),
        )

        # 4) TTS bake → speech playback payload (same seam as one-shot)
        try:
            adaptor = self.speech_adaptor_getter()
            self.audio_path.unlink(missing_ok=True)
            adaptor.synthesize_and_save(sentence, str(self.audio_path))
            event["audio_url"] = f"/audio/{self.audio_path.name}"
        except Exception as exc:  # noqa: BLE001
            event["speech_error"] = str(exc)

        return event

    # ---------- public ----------
    def start(
        self,
        tags: List[str],
        tokens: List[Dict[str, Any]],
        emitter: Callable[[Dict[str, Any]], None],
    ) -> Dict[str, Any]:
        """Fire the chain on a worker thread; `emitter` gets the result once
        every stage (and the TTS bake) resolves. The WS loop calls emitter
        from the thread — it must be thread-safe (it is: the loop only
        copies onto a deque)."""
        def _job():
            event = self._run_stages(list(tags), list(tokens or []))
            try:
                emitter(event)
            except Exception:
                pass

        thread = threading.Thread(target=_job, name="mova-ai", daemon=True)
        thread.start()
        return self._state(
            result="pending",
            stage="roundtrip",
            tags=list(tags),
            tokens=list(tokens or []),
            miniml_available=minilm_enabled(),
            decision_available=cloudflare_clef.is_configured(),
        )


_builder: Optional[AiSentenceBuilder] = None
_builder_lock = threading.Lock()


def get_ai_builder() -> AiSentenceBuilder:
    global _builder
    if _builder is None:
        with _builder_lock:
            if _builder is None:
                def _speech():
                    from audio_adaptor import get_speech_adaptor
                    return get_speech_adaptor()

                _builder = AiSentenceBuilder(speech_adaptor_getter=_speech)
    return _builder
