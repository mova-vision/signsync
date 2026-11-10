"""AI sentence builder — the stage orchestrator.

Pipeline:

    tags → ordered readings → one LLM call → TTS

The old MiniLM-reduce and Cloudflare-clef stages are gone: the LLM sees the
ordered gestures with their possible meanings (and spell-mode words pinned
in place) and composes the sentence directly — no candidate enumeration,
no reranker, no decision model. Fast path: a sequence with exactly one
possible reading skips the network entirely.

The returned events carry per-stage latencies straight into the frontend,
nothing hardcoded. An LLM error is surfaced as an error event, never
dropped silently. Stage-config from .env (no code touch):

    AI_ENABLED=0            → layer fully off (no events)
    AI_LLM_BASE_URL / AI_LLM_MODEL / AI_LLM_API_KEY → the OpenAI-compatible
      endpoint to call (see src/ai/llm_sentence.py)
"""
import json
import os
import pathlib
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from src.ai import llm_sentence
from src.ai.permutation import slot_parts

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

        # 1) ordered readings: each gesture with its possible meanings,
        #    spell-mode words pinned in place. Combination count is the
        #    product of the option lengths — displayed, never enumerated.
        slots, meaning_map = slot_parts(list(tags), list(tokens or []))
        option_count = 1
        for slot in slots:
            if slot.get("type") == "value":
                option_count *= max(1, len(slot.get("options") or []))
        perm_ms = (time.perf_counter() - t0) * 1000.0

        if not slots:
            return self._state(
                result="error",
                error="No candidate sentences for this tag set",
                perm_ms=perm_ms,
                option_count=0,
                llm_available=llm_sentence.is_configured(),
                tags=list(tags),
                tokens=list(tokens or []),
            )

        # 2) the one LLM call — composes the most sensible sentence.
        #    Even a single reading goes through the LLM: turn telegraphic
        #    readings ("hello wave") into natural speech ("Hello!").
        llm_available = llm_sentence.is_configured()
        decision = None
        decision_error = None
        decision_note = None
        if llm_available:
            try:
                decision = llm_sentence.compose(
                    slots,
                    history=list(self.history),
                )
            except Exception as exc:  # noqa: BLE001 — surfaced to the UI, not swallowed
                decision_error = str(exc)
        else:
            decision_error = (
                "AI_LLM_API_KEY / AI_LLM_MODEL missing in .env"
            )

        if not decision:
            return self._state(
                result="error",
                error=decision_error,
                perm_ms=round(perm_ms, 2),
                option_count=option_count,
                llm_available=llm_available,
                tags=list(tags),
                tokens=list(tokens or []),
            )

        sentence = decision["selected"]
        total_ms = (time.perf_counter() - t0) * 1000.0
        self.remember(sentence)

        event = self._state(
            result="ok",
            sentence=sentence,
            model=decision.get("model"),
            confidence=decision.get("confidence"),
            probabilities=decision.get("probabilities"),
            llm_ms=round(decision.get("latency_ms", 0.0), 2),
            decision_note=decision_note,
            total_ms=round(total_ms, 2),
            perm_ms=round(perm_ms, 2),
            option_count=option_count,
            llm_available=llm_available,
            tags=list(tags),
            tokens=list(tokens or []),
        )

        # 3) TTS bake → speech playback payload (same seam as one-shot)
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
            llm_available=llm_sentence.is_configured(),
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
                # pre-open the LLM TLS connection (latency fix)
                try:
                    from src.ai import llm_sentence
                    llm_sentence.warm_up()
                except Exception:
                    pass
    return _builder
