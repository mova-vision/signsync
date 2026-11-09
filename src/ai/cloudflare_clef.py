"""Decision-model adaptor: Cloudflare clef-flash.

Decides which candidate sentence the gesture sequence most likely meant
(text-only — no images anywhere in this protocol).

Call shape (exactly the documented Cloudflare Decisions protocol):

    POST https://api.cloudflare.com/client/v4/accounts/{account}/ai/run
         /@cf/cloudflare/clef-flash
    Authorization: Bearer {token}
    cf-aig-gateway-id: {gateway}
    {
      "model": "clef-flash",
      "state": { pool of context + candidate sentences },
      "questions": { "sentence": { choice-type question } }
    }

Answer: payload["answers"]["sentence"]["choice"] etc.

Credentials come from the environment (never hardcoded):
    CLOUDFLARE_ACCOUNT_ID   — set by the user in .env
    CLOUDFLARE_API_TOKEN    — set by the user in .env
    CLOUDFLARE_MODEL        — optional override (default clef-flash)
"""
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

RUN_URL = "https://api.cloudflare.com/client/v4/accounts/{account}/ai/run/@cf/cloudflare/clef-flash"
DEFAULT_MODEL = "clef-flash"
DEFAULT_GATEWAY = "default"
REQUEST_TIMEOUT = 30.0


def account_id() -> str:
    return (os.environ.get("CLOUDFLARE_ACCOUNT_ID") or "").strip()


def api_token() -> str:
    return (os.environ.get("CLOUDFLARE_API_TOKEN") or "").strip()


def gateway_id() -> str:
    return (os.environ.get("CLOUDFLARE_GATEWAY") or DEFAULT_GATEWAY).strip() or DEFAULT_GATEWAY


def model_id() -> str:
    return (os.environ.get("CLOUDFLARE_MODEL") or DEFAULT_MODEL).strip() or DEFAULT_MODEL


def is_configured() -> bool:
    return bool(account_id() and api_token())


SCENE_TEXT = "Mute user talks to one partner using only hand gestures. Candidates are raw gesture readings and may be broken grammar."

def build_state(tags: List[str], sentences: List[str], history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    """State pool mirrors the experiment's structure (participants/scene/history)."""
    return {
        "participants": [
            {"id": "U", "role": "mute_user"},
            {"id": "P", "role": "partner"},
        ],
        "scene": SCENE_TEXT,
        "conversation_history": history or [],
        "current_communication": {
            "speaker": "U",
            "gesture": " + ".join(tags),
            "possible_meanings": sentences,
        },
    }


def build_question(sentences: List[str]) -> Dict[str, Any]:
    criteria = {s: f'User may mean: "{s}"' for s in sentences}
    return {
        "type": "choice",
        "instructions": (
            "Pick the ONE candidate sentence the mute user most likely means. "
            "Judged by: (1) fit with the gesture sequence, (2) fit with the scene and "
            "conversation history, (3) which real-life scenario makes most sense. "
            "Broken gesture-grammar is fine — meaning beats grammar. "
            "Answer with exactly one choice from the criteria list."
        ),
        "criteria": criteria,
    }


def decide(
    tags: List[str],
    sentences: List[str],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """One decision call. Returns a latency-carrying result dict.

    Raises RuntimeError on configuration/network problems — the orchestrator
    turns that into a visible ai-state error, never a silent hang.
    """
    start = time.perf_counter()
    result: Dict[str, Any] = {
        "model": model_id(),
        "selected": None,
        "confidence": None,
        "probabilities": None,
        "latency_ms": 0.0,
    }
    if not is_configured():
        raise RuntimeError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN missing in .env")

    body = json.dumps(
        {
            "model": model_id(),
            "state": build_state(tags, sentences, history),
            "questions": {"sentence": build_question(sentences)},
        }
    ).encode("utf-8")
    url = RUN_URL.format(account=account_id())
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {api_token()}",
            "cf-aig-gateway-id": gateway_id(),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            detail = exc.read().decode("utf-8")
        except Exception:
            detail = ""
        raise RuntimeError(f"Cleff decision HTTP {exc.code}: {detail or exc.reason}") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"Cleff decision network error: {exc}") from exc

    result["latency_ms"] = (time.perf_counter() - start) * 1000.0
    # Cloudflare's API wraps the answer envelope: {result, success, errors, messages}
    payload = payload.get("result") or payload
    if isinstance(payload, dict) and payload.get("success") is False:
        raise RuntimeError(f"Cleff decision API error: {payload.get('errors')}")
    answer = (payload.get("answers") or {}).get("sentence") or {}
    result["selected"] = answer.get("choice")
    result["confidence"] = answer.get("confidence")
    result["probabilities"] = answer.get("probabilities")
    result["usage"] = payload.get("usage")
    if not result["selected"]:
        raise RuntimeError("Decision returned no choice")
    return result
