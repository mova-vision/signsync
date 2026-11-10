"""Sentence adaptor: one OpenAI-compatible chat call.

Replaces the old MiniLM + Cloudflare clef chain. The orchestrator hands us
the ordered gesture readings (each with its possible meanings, plus any
spell-mode words fixed in place) and we make ONE chat completion call that
composes the most sensible natural sentence.

All credentials come from the environment (.env), never hardcoded:

    AI_LLM_BASE_URL   OpenAI-compatible base (default https://api.openai.com/v1)
    AI_LLM_MODEL      model name, e.g. gpt-4o-mini
    AI_LLM_API_KEY    the key

Speed measures baked in:
- no chain-of-thought: reasoning/tracing is stripped with `reasoning_effort`
  (and automatically retried without it if a server rejects that field)
- low temperature, small max_tokens, non-streaming, short timeout
"""
import json
import os
import pathlib
import re
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional

_BASE = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-4o-mini"
REQUEST_TIMEOUT = 25.0
TEMPERATURE = 0.2     # near-deterministic: same gestures → same sensible line
MAX_TOKENS = 200      # one short sentence; headroom so a stray reasoning
                      # burst can't truncate the answer to empty


def _env(key: str, default: str = "") -> str:
    """os.environ first, then a direct .env parse (the server never loads
    .env itself, so the clef adaptor grew this same fallback)."""
    raw = (os.environ.get(key) or "").strip()
    if not raw:
        env_path = _BASE / ".env"
        if env_path.is_file():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith(f"{key}="):
                    raw = line.split("=", 1)[1].split("#", 1)[0].strip()
                    break
    return raw if raw else default

SYSTEM_PROMPT = """\
You render a mute person's ordered hand gestures into ONE natural spoken \
sentence. You ARE the mute person — speak in first person, directly to the \
partner in front of you, as if saying it out loud.

STRICT:
- NEVER mention gestures, meanings, options, or the words in parentheses.
  Never explain, describe, or comment. Output the spoken words ONLY.
- If a gesture means a question, end with "?". If it's a greeting, greet.
- Add natural grammar (articles, contractions, verb forms) but invent \
 NOTHING beyond what the gestures support.

Method:
1. Gestures arrive in the EXACT order they were performed.
2. Among each gesture's possible meanings, pick the one that fits the \
whole sequence best.
3. Compose ONE short sentence that makes the most sense in a real-life \
conversation.

Examples:
i("i, me") → look("look, appear, see") → good("good, well, nice") → \
question("?")
    Am I looking good?
hello("hi, hello") → you("you") → need("need, want") → help("help")
    Hi, do you need help?
water("water, drink") → where("where")
    Where is the water?
name("name") → RAVI(spelled word)
    My name is Ravi.
you("you") → look("look, appear, see") → good("good, well, nice")
    You look good.
"""


def base_url() -> str:
    return _env("AI_LLM_BASE_URL", DEFAULT_BASE_URL)


def model_id() -> str:
    return _env("AI_LLM_MODEL", DEFAULT_MODEL)


def api_key() -> str:
    return _env("AI_LLM_API_KEY")


def reasoning_effort() -> str:
    """Finding from live A/B (qwen3.8-27b on Groq):
    "low" still triggers chain-of-thought → sometimes a 5 s stall, and the
    thinking can consume the whole token budget → empty sentence error.
    "none" = no reasoning at all, ~0.4 s, always full answers."""
    return _env("AI_LLM_REASONING", "none")


def gateway_id() -> str:
    """AI Gateway routing header (cf-aig-gateway-id); blank/none = plain call."""
    return _env("AI_LLM_GATEWAY", "default")


def is_configured() -> bool:
    return bool(api_key())


# ---------- prompt ----------

def build_user_prompt(slots: List[Dict[str, Any]], history: Optional[List[Dict[str, str]]] = None) -> str:
    """Slots are in performance order: {"type":"value","tag":...,"options":[…]}
    for gestures and {"type":"fixed","word":…} for spell-token words."""
    lines = []
    if history:
        lines.append("Recent conversation:")
        lines.extend(
            f"- {h.get('role', 'U')}: {h.get('text', '')}" for h in history[-4:]
        )
        lines.append("")
    lines.append("gestures:")
    step = 0
    for slot in slots or []:
        if slot.get("type") == "fixed":
            lines.append(f'- spoken word "{slot.get("word", "")}" (fixed, from spelling mode)')
            continue
        step += 1
        options = slot.get("options") or []
        meanings = " / ".join(options) if options else "unknown"
        lines.append(f'{step}. {slot.get("tag", "?")} (meanings: {meanings})')
    lines.append("Sentence:")
    return "\n".join(lines)


def _strip_answer(text: str) -> str:
    out = (text or "").strip()
    # qwen-family models sometimes wrap reasoning in think tags -- keep
    # only what comes after the last closing tag
    marker = chr(0x3C) + "think" + chr(0x3E)
    if marker in out:
        out = out.split(marker + chr(0x2F))[-1].strip()
    out = re.sub(r"^(?:sentence|answer)\s*[:\-\u2014]\s*", "", out, flags=re.I)
    return out.strip().strip('"').strip("'").strip()


# ---------- transport (persistent keep-alive connection) ----------

class _HTTPish(Exception):
    def __init__(self, code: int, detail: str = ""):
        super().__init__(f"HTTP {code}: {detail}")
        self.code = code
        self.detail = detail


_conn_cache: Dict[str, Any] = {}
_conn_lock = threading.Lock()


def _new_conn(url: str):
    import http.client as hc
    from urllib.parse import urlparse
    u = urlparse(url)
    if u.scheme != "https":
        raise RuntimeError(f"Only HTTPS base URLs are supported, got: {url}")
    return hc.HTTPSConnection(u.hostname, u.port or 443, timeout=REQUEST_TIMEOUT)


def _get_conn(url: str):
    """One reused HTTPS connection per base-url. No pre-probe: a dead socket
    surfaces on the real request, and compose() rebuilds + retries once."""
    with _conn_lock:
        conn = _conn_cache.get(url)
        if conn is None:
            conn = _new_conn(url)
            _conn_cache[url] = conn
    return conn


def _reset_conn(url: str) -> None:
    with _conn_lock:
        old = _conn_cache.pop(url, None)
    if old is not None:
        try:
            old.close()
        except Exception:
            pass


def warm_up() -> None:
    """Pre-open the TLS connection so the first real gesture sequence doesn't
    pay the DNS+TCP+TLS setup cost (which shows up as multi-second latency on
    some edges). Best effort — failures are silent."""
    def _job():
        try:
            conn = _get_conn(base_url())
            conn.connect()  # force DNS+TCP+TLS now, not on the first gesture
        except Exception:
            _reset_conn(base_url())
    threading.Thread(target=_job, name="mova-llm-warmup", daemon=True).start()


def compose(
    slots: List[Dict[str, Any]],
    history: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """One completion call → {"selected", "model", "latency_ms", "usage"}.

    Raises RuntimeError on config/network problems — the orchestrator
    surfaces that as a visible error event, never a silent hang.
    """
    start = time.perf_counter()
    result: Dict[str, Any] = {
        "model": model_id(),
        "selected": None,
        "confidence": None,
        "probabilities": None,
        "latency_ms": 0.0,
    }
    key = api_key()
    if not key:
        raise RuntimeError("AI_LLM_API_KEY / AI_LLM_MODEL missing in .env")

    body: Dict[str, Any] = {
        "model": model_id(),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(slots or [], history)},
        ],
    }
    # strip thinking/reasoning traces where the server understands it;
    # retried below without the field when a server rejects it.
    effort = reasoning_effort()
    if effort and effort.lower() != "off":
        body["reasoning_effort"] = effort

    # ---- persistent connection -------------------------------------------
    # a fresh urllib call re-does DNS + TCP + TLS every time (~seconds of
    # variance on flaky free-tier edges); one keep-alive connection reused
    # across calls removes all of that.
    conn = _get_conn(base_url())
    _path_prefix = urlparse(base_url()).path.rstrip("/")

    def _flush_request(payload: Dict[str, Any], use_reasoning: bool = True):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
            # bare urllib UA gets 403'd (error 1010) by several edges
            "User-Agent": "MOVA-sentence/1.0",
            # identical gesture sequences must never replay a cached reply
            "cf-aig-skip-cache": "true",
        }
        gw = gateway_id()
        if gw and gw.lower() != "none":
            headers["cf-aig-gateway-id"] = gw
        req_body = dict(payload)
        if not use_reasoning:
            req_body.pop("reasoning_effort", None)
        conn.request(
            "POST",
            _path_prefix + "/chat/completions",
            body=json.dumps(req_body).encode("utf-8"),
            headers=headers,
        )
        t_first = time.perf_counter()
        resp = conn.getresponse()
        raw = resp.read()
        t_read = time.perf_counter()
        result["setup_ms"] = round((t_first - start) * 1000.0, 2)
        result["net_ms"] = round((t_read - t_first) * 1000.0, 2)
        if resp.status != 200:
            detail = raw.decode("utf-8", "ignore")[:400]
            try:
                conn.close()
            except Exception:
                pass
            _reset_conn(base_url())
            raise _HTTPish(resp.status, detail)
        return json.loads(raw.decode("utf-8"))

    use_reasoning = "reasoning_effort" in body
    try:
        payload = _flush_request(body)
    except _HTTPish as exc:
        # incompatible models reject unknown params (#400) — retry once
        # without the reasoning kwarg before giving up
        if exc.code == 400 and use_reasoning:
            try:
                payload = _flush_request(body, use_reasoning=False)
            except _HTTPish as exc2:
                raise RuntimeError(f"LLM HTTP {exc2.code}: {exc2.detail or 'request failed'}") from exc2
        else:
            raise RuntimeError(f"LLM HTTP {exc.code}: {exc.detail or 'request failed'}") from exc
    except (TimeoutError, OSError) as exc:
        # stale/broken socket → rebuild once and try again, then give up
        _reset_conn(base_url())
        try:
            payload = _flush_request(body)
        except Exception as exc2:
            raise RuntimeError(f"LLM network error: {exc2}") from exc2

    result["latency_ms"] = (time.perf_counter() - start) * 1000.0
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise RuntimeError(f"LLM returned no message content: {json.dumps(payload)[:300]}")
    result["usage"] = payload.get("usage")
    result["selected"] = _strip_answer(text)
    if not result["selected"]:
        raise RuntimeError("LLM returned an empty sentence")
    return result
