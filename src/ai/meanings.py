"""Tag → candidate meanings resolver.

Maps every collected gesture tag (gesture name) to the list of words it
could mean. Resolution order (the user's own vocabulary always wins):

1. registered labels   → `data/gestures.json` gestures[name].labels — the
   tags the user typed at registration ("palm_open_five" → hi, hello, 5)
2. `data/meanings.json` → manual overrides, `{"gesture_name": ["word", …]}`
   (extras for gestures that have no labels)
3. experiment dataset   → `experiments/gesture_permutations_pick/dataset.json`
   (matched by exact name; those shapes carry curated `meanings`)
4. fallback             → the tag itself, underscores turned to spaces.
"""
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List

_BASE = Path(__file__).resolve().parent.parent.parent
_OVERRIDES_PATH = _BASE / "data" / "meanings.json"
_EXPERIMENT_PATH = (
    _BASE / "experiments" / "gesture_permutations_pick" / "dataset.json"
)


@lru_cache(maxsize=8)
def _registered_labels(key) -> Dict[str, List[str]]:
    """Name → labels map from the gesture store keydata. Cached against
    (workspace, file mtime) so UI edits and workspace switches go live
    without a restart."""
    workspace, mtime_ns = key
    from src.storage.workspace import get_workspace_manager
    path = get_workspace_manager().gestures_path(workspace)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        items = raw.get("gestures", [])
        if isinstance(items, dict):
            items = list(items.values())
        return {
            g.get("name"): [str(w).strip() for w in g.get("labels", []) if str(w).strip()]
            or []
            for g in items
            if isinstance(g, dict) and g.get("name")
        }
    except (OSError, ValueError):
        return {}


def _labels_now() -> Dict[str, List[str]]:
    from src.storage.workspace import get_workspace_manager
    ws = get_workspace_manager()
    path = Path(ws.gestures_path())
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        return {}
    return _registered_labels((ws.active(), mtime_ns))


@lru_cache(maxsize=1)
def _experiment_meanings() -> Dict[str, List[str]]:
    try:
        with open(_EXPERIMENT_PATH, encoding="utf-8") as fh:
            raw = json.load(fh)
        return {
            name: [str(w) for w in entry.get("meanings", [])]
            for name, entry in raw.get("gestures", {}).items()
            if isinstance(entry, dict) and entry.get("meanings")
        }
    except (OSError, ValueError):
        return {}


def resolve_meanings(tag: str) -> List[str]:
    """One tag → candidate word list (never empty)."""
    tag = str(tag or "").strip()
    if not tag:
        return ["…"]

    # 1) the tags the user registered with this gesture — the real vocabulary
    own = _labels_now().get(tag)
    if own:
        return own

    # 2) manual AI-vocabulary overrides
    try:
        with open(_OVERRIDES_PATH, encoding="utf-8") as fh:
            overrides = json.load(fh)
        manual = [str(w) for w in overrides.get(tag, []) if str(w).strip()]
        if manual:
            return manual
    except (OSError, ValueError):
        pass

    # 3) curated experiment readings
    curated = _experiment_meanings().get(tag)
    if curated:
        return curated

    # 4) no tags anywhere → the name itself is the only reading
    return [tag.replace("_", " ") or "…"]
