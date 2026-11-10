"""Workspace manager — fully separate experiment datasets.

A workspace is one complete dataset: its own gestures.json,
sign_spaces.json, gesture_previews/ (captured user images) and audio/
(one-shot speech bakes). Switching swaps ALL of it — nothing is merged
or carried over, which is exactly what you want when testing a totally
new dataset or per-user sets.

Layout:

    data/gestures.json, data/sign_spaces.json, …   ← workspace "default"
    data/workspaces/<name>/gestures.json …         ← extra workspaces
    data/workspaces.json                           ← {active, workspaces}

"active" survives restarts. Names: lowercase letters/digits/-/_.
"""
import json
import re
import threading
import time
from pathlib import Path

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class WorkspaceManager:
    def __init__(self, base: str = "data"):
        self.base = Path(base)
        self.manifest_path = self.base / "workspaces.json"
        self.workspaces_root = self.base / "workspaces"
        self.workspaces_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._manifest = self._load_manifest()

    # ------------------------------------------------------------ manifest

    def _load_manifest(self) -> dict:
        manifest = {"active": "default", "workspaces": []}
        loaded = {}
        try:
            with open(self.manifest_path, encoding="utf-8") as fh:
                loaded = json.load(fh) or {}
                manifest.update(loaded)
        except (OSError, ValueError):
            pass
        # any folder that exists counts as a workspace even if the manifest
        # entry was hand-deleted
        known = set(manifest.get("workspaces") or [])
        for child in self.workspaces_root.iterdir():
            if child.is_dir() and child.name not in known and _NAME_RE.match(child.name or ""):
                known.add(child.name)
        if "default" in known:
            known.remove("default")
        manifest["workspaces"] = sorted(known)
        # A fresh server boot STARTS ON DEFAULT — no session leftovers.
        # (The previous active is kept only as a hint for humans reading the
        # file; runtime switches update it so admins can see what's live.)
        manifest["active"] = "default"
        manifest["_last_active"] = (loaded or {}).get("active", "default")
        return manifest

    def _save_manifest(self) -> None:
        with self._lock:
            with open(self.manifest_path, "w", encoding="utf-8") as fh:
                json.dump(self._manifest, fh, indent=2)

    # ------------------------------------------------------------ public

    def active(self) -> str:
        with self._lock:
            return self._manifest.get("active", "default")

    def list(self) -> list:
        with self._lock:
            return ["default"] + list(self._manifest.get("workspaces", []))

    def create(self, name: str) -> dict:
        """Fresh empty dataset folder. Returns {"ok": bool, "error"/self}."""
        name = (name or "").strip().lower()
        if not _NAME_RE.match(name):
            return {"ok": False, "error": "name must be 1-32 chars: a-z 0-9 - _"}
        with self._lock:
            if name == "default":
                return {"ok": False, "error": "'default' is the built-in workspace"}
            if name in self._manifest.get("workspaces", []):
                return {"ok": False, "error": f"workspace '{name}' already exists"}
            root = self.space_dir(name)
            root.mkdir(parents=True, exist_ok=True)
            for fn, stub in (("gestures.json", {"schema_version": 2, "gestures": []}),
                             ("sign_spaces.json", {"schema_version": 1, "spaces": []})):
                fp = root / fn
                if not fp.exists():
                    fp.write_text(json.dumps(stub, indent=2), encoding="utf-8")
            self._manifest.setdefault("workspaces", []).append(name)
            self._manifest["workspaces"] = sorted(self._manifest["workspaces"])
            self._save_manifest()
            return {"ok": True}

    def set_active(self, name: str) -> bool:
        with self._lock:
            if name not in ([("default")] + self._manifest.get("workspaces", [])):
                return False
            self._manifest["active"] = name
            self._manifest["_last_active"] = name  # informational only
            self._save_manifest()
            return True

    # ------------------------------------------------------------ paths

    def space_dir(self, name: str = None) -> Path:
        """Real dataset folder for a workspace. 'default' = the root data/."""
        name = (name or self.active()).strip().lower()
        if name == "default":
            return self.base
        return self.workspaces_root / name

    def gestures_path(self, name: str = None) -> str:
        return str(self.space_dir(name) / "gestures.json")

    def spaces_path(self, name: str = None) -> str:
        return str(self.space_dir(name) / "sign_spaces.json")

    def previews_dir(self, name: str = None) -> Path:
        d = self.space_dir(name) / "gesture_previews"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def audio_dir(self, name: str = None) -> Path:
        d = self.space_dir(name) / "audio"
        d.mkdir(parents=True, exist_ok=True)
        return d


_manager: WorkspaceManager = None
_manager_lock = threading.Lock()


def get_workspace_manager() -> WorkspaceManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = WorkspaceManager()
    return _manager
