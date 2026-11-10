"""MOVA Sign Space tracker.

Owns the which-space-is-active state machine and computes the gesture
candidate list for scoped matching (docs/005-spell-mode.md):

- standard: all "normal" gestures + every space's entry/toggle (+ exit) signs
- space X:  X's members + X's control signs only

Because scoping happens before scoring, a letter sign can never collide with
a normal keyword — group-vs-group competition is eliminated at the source.
"""

from typing import Dict, Any, List, Optional
import time

# Roles that stay scoreable outside any space.
CONTROL_ROLES = {"space-entry", "space-exit", "space-toggle"}

# After a "quiet" auto-close the space closes immediately upon accepting the sign.
# Normal gestures are briefly locked out (150ms) so the held/transitioning pose
# doesn't immediately misfire as a standard lookalike. Control signs (entry/toggle)
# remain scoreable.
_QUIET_COOLDOWN_S = 0.15


class SpaceTracker:
    def __init__(self, gesture_store, space_store):
        self.gestures = gesture_store
        self.spaces = space_store
        self.active_space_id: Optional[str] = None
        self.space_buffer: List[str] = []
        # Deferred close (auto_close): the space's job is done but the hand
        # is still in frame holding the sign. The space stays scoped so the
        # lingering pose can't fall back into Standard scope and match a
        # lookalike gesture. Cleared by flush_hand_off() when hands leave.
        self.pending_hand_off = False
        # join_mode of the space holding the deferred close — the pipeline
        # needs it to decide whether closing also resets the sentence
        # collection (one_shot: yes; words: the phrase keeps collecting).
        self.pending_join_mode: Optional[str] = None
        self.last_flush_join_mode: Optional[str] = None
        self._cooldown_until: float = 0.0
        # name -> (id, role); rebuilt lazily
        self._role_by_name: Dict[str, Dict[str, str]] = {}

    # ------------------------------------------------------------------ index

    def refresh(self) -> None:
        """Rebuild the name -> {id, role} map from the gesture store."""
        self._role_by_name = {
            g["name"]: {"id": g["id"], "role": g.get("role", "normal")}
            for g in self.gestures.get_all()
            if g.get("name")
        }

    def _space_of_gesture_id(self, gesture_id: str) -> Optional[Dict[str, Any]]:
        return self.spaces.space_of_gesture(gesture_id)

    def _info(self, gesture_name: str) -> Optional[Dict[str, str]]:
        return self._role_by_name.get(gesture_name)

    # ------------------------------------------------------------- candidates

    def scoped_candidates(self) -> List[Dict[str, Any]]:
        """Gesture list handed to score_gestures for the current mode."""
        self.refresh()
        if self.active_space_id:
            space = self.spaces.get_by_id(self.active_space_id)
            if space is None:
                self.active_space_id = None
                return self._standard_candidates()
            member_ids = set(space.get("gesture_ids", []))
            return [g for g in self.gestures.get_all() if g.get("id") in member_ids]
        if time.monotonic() < self._cooldown_until:
            # During cooldown: normal gestures stay locked out (so a lingering/transitioning
            # pose can't match a lookalike). Only control signs keep working (e.g. to re-enter a group).
            return [g for g in self.gestures.get_all() if g.get("role", "normal") in CONTROL_ROLES]
        return self._standard_candidates()

    def _standard_candidates(self) -> List[Dict[str, Any]]:
        return [g for g in self.gestures.get_all() if g.get("role", "normal") in {"normal"} | CONTROL_ROLES]

    # ----------------------------------------------------------- state machine

    def on_accepted(self, gesture_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Feed an 'accepted' buffer event. Returns a space token dict when a
        space window closed and its letters flushed, else None."""
        if not gesture_name:
            return None
        self.refresh()
        info = self._info(gesture_name)
        if info is None:
            return None
        role = info["role"]
        if self.pending_hand_off and role not in CONTROL_ROLES:
            # Deferred close in progress and the hand still holds a member
            # pose: member re-triggers are swallowed — but this space's own
            # control signs (toggle/exit) stay fully working, so the user
            # can always close manually, no matter how auto_close is set.
            return None

        if role == "space-sign":
            space = self._space_of_gesture_id(info["id"])
            if space and self.active_space_id == space["id"]:
                join_mode = space.get("join_mode", "words")
                if join_mode == "spell":
                    self.space_buffer.append(gesture_name)
                else:
                    # Words mode: every accepted member is its own word
                    # token, emitted immediately — same packaging as a
                    # normal gesture, only namespaced.
                    token = {
                        "type": "space",
                        "space": space.get("key"),
                        "space_id": space["id"],
                        "mode": "words",
                        "letters": [gesture_name],
                    }
                    # Auto-close groups: the sign was already marked, so the
                    # space can close.
                    # In "quiet" (on-screen) mode: close immediately with a tiny
                    # 150ms cooldown so the held pose doesn't immediately false-trigger.
                    # In "hand_off" (move away) mode: hold scope until hand leaves camera.
                    # Spell keeps chaining, so it never auto-closes.
                    if space.get("auto_close", False) and join_mode == "words":
                        if space.get("close_mode", "hand_off") == "quiet":
                            self.deactivate()
                            self._cooldown_until = time.monotonic() + _QUIET_COOLDOWN_S
                        else:
                            self.pending_hand_off = True
                            self.pending_join_mode = "words"
                    return token
            return None

        if role in CONTROL_ROLES:
            space = self._space_of_gesture_id(info["id"])
            if space is None:
                return None
            sid = space["id"]

            if role == "space-entry":
                if not self.active_space_id:
                    self._activate(sid)
                return None

            if role == "space-toggle":
                if self.active_space_id == sid:
                    return self._flush()
                if not self.active_space_id:
                    self._activate(sid)
                return None

            if role == "space-exit":
                if self.active_space_id == sid:
                    return self._flush()
                return None

        return None

    def on_completed(self) -> Optional[Dict[str, Any]]:
        """The buffer fired 'completed' (inactivity close). If a space was
        open it closes here too and flushes its letters."""
        if self.pending_hand_off:
            # Inactivity timeout may fire while the hand still holds the
            # just-marked sign — the real close is the hand-off, not this.
            return None
        if self.active_space_id:
            space = self.spaces.get_by_id(self.active_space_id)
            if space and not space.get("auto_close", True):
                # auto_close is OFF: only toggle/exit sign closes the group.
                # Do NOT close on inactivity / hand-away.
                return None
            return self._flush()
        return None

    def role_of(self, gesture_name: Optional[str]) -> str:
        """Gesture role for queue/AI packaging ("normal" for unknown names)."""
        info = self._info(gesture_name or "")
        return info["role"] if info else "normal"

    def on_reset(self) -> None:
        """Hard reset (client 'reset' action): back to standard, no token."""
        self.active_space_id = None
        self.space_buffer = []
        self.pending_hand_off = False
        self.pending_join_mode = None
        self.last_flush_join_mode = None
        self._cooldown_until = 0.0

    def deactivate(self) -> None:
        """Leave the active space silently (one-shot speech already spoke)."""
        self.active_space_id = None
        self.space_buffer = []
        self.pending_hand_off = False
        self.pending_join_mode = None

    def flush_hand_off(self) -> bool:
        """Deferred-close counterpart: called by the pipeline when no hands
        are visible. True exactly once when the pending space is closed.
        `last_flush_join_mode` tells the caller what kind of exchange ended:
        one_shot = spoken, done → caller resets the collection; words = the
        phrase must keep collecting, closing only restores Standard scope."""
        if not self.pending_hand_off:
            self.last_flush_join_mode = None
            return False
        self.last_flush_join_mode = self.pending_join_mode
        self.deactivate()
        return True

    def _activate(self, space_id: str) -> None:
        self.active_space_id = space_id
        self.space_buffer = []
        self._cooldown_until = 0.0

    def _flush(self) -> Optional[Dict[str, Any]]:
        if not self.active_space_id:
            self.space_buffer = []
            return None
        space = self.spaces.get_by_id(self.active_space_id)
        token = None
        if space and self.space_buffer and space.get("join_mode", "words") == "spell":
            token = {
                "type": "space",
                "space": space.get("key"),
                "space_id": space["id"],
                "mode": "spell",
                "letters": list(self.space_buffer),
            }
        elif space and space.get("join_mode") == "words":
            # Words-mode members were already emitted per acceptance; the
            # bucket holds nothing to flush.
            token = None
        self.active_space_id = None
        self.space_buffer = []
        self.pending_hand_off = False
        self.pending_join_mode = None
        return token

    # ------------------------------------------------------------------- state

    def get_state(self) -> Dict[str, Any]:
        space = self.spaces.get_by_id(self.active_space_id) if self.active_space_id else None
        return {
            "active_space": space["key"] if space else None,
            "space_buffer": list(self.space_buffer),
        }
