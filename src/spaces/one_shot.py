"""MOVA One-shot speech layer.

A one-shot sign space (join_mode == "one_shot") holds gestures that each
carry a pre-baked spoken sentence (gesture["shot"]). While that space is
active, the FIRST accepted member gesture speaks its sentence immediately:

- there is no queue round-trip and no AI processing — the audio file was
  generated at registration time, so playback latency is zero;
- the space deactivates right after speaking, exactly like a spoken
  utterance: one sign, one sentence, done — re-activate to speak again;
- anything buffered BEFORE activation is irrelevant: candidate scoping
  already prevents foreign gestures from matching inside the space.

This module stays deliberately separate from SpaceTracker and the
detection pipeline beyond a single hook call per accepted gesture.
"""

from typing import Any, Dict, Optional
import time


class OneShotSpeaker:
    """Detects one-shot speech moments and reports what the client should say."""

    def __init__(self, gesture_store, space_store, space_tracker):
        self.gestures = gesture_store
        self.spaces = space_store
        self.tracker = space_tracker
        # pending_hand_off now lives on the tracker so auto-close words
        # spaces and one-shot spaces share the exact same deferred close.

    def on_accepted(self, gesture_name: Optional[str]) -> Optional[Dict[str, Any]]:
        """Call after each accepted gesture.

        Returns a speech payload dict when the accepted gesture triggered a
        one-shot sentence, else None:
            {"source", "sentence", "gesture", "audio_url", "space", "space_id"}

        First accepted member sign speaks and marks the space for deferred
        deactivation; additional signs while the hand lingers are ignored —
        the exchange already happened and re-speaking needs re-activation.
        """
        if not gesture_name or not self.tracker.active_space_id:
            return None
        if self.tracker.pending_hand_off:
            return None  # wait for the hand to leave before closing out
        space = self.spaces.get_by_id(self.tracker.active_space_id)
        if not space or space.get("join_mode") != "one_shot":
            return None

        gesture = self.gestures.get_by_name(gesture_name)
        if not gesture:
            return None
        if gesture.get("id") not in set(space.get("gesture_ids", [])):
            return None

        shot = gesture.get("shot") or {}
        if not shot.get("ready", False) or not shot.get("sentence"):
            return None

        audio_key = shot.get("audio_key")
        payload = {
            "source": "one_shot",      # future layers: "ai", "converter", …
            "sentence": shot["sentence"],
            "gesture": gesture["name"],
            "audio_url": f"/audio/{audio_key}" if audio_key else None,
            "space": space.get("key"),
            "space_id": space["id"],
        }
        # One sign, one sentence. With auto-close (the default for one-shot
        # spaces):
        # - in "quiet" (on-screen) mode: close immediately with a tiny 150ms cooldown.
        # - in "hand_off" (move away) mode: hold scope until hand leaves camera.
        # With auto-close OFF the user keeps the space open — further member signs speak
        # again immediately — and closes it via its toggle/exit sign.
        if space.get("auto_close", True):
            if space.get("close_mode", "hand_off") == "quiet":
                self.tracker.deactivate()
                self.tracker._cooldown_until = time.monotonic() + 0.15
            else:
                self.tracker.pending_hand_off = True
                self.tracker.pending_join_mode = "one_shot"
        return payload

    def flush_hand_off(self) -> bool:
        """Finish the exchange once no hands are visible: deactivate the
        space. The pipeline clears the buffer alongside this call."""
        return self.tracker.flush_hand_off()

    def on_reset(self) -> None:
        """Hard reset: drop any pending hand-off immediately."""
        self.tracker.pending_hand_off = False
