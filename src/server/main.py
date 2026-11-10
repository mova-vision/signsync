from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import base64
import cv2
import numpy as np
import json
import asyncio
import time
import uuid
from collections import deque
from pathlib import Path

from src.storage.gesture_store import get_store
from src.storage.sign_space_store import get_space_store
from src.vision.mediapipe_processor import (
    MediaPipeProcessor, 
    extract_features, 
    extract_two_hand_features,
    normalize_landmarks,
    score_gestures,
)
from src.pipeline.detection_pipeline import RealTimePipeline
from src.buffer.gesture_buffer import GestureBuffer

app = FastAPI(title="MOVA Gesture Registration")

# The stream-capture page (node server, e.g. :3300) posts frames cross-origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _warm_tts() -> None:
    """Pre-load neural TTS voices in the background so the first Generate
    doesn't stall on model loading. Never blocks or breaks startup."""
    def _load():
        from audio_adaptor import get_speech_adaptor
        problems = get_speech_adaptor().warm_up()
        for problem in problems:
            print(f"[tts-warmup] {problem}", flush=True)

    import os as _os
    import threading as _threading
    if (_os.environ.get("TTS_PROVIDER") or "piper").strip().lower() in ("piper", "piper-tts"):
        _threading.Thread(target=_load, name="tts-warmup", daemon=True).start()


BASE_DIR = Path(__file__).resolve().parent.parent.parent
PREVIEW_DIR = BASE_DIR / "data" / "gesture_previews"
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR = BASE_DIR / "data" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/gesture-previews", StaticFiles(directory=str(PREVIEW_DIR)), name="gesture-previews")

# --- Workspaces (namespaced experiment datasets) -----------------------------
# /wsdata serves every extra workspace's previews+audio; /audio = route below
# resolves across the ACTIVE workspace and the shared default audio dir.
# NOTE: must NOT start with "/ws" — that would shadow the /ws/pipeline and
# /ws/detect WebSocket routes (StaticFiles asserts http scope and would
# swallow every playground connection).
from src.storage.workspace import get_workspace_manager  # noqa: E402

WSMAN = get_workspace_manager()
app.mount("/wsdata", StaticFiles(directory=str(WSMAN.workspaces_root)), name="workspaces")


def _previews_dir() -> Path:
    """Previews of the ACTIVE workspace ('default' keeps the root dir)."""
    if WSMAN.active() == "default":
        return PREVIEW_DIR
    return WSMAN.previews_dir()


def _preview_url(file_name: str) -> str:
    if WSMAN.active() == "default":
        return f"/gesture-previews/{file_name}"
    return f"/wsdata/{WSMAN.active()}/gesture_previews/{file_name}"


def _audio_dir() -> Path:
    """Audio dir of the ACTIVE workspace (files also served from the shared
    default dir as a fallback so nothing regenerates needlessly)."""
    if WSMAN.active() == "default":
        return AUDIO_DIR
    return WSMAN.audio_dir()


@app.get("/audio/{key:path}")
async def serve_audio(key: str):
    """Resolve /audio/{key} across the active workspace, then the shared
    default audio dir — one URL surface for every workspace."""
    from fastapi.responses import FileResponse
    parts = Path(key).parts
    if not parts or any(p in ("..", "", ".") for p in parts):
        raise HTTPException(404, "audio not found")
    gone = [d for d in (_audio_dir(), AUDIO_DIR)]
    seen = set()
    for root in gone:
        rp = str(root.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        candidate = (root / key).resolve()
        if candidate.is_file() and str(candidate).startswith(rp):
            return FileResponse(candidate)
    raise HTTPException(404, "audio not found")


templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Lazy-loaded processor
processor = None
store = get_store()
space_store = get_space_store()

def get_processor():
    global processor
    if processor is None:
        processor = MediaPipeProcessor()
    return processor

active_captures: Dict[str, Dict] = {}


class CaptureStartRequest(BaseModel):
    # The UI detects the number of visible hands from each frame.
    hand_count: int = 2


class CaptureFrameRequest(BaseModel):
    image_base64: str
    capture_id: str


class DetectFrameRequest(BaseModel):
    image_base64: str


class SaveGestureRequest(BaseModel):
    capture_id: str
    name: str
    labels: List[str] = []
    role: str = "normal"          # see GestureStore.ROLES
    space_id: Optional[str] = None  # required for any space-* role
    sentence: Optional[str] = None  # one-shot members: spoken transcript
    voice_gender: str = "male"      # "male" | "female"


class UpdateGestureRequest(BaseModel):
    name: Optional[str] = None
    labels: Optional[List[str]] = None
    role: Optional[str] = None
    space_id: Optional[str] = None


class SpaceRequest(BaseModel):
    name: str
    toggle_mode: str = "shared"
    join_mode: str = "words"  # "spell", "words", "one_shot"
    auto_close: Optional[bool] = None  # None = mode default (one_shot: on)


class SpaceUpdateRequest(BaseModel):
    name: Optional[str] = None
    toggle_mode: Optional[str] = None
    join_mode: Optional[str] = None
    auto_close: Optional[bool] = None


class GestureSentenceRequest(BaseModel):
    sentence: str
    gender: str = "male"  # "male" | "female"


class WorkspaceCreateRequest(BaseModel):
    name: str


def _link_gesture_to_space(gesture_id: str, space_id: str, role: str) -> Optional[str]:
    """Attach a gesture to a space and set its role. Returns an error message
    or None on success."""
    space = space_store.get_by_id(space_id)
    if not space:
        return "Sign space not found"
    if gesture_id not in space["gesture_ids"]:
        space_store.add_member(space_id, gesture_id)
    store.set_role(gesture_id, role)
    return None


def _unlink_gesture_from_all_spaces(gesture_id: str) -> None:
    for space in space_store.get_all():
        space_store.remove_member(space["id"], gesture_id)


def decode_base64_image(data: str) -> np.ndarray:
    if "," in data:
        data = data.split(",")[1]
    img_bytes = base64.b64decode(data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def encode_image_to_base64(frame: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buffer).decode('utf-8')


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/admin", response_class=HTMLResponse)
async def admin_root(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/admin/register", response_class=HTMLResponse)
async def admin_register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get("/admin/groups", response_class=HTMLResponse)
async def admin_groups(request: Request):
    return templates.TemplateResponse("groups.html", {"request": request})


@app.get("/admin/gestures", response_class=HTMLResponse)
async def admin_gestures(request: Request):
    return templates.TemplateResponse("gestures.html", {"request": request})


@app.get("/admin/live-test", response_class=HTMLResponse)
async def admin_live_test(request: Request):
    return templates.TemplateResponse("live_test.html", {"request": request})


@app.get("/admin/pipeline", response_class=HTMLResponse)
async def admin_pipeline(request: Request):
    return templates.TemplateResponse("pipeline.html", {"request": request})


@app.get("/pipeline", response_class=HTMLResponse)
async def pipeline_page(request: Request):
    return templates.TemplateResponse("pipeline.html", {"request": request})


@app.get("/gestures", response_class=HTMLResponse)
async def gestures_page(request: Request):
    return templates.TemplateResponse("gestures.html", {"request": request})


@app.get("/playground", response_class=HTMLResponse)
async def playground_page(request: Request):
    return templates.TemplateResponse("playground.html", {"request": request})


@app.get("/api/playground/library")
async def playground_library():
    """Lightweight gesture metadata (no feature vectors) for playground cards."""
    return {
        "gestures": [
            {
                "id": g.get("id"),
                "name": g.get("name"),
                "hand_count": g.get("hand_count", 1),
                "preview_path": g.get("preview_path"),
                "landmarks": g.get("landmarks", []),
                "role": g.get("role", "normal"),
            }
            for g in store.get_all()
        ],
        "spaces": space_store.get_state(),
    }


@app.post("/api/capture/start")
async def start_capture(req: CaptureStartRequest):
    capture_id = str(int(time.time() * 1000))
    active_captures[capture_id] = {
        "hand_count": req.hand_count,
        "started_at": time.time(),
        "frames": [],
        "landmarks": None,
        "features": None
    }
    return {"capture_id": capture_id, "hand_count": req.hand_count}


@app.post("/api/capture/frame")
async def process_frame(req: CaptureFrameRequest):
    if req.capture_id not in active_captures:
        raise HTTPException(404, "Capture session not found")
    
    capture = active_captures[req.capture_id]
    frame = decode_base64_image(req.image_base64)
    
    if frame is None:
        raise HTTPException(400, "Invalid image data")
    
    proc = get_processor()
    result = proc.process_frame(frame)
    annotated = proc.draw_landmarks(frame.copy(), result)
    
    hands = proc.extract_landmarks(result)
    handedness = proc.get_handedness(result)
    
    capture["last_frame"] = annotated
    capture["last_result"] = result
    capture["hands"] = hands
    capture["handedness"] = handedness
    
    if hands:
        # A single gesture can use one or two hands.  Do not make the user
        # choose in advance; preserve the actual count from the good frame.
        if len(hands) == 1:
            features = extract_features(hands[0])
            capture["features"] = features
            capture["landmarks"] = hands[0]
            capture["hand_count"] = 1
        elif len(hands) >= 2:
            left_pts = None
            right_pts = None
            for i, h in enumerate(handedness):
                pts = hands[i]
                if h == "Left":
                    left_pts = pts
                else:
                    right_pts = pts
            if left_pts and right_pts:
                features = extract_two_hand_features(left_pts, right_pts)
                capture["features"] = features
                capture["landmarks"] = {"left": left_pts, "right": right_pts}
                capture["hand_count"] = 2
    
    return {
        "image_base64": encode_image_to_base64(annotated),
        "hand_count": len(hands),
        "handedness": handedness,
        "has_features": capture.get("features") is not None
    }


@app.post("/api/gestures/save")
async def save_gesture(req: SaveGestureRequest):
    if req.capture_id not in active_captures:
        raise HTTPException(404, "Capture session not found")
    
    capture = active_captures[req.capture_id]
    
    if not capture.get("features"):
        raise HTTPException(400, "No hand features captured. Make sure hand is visible.")
    
    if store.get_by_name(req.name):
        raise HTTPException(400, f"Gesture with name '{req.name}' already exists")

    role = req.role if req.role in store.ROLES else "normal"
    is_space_role = role.startswith("space-")
    if is_space_role and not req.space_id:
        raise HTTPException(400, "A sign space must be selected for space roles")

    landmarks = capture["landmarks"]
    if isinstance(landmarks, dict):
        landmarks_list = landmarks["left"] + landmarks["right"]
    else:
        landmarks_list = landmarks
    
    gesture_id = str(uuid.uuid4())[:8]
    preview_file = _previews_dir() / f"{gesture_id}.jpg"
    if not cv2.imwrite(str(preview_file), capture["last_frame"]):
        raise HTTPException(500, "Could not save gesture preview")

    gesture = store.create(
        gesture_id=gesture_id,
        name=req.name,
        landmarks=landmarks_list,
        hand_count=capture["hand_count"],
        handedness=capture.get("handedness", ["Right"])[0] if capture.get("handedness") and len(capture["handedness"]) == 1 else capture.get("handedness"),
        features=capture["features"],
        labels=req.labels,
        preview_path=_preview_url(preview_file.name),
        role=role
    )

    if is_space_role:
        error = _link_gesture_to_space(gesture_id, req.space_id, role)
        if error:
            store.delete(gesture_id)
            raise HTTPException(400, error)

    # One-shot members get their sentence baked as audio right at save time
    speech_error = None
    if req.sentence is not None and req.sentence.strip():
        space = space_store.get_by_id(req.space_id) if req.space_id else None
        if not space or space.get("join_mode") != "one_shot":
            store.delete(gesture_id)
            raise HTTPException(400, "Spoken sentences belong to one_shot groups")
        try:
            baked_key = await _bake_gesture_speech(gesture_id, req.sentence, req.voice_gender)
            store.set_shot(gesture_id, sentence=req.sentence, gender=req.voice_gender,
                           audio_key=baked_key, ready=True)
        except Exception as exc:
            speech_error = str(exc.detail if hasattr(exc, "detail") else exc)

    del active_captures[req.capture_id]

    if speech_error:
        return {"success": True, "gesture": gesture, "speech_error": speech_error}

    return {"success": True, "gesture": gesture}


@app.post("/api/detect/frame")
async def detect_gesture_frame(req: DetectFrameRequest):
    """HTTP fallback for analysing a browser camera frame."""
    try:
        return analyse_live_frame(req.image_base64)
    except ValueError as error:
        raise HTTPException(400, str(error))


def analyse_live_frame(image_base64: str) -> Dict[str, Any]:
    """Extract features and rank saved gestures for one browser frame."""
    try:
        frame = decode_base64_image(image_base64)
    except (ValueError, base64.binascii.Error):
        frame = None
    if frame is None:
        raise ValueError("Invalid image data")

    proc = get_processor()
    result = proc.process_frame(frame)
    hands = proc.extract_landmarks(result)
    handedness = proc.get_handedness(result)
    live_features = None

    if len(hands) == 1:
        live_features = extract_features(hands[0])
    elif len(hands) >= 2:
        left, right = None, None
        for index, side in enumerate(handedness):
            if side == "Left":
                left = hands[index]
            elif side == "Right":
                right = hands[index]
        if left is not None and right is not None:
            live_features = extract_two_hand_features(left, right)

    return {
        "hand_count": len(hands),
        "landmarks": [[[point[0], point[1]] for point in hand] for hand in hands],
        "scores": score_gestures(live_features, store.get_all(), live_handedness=handedness) if live_features else [],
    }


@app.websocket("/ws/detect")
async def detect_gesture_stream(websocket: WebSocket):
    """Persistent live gesture-score stream for the admin test screen."""
    await websocket.accept()
    try:
        while True:
            payload = json.loads(await websocket.receive_text())
            image_base64 = payload.get("image_base64")
            if not image_base64:
                await websocket.send_json({"error": "A camera frame is required"})
                continue
            try:
                await websocket.send_json(analyse_live_frame(image_base64))
            except (ValueError, cv2.error) as error:
                await websocket.send_json({"error": str(error)})
    except (WebSocketDisconnect, json.JSONDecodeError):
        return


@app.websocket("/ws/pipeline")
async def pipeline_stream(websocket: WebSocket):
    """Real-time gesture detection and buffering WebSocket stream."""
    await websocket.accept()
    pipeline = RealTimePipeline(store=store, processor_getter=get_processor)
    try:
        while True:
            raw_text = await websocket.receive_text()
            payload = json.loads(raw_text)

            # Handle control commands
            action = payload.get("action")
            if action in ("reset", "clear"):
                pipeline.reset_buffer()
                await websocket.send_json({
                    "action_acknowledged": action,
                    "buffer_state": pipeline.buffer.get_state()
                })
                continue
            elif action == "set_threshold":
                pipeline.set_threshold(float(payload.get("threshold", 60.0)))
                continue
            elif action == "set_timeout":
                pipeline.buffer.end_signal_frames = int(payload.get("frames", 25))
                continue
            elif action == "init":
                if "threshold" in payload:
                    pipeline.set_threshold(float(payload["threshold"]))
                if "frames" in payload:
                    pipeline.buffer.end_signal_frames = int(payload["frames"])
                if "window" in payload:
                    pipeline.set_window_size(int(payload["window"]))
                continue

            image_base64 = payload.get("image_base64")
            if not image_base64:
                continue

            try:
                frame = decode_base64_image(image_base64)
                if frame is None:
                    await websocket.send_json({"error": "Invalid camera frame"})
                    continue
                result = pipeline.process_frame(frame)
                await websocket.send_json(result)
            except (ValueError, cv2.error) as error:
                await websocket.send_json({"error": str(error)})
    except (WebSocketDisconnect, json.JSONDecodeError):
        return


@app.get("/api/pipeline/state")
async def get_pipeline_state():
    """Diagnostic check for pipeline module."""
    sample_buf = GestureBuffer()
    return {"status": "ok", "default_buffer": sample_buf.get_state()}


@app.get("/api/gestures")
async def list_gestures():
    gestures = store.get_all()
    return {"gestures": gestures}


@app.get("/api/gestures/{gesture_id}")
async def get_gesture(gesture_id: str):
    gesture = store.get_by_id(gesture_id)
    if not gesture:
        raise HTTPException(404, "Gesture not found")
    return gesture


@app.patch("/api/gestures/{gesture_id}")
async def update_gesture(gesture_id: str, req: UpdateGestureRequest):
    gesture = store.update(gesture_id, name=req.name, labels=req.labels)
    if not gesture:
        raise HTTPException(404, "Gesture not found")

    if req.role is not None:
        if req.role in store.ROLES:
            if req.role.startswith("space-"):
                if req.space_id:
                    _unlink_gesture_from_all_spaces(gesture_id)
                    error = _link_gesture_to_space(gesture_id, req.space_id, req.role)
                    if error:
                        raise HTTPException(400, error)
                else:
                    raise HTTPException(400, "A sign space must be selected for space roles")
            elif req.role == "normal":
                _unlink_gesture_from_all_spaces(gesture_id)
                store.set_role(gesture_id, "normal")
        return {"success": True, "gesture": store.get_by_id(gesture_id)}

    return {"success": True, "gesture": gesture}


@app.delete("/api/gestures/{gesture_id}")
async def delete_gesture(gesture_id: str):
    _unlink_gesture_from_all_spaces(gesture_id)
    if store.delete(gesture_id):
        return {"success": True}
    raise HTTPException(404, "Gesture not found")


@app.post("/api/capture/cancel/{capture_id}")
async def cancel_capture(capture_id: str):
    if capture_id in active_captures:
        del active_captures[capture_id]
    return {"success": True}


@app.get("/api/spaces")
async def list_spaces():
    """Sign spaces with lightweight display fields."""
    return {"spaces": space_store.get_state()}


@app.post("/api/spaces")
async def create_space(req: SpaceRequest):
    space = space_store.create(req.name, req.toggle_mode, req.join_mode, auto_close=req.auto_close)
    if not space:
        raise HTTPException(400, f"Space key already exists for '{req.name}'")
    return {"success": True, "space": space}


@app.patch("/api/spaces/{space_id}")
async def update_space(space_id: str, req: SpaceUpdateRequest):
    space = space_store.update(space_id, name=req.name, toggle_mode=req.toggle_mode,
                               join_mode=req.join_mode, auto_close=req.auto_close)
    if not space:
        raise HTTPException(404, "Space not found")
    return {"success": True, "space": space}


@app.delete("/api/spaces/{space_id}")
async def delete_space(space_id: str):
    member_ids = space_store.delete(space_id)
    for gid in member_ids:
        store.set_role(gid, "normal")
    return {"success": True, "released": member_ids}


async def _bake_gesture_speech(gesture_id: str, sentence: str, gender: str):
    """Synthesize & cache a gesture's one-shot sentence speech (piper/espeak/ElevenLabs)."""
    sentence = sentence.strip()
    if not sentence:
        raise HTTPException(400, "Sentence cannot be empty")
    if len(sentence) > 300:
        raise HTTPException(400, "Sentence too long (max 300 chars)")

    from audio_adaptor import get_speech_adaptor
    adaptor = get_speech_adaptor()
    ext = getattr(adaptor.provider, "AUDIO_EXT", "mp3")
    audio_key = f"gest_{gesture_id}.{ext}"
    dest = _audio_dir() / audio_key
    try:
        # Clean stale baked files from a different provider so /audio never
        # serves old audio next to the fresh one.
        for stale in _audio_dir().glob(f"gest_{gesture_id}.*"):
            stale.unlink(missing_ok=True)
        voice = adaptor.voice_for_gender(gender)
        await asyncio.to_thread(adaptor.synthesize_and_save, sentence, dest, voice=voice)
    except Exception as exc:  # API/auth errors surface to the admin UI
        raise HTTPException(400, f"Speech generation failed: {exc}")
    return audio_key


def _one_shot_space_of_gesture(gesture_id: str):
    """Return the gesture's space when it belongs to a one_shot group, else None."""
    space = space_store.space_of_gesture(gesture_id)
    return space if space and space.get("join_mode") == "one_shot" else None


@app.post("/api/gestures/{gesture_id}/sentence")
async def set_gesture_sentence(gesture_id: str, req: GestureSentenceRequest):
    """(Re)generate & cache the one-shot sentence audio for a gesture."""
    gesture = store.get_by_id(gesture_id)
    if not gesture:
        raise HTTPException(404, "Gesture not found")
    if not _one_shot_space_of_gesture(gesture_id):
        raise HTTPException(400, "Spoken sentences belong to gestures inside one_shot groups")
    audio_key = await _bake_gesture_speech(gesture_id, req.sentence, req.gender)
    updated = store.set_shot(gesture_id, sentence=req.sentence, gender=req.gender,
                             audio_key=audio_key, ready=True)
    return {"success": True, "gesture": updated, "audio_url": f"/audio/{audio_key}"}


@app.get("/api/gestures/{gesture_id}/sentence/audio")
async def get_gesture_sentence_audio(gesture_id: str):
    """Serve the cached speech file (used later by the playground runtime card)."""
    gesture = store.get_by_id(gesture_id)
    shot = (gesture or {}).get("shot") or {}
    if not shot.get("audio_key"):
        raise HTTPException(404, "No speech generated for this gesture")
    path = _audio_dir() / shot["audio_key"]
    if not path.exists():
        raise HTTPException(404, "Speech file missing — regenerate it")
    from fastapi.responses import FileResponse
    return FileResponse(path, media_type="audio/mpeg", filename=shot["audio_key"])


# --- Workspaces API -----------------------------------------------------------

@app.get("/api/workspaces")
async def list_workspaces():
    return {"active": WSMAN.active(), "workspaces": WSMAN.list()}


@app.post("/api/workspaces")
async def create_workspace(req: WorkspaceCreateRequest):
    result = WSMAN.create(req.name)
    if not result.get("ok"):
        raise HTTPException(400, result.get("error", "could not create workspace"))
    return {"success": True, "active": WSMAN.active(), "workspaces": WSMAN.list()}


@app.post("/api/workspaces/switch")
async def switch_workspace(req: WorkspaceCreateRequest):
    name = (req.name or "").strip().lower()
    if not WSMAN.set_active(name):
        raise HTTPException(404, f"workspace '{name}' not found")
    # swap every store's dataset in place (singletons stay alive)
    store.reload_path(WSMAN.gestures_path())
    space_store.reload_path(WSMAN.spaces_path())
    # half-finished captures belong to the previous dataset
    active_captures.clear()
    # reset detection state: buffers/space trackers must not leak gesture
    # references from the previous dataset
    try:
        pipeline.reset_buffer()
    except NameError:
        pass
    except Exception:
        pass
    print(f"[workspace] switched → '{name}' "
          f"(gestures: {len(store.get_all())}, spaces: {len(space_store.get_all())})",
          flush=True)
    return {"success": True, "active": name, "workspaces": WSMAN.list()}


@app.get("/health")
async def health():    return {"status": "ok", "gestures_count": len(store.get_all())}


# ============================================================
# Camera relay — the minimal node capture page (or any sender)
# pushes frames here, and every admin page / test surface
# consumes them as a standard MJPEG stream.
# ============================================================
_stream_frame: Dict[str, Any] = {"jpeg": None, "ts": 0.0}   # latest pushed frame
_stream_push_times: deque = deque(maxlen=90)                # timestamps of pushes (fps)
_stream_fps: float = 0.0


@app.post("/api/stream/frame")
async def receive_stream_frame(request: Request):
    """Receiver endpoint for the capture page: JPEG frame as base64 JSON."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Body must be JSON")

    b64 = body.get("image_base64", "")
    if not b64:
        raise HTTPException(400, "image_base64 is required")
    if b64.startswith("data:"):
        b64 = b64.split(",", 1)[1]

    try:
        jpg = base64.b64decode(b64)
    except Exception:
        raise HTTPException(400, "image_base64 is not valid base64")

    # sanity: it should decode as an image
    img_arr = np.frombuffer(jpg, np.uint8)
    if cv2.imdecode(img_arr, cv2.IMREAD_COLOR) is None:
        raise HTTPException(400, "Frame is not a decodable JPEG")

    global _stream_fps
    now = time.time()
    _stream_push_times.append(now)
    if _stream_push_times and now - _stream_push_times[0] > 0:
        dt = now - _stream_push_times[0]
        _stream_fps = round((len(_stream_push_times) - 1) / dt, 1)

    _stream_frame["jpeg"] = jpg
    _stream_frame["ts"] = now
    return {"ok": True}


@app.get("/api/stream/video")
async def stream_video():
    """MJPEG stream of the latest pushed frame — consumed by <img> tags."""
    boundary = "frame"

    async def gen():
        last_ts = 0.0
        while True:
            jpg = _stream_frame["jpeg"]
            if jpg is not None and _stream_frame["ts"] != last_ts:
                last_ts = _stream_frame["ts"]
                yield (
                    b"--" + boundary.encode() + b"\r\n"
                    b"Content-Type: image/jpeg\r\nContent-Length: "
                    + str(len(jpg)).encode() + b"\r\n\r\n"
                ) + jpg + b"\r\n"
            # ~40 fps ceiling; only forwards when a new frame arrives
            await asyncio.sleep(0.025)

    return StreamingResponse(
        gen(),
        media_type=f"multipart/x-mixed-replace; boundary={boundary}",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/api/stream/status")
async def stream_status():
    """Is a sender pushing frames right now, and at what rate?"""
    fresh = _stream_frame["jpeg"] is not None and (time.time() - _stream_frame["ts"]) < 3.0
    return {
        "available": fresh,
        "fps": _stream_fps,
        "age": round(time.time() - _stream_frame["ts"], 2) if _stream_frame["jpeg"] else None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
