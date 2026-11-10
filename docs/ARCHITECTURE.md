# MOVA System Architecture & Pipeline Documentation

A concise visual and architectural reference for the MOVA gesture recognition, sign space tracking, and speech pipeline.

---

## 1. Full System End-to-End Pipeline

### High-Level Architecture (Abstract Flow)

```
 [ Webcam Video Feed ] (6-30 FPS)
         │
         ▼
 ┌────────────────────────────────────────┐
 │ 1. Vision & Landmark Extraction        │  MediaPipe Hand Tracking (1 or 2 hands)
 └───────────────────┬────────────────────┘
                     │ Normalized Landmarks / Geometric Features
                     ▼
 ┌────────────────────────────────────────┐
 │ 2. Sign Space Candidate Scoping        │  SpaceTracker isolates active gestures
 └───────────────────┬────────────────────┘  (eliminates group collision)
                     │ Scoped Candidate Subset
                     ▼
 ┌────────────────────────────────────────┐
 │ 3. Cosine & Finger Profile Matcher     │  score_gestures() checks against threshold
 └───────────────────┬────────────────────┘
                     │ Top Detected Gesture
                     ▼
 ┌────────────────────────────────────────┐
 │ 4. Sliding Window & Gesture Buffer     │  Outlier removal, persistence validation,
 └───────────────────┬────────────────────┘  deduplication
                     │ Confirmed Event ("accepted" / "completed")
                     ▼
 ┌────────────────────────────────────────┐
 │ 5. Space Routing & Dispatch            │  Standard, Words, Spell, or One-Shot
 └─────────┬───────────────────┬──────────┘
           │                   │
  [ Instant Speech ]   [ Word / Spell Queue ]
           │                   │
           ▼                   ▼
     Audio Player      AI Sentence Formulator (Direct LLM)
                               │
                               ▼
                    TTS Voice Output (Piper / eSpeak / ElevenLabs)
```

---

## 2. Core Internal Stages

### 2.1 Vision & Feature Extraction (`src/vision/mediapipe_processor.py`)
* Extracts 21 3D landmarks per hand.
* Single Hand (`71` features):
  * Normalized relative coords (63 floats)
  * Shape angle & finger extension profile (5 values: thumb, index, middle, ring, pinky)
  * Inter-finger tip gap metrics (thumb-index, index-middle, etc.)
* Two Hands (`145` features):
  * Left & Right hand profiles + relative cross-hand distance & posture.

### 2.2 Scoped Candidate Filtering (`src/spaces/space_tracker.py`)
* **Eliminates False Matches**: A letter or group sign never competes with normal standard signs.
* **Scope States**:
  * **Standard (No Group Active)**: Normal gestures + Space entry/toggle signs.
  * **Space Active**: Only that space's member gestures + its control signs (toggle/exit).
  * **Post-Close Cooldown**: Only control signs scoreable during quiet transition grace.

### 2.3 Gesture Buffer & Smoothing (`src/buffer/gesture_buffer.py`)
```
Frame N-2   Frame N-1   Frame N (Center)   Frame N+1   Frame N+2
  [A]         [A]            [B]              [A]         [A]
                               │
                       [ Outlier Check ]
                     Isolated spike filtered!
                               │
                     [ Persistence Check ]
                Needs >= min_persistence (3/5)
                               │
                     [ Deduplication Check ]
                  candidate == last_accepted?
                 ├── Yes: Drop (no re-trigger)
                 └── No:  Emit "accepted" event
```

---

## 3. Operational Modes & Workflows

### 3.1 Standard Mode (No Group Active)
* **Purpose**: General conversational gestures.
* **Lifecycle**:
```
User Signs Gesture A ──► Accepted into Queue ──► Buffer collects phrase
                                                        │
                      Hand moves away (Inactivity) ─────┘
                               │
                               ▼
                       Buffer "completed"
                               │
                               ▼
                    AI Formulates Sentence ──► TTS Audio
```

---

### 3.2 Words Group Mode (`join_mode: "words"`)
* **Purpose**: Namespaced topic vocabulary (e.g., Colors, Food, Family).
* **Behavior**: Each accepted gesture immediately appends as an isolated word token.
* **Auto-Close Variants**:
  * **Auto-Close OFF**: Group stays active; switch/exit via `space-toggle` or `space-exit`.
  * **Auto-Close ON (Move Away `hand_off`)**: Hand leaves screen (`hand_count == 0`) ──► Group closes back to Standard.
  * **Auto-Close ON (Keep on Screen `quiet`)**: Sign accepted ──► Group closes immediately ──► 150ms cooldown prevents lookalike misfires.

```
       [ Enter Group (Toggle Sign) ]
                     │
                     ▼
             [ Active Group ]
                     │
         User signs member (e.g. "RED")
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[ Auto-Close OFF ]          [ Auto-Close ON ]
Stays in group.                    │
Sign toggle to exit.               ├──► hand_off: Closes when hand leaves screen
                                   └──► quiet: Closes instantly + 150ms lockout
```

---

### 3.3 Spell Group Mode (`join_mode: "spell"`)
* **Purpose**: Fingerspelling words letter-by-letter (e.g. `B` + `L` + `U` + `E` ──► `BLUE`).
* **Behavior**:
  * Signs chain together into a single token string.
  * Auto-close is **always disabled** (spelling requires continuous chaining).
  * Exits on `space-toggle`, `space-exit`, or inactivity timeout.

```
 [ Letter A ] ──► [ Letter B ] ──► [ Letter C ]
                         │
        Toggle Sign / Inactivity Flush
                         │
                         ▼
             Token Emitted: "ABC" (Single Word)
```

---

### 3.4 One-Shot Speech Mode (`join_mode: "one_shot"`)
* **Purpose**: Instant pre-recorded / pre-synthesized sentence speech with zero latency.
* **Behavior**: First matching sign bypasses queue and AI formulation, playing audio immediately.

```
 User signs One-Shot Sign (e.g. "NEED_HELP")
                     │
                     ▼
         Audio Plays Instantly (0 ms AI delay)
                     │
       ┌─────────────┴─────────────┐
       ▼                           ▼
[ Auto-Close ON ]           [ Auto-Close OFF ]
Closes space immediately:   Group stays open:
• hand_off: on hand away    • Held gesture does NOT repeat in a loop
• quiet: instant + grace    • Switch to new sign to speak next sentence
```

---

## 4. Transition & Deduplication Mechanics

### Holding Gesture vs Switching Flow
```
 Scenario A: Holding the exact same sign
 [ Sign A Accepted ] ──► User holds pose ──► Deduplication blocks repeat (stays quiet)

 Scenario B: Switching signs
 [ Sign A Accepted ] ──► User signs Sign B ──► Flow broken ──► Sign B accepted immediately

 Scenario C: Returning to Sign A
 [ Sign A ] ──► [ Sign B ] ──► [ Sign A ]
                                    │
                                    └──► Accepted immediately (flow was broken)
```

### Auto-Close Comparison: `hand_off` vs `quiet`

| Feature | Move Away (`hand_off`) | Keep On Screen (`quiet`) |
| :--- | :--- | :--- |
| **Trigger** | Camera detects 0 hands | Gesture confirmed / accepted |
| **Hand Action** | Move hand out of camera frame | Keep hand resting / visible |
| **Closing Time** | When hand leaves frame | Immediate upon recognition |
| **Safety Guard** | Scoped candidates held until hand-away | 150ms lockout of standard normal gestures |
| **Best For** | Deliberate isolated sentences | Fast rapid-fire sequencing between groups |

---

## 5. AI Sentence Pipeline & TTS (`src/ai/`)

* **Direct LLM Architecture**: MiniLM reranker and decision model were completely removed. The pipeline goes directly from ordered readings into one LLM call, reducing latency.
* **Fast-Path**: If a sequence has only one possible unambiguous reading, it skips the network call entirely.

```
 [ Raw Accepted Sequence ]  (e.g., ["store", "go", "i", "milk"])
            │
            ▼
 ┌────────────────────────────────────────┐
 │ 1. Tag Ordering & Meanings Mapper      │  Pins spell words, maps multi-meaning tags
 └───────────────────┬────────────────────┘
                     │ Ordered Readings
                     ▼
 ┌────────────────────────────────────────┐
 │ 2. Direct LLM Call (`llm_sentence.py`) │  Single prompt to OpenAI-compatible endpoint
 └───────────────────┬────────────────────┘  Produces natural sentence: "I am going to the store for milk."
                     │
                     ▼
 ┌────────────────────────────────────────┐
 │ 3. TTS Voice Synthesis (Text-to-Speech)│  Piper / eSpeak / ElevenLabs
 └───────────────────┬────────────────────┘  Generates speech audio wav/mp3
                     │
                     ▼
      Client Audio Stream (WebSocket) ──► Instant Browser Audio Playback
```

---

## 6. Project Directory Map

* `src/vision/` — MediaPipe landmark processor, feature extraction, cosine scoring.
* `src/spaces/` — Sign space tracker, candidate scoping, one-shot instant speech.
* `src/buffer/` — Outlier filtering, persistence sliding window, deduplication.
* `src/pipeline/` — Central `RealTimePipeline` orchestrating vision, spaces, and buffer.
* `src/ai/` — Direct LLM sentence synthesis, meanings mapper, tag ordering, and TTS integration.
* `src/storage/` — JSON stores for gestures, sign spaces, and workspaces.
* `src/server/` — FastAPI backend with WebSocket stream and REST admin APIs.
* `templates/` — Web UI for main real-time dashboard, group management, and gesture registration.
