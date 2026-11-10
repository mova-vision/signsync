# MOVA: Motion-Oriented Versatile Automation

## 1. Overview & Dual Purpose

**MOVA** is built around two core pillars:

1. **A Modular Hardware & Peripheral Framework:** An embedded MCU architecture that abstracts physical microcontroller pins into reconfigurable logical sockets, standard drivers, adaptors, and event hooks. It allows arbitrary sensors and actuators to be plugged in, remapped, and controlled via wireless events without touching low-level pin logic.
2. **A Motion-Driven Interaction Pipeline:** A First-Person Perspective (FPP) computer-vision pipeline that normalizes hand gestures in real time, translating natural physical motions into structured action events.

---

## 2. Implementations & Use Cases

While MOVA provides a general-purpose hardware-software event architecture, its primary real-world implementation focuses on assistive tech:

* **Assistive Communication for the Mute & Speech-Impaired (Primary Implementation):** 
  A wearable, head-mounted FPP camera captures hand gestures as the user signs naturally over their chest area. The pipeline decodes the handshapes, spatial orientation, and multi-hand relationships, converting them into live speech synthesis and structured phrases via downstream language processing.
* **Touchless Hardware & Smart Device Automation:**
  Using the modular hardware layer, decoded gesture events (e.g. action commands, directional triggers) are broadcast over Wi-Fi to MCU-controlled devices—operating lights, appliances, robotics, or environment controls hands-free.
* **Plug-and-Play Embedded Prototyping:**
  The hardware socket/adaptor system serves as a standalone framework for rapid MCU prototyping, allowing developers to swap physical peripherals (I2C, GPIO, ultrasonic, LEDs, relays) across logical sockets with zero code rewrites.

---

## 2. System Architecture

For complete diagrams, space state machines, and pipeline flows, see [Architecture & Pipelines](docs/ARCHITECTURE.md).

```text
       [ Wearable FPP Camera ]
                  │ (Live Video Stream)
                  ▼
       [ Software & Vision Layer ]
       ├── MediaPipe Landmark Detection (21 points/hand)
       ├── Geometric Normalization (Wrist-centered, scale invariant)
       ├── Sign Space Scoping (Words, Spell, One-Shot)
       ├── Sliding Window Gesture Buffer & Deduplication
       └── Direct LLM Sentence Formulation + TTS Audio Stream
                  │ (Event / Action / Speech over Wi-Fi / WebSockets)
                  ▼
       [ Hardware & Audio Outputs ]
       ├── Web Audio Player (Instant Speech & LLM Audio Output)
       └── Modular MCU Hardware (Sockets, Drivers, Adaptors, Actuators)
```

---

## 3. Project Structure

```text
Mova/
├── docs/                        # Project & hardware documentation
│   ├── ARCHITECTURE.md          # Full system architecture, mode workflows & pipeline diagrams
│   └── hardware/
│       ├── DEVELOPER_GUIDE.md   # Hardware framework internal architecture
│       └── USER_GUIDE.md        # Pin socket mapping & component usage
│
├── src/                         # Production application & recognition engine
│   ├── server/                  # FastAPI web server, WebSocket feed & REST APIs
│   ├── pipeline/                # Real-time pipeline orchestrating vision & buffering
│   ├── vision/                  # MediaPipe landmark tracking & geometric scoring
│   ├── spaces/                  # Sign Space tracker & One-Shot instant speech
│   ├── buffer/                  # Gesture smoothing, outlier filtering & deduplication
│   ├── ai/                      # Direct LLM sentence synthesis & TTS speech
│   └── storage/                 # Gestures, Sign Spaces & Workspace stores
│
├── templates/                   # Frontend web UI (Dashboard, Groups, Register, Settings)
├── data/                        # JSON persistent storage (gestures, spaces, workspaces)
│
├── hardware/                    # Embedded MCU & firmware layer
│   └── simulation/              # Velxio / Wokwi hardware simulation prototype
└── assets/                      # CAD 3D models, PCB files, schematics, references
```

---

## 4. How the Key Layers Work

### A. The Hand Gesture Pipeline (`experiments/gesture_matching/`)
* **First-Person View (FPP):** Optimized for a user looking down at their own hands.
* **Why raw coordinates are NOT used:** Distance and position change as hands move. Instead, coordinates are centered on the wrist and scaled by the hand size.
* **Orientation Awareness:** Calculates a 3D Palm Normal vector so pointing forward ("You") and pointing backward ("Me") are never confused.
* **Two-Hand Support:** Evaluates each hand shape independently plus the relative direction vector between wrists (e.g. supporting gestures like "Cup & Tray" or "Namaste").
* **Top-N Output:** Returns a ranked list of scores rather than a hard single winner. This enables a downstream language model (LLM) to infer the intended sentence even if a finger is partially blocked.

### B. The Hardware Architecture (`hardware/simulation/`)
Instead of hardcoding pin numbers across the code:
* **Sockets:** MCU pins are mapped to logical sockets (`SOCKET_GPIO_1`, `SOCKET_I2C_1`).
* **Drivers:** Handle raw signal communication with physical components.
* **Adaptors:** Provide unified methods (e.g., `turnOn()`, `getDistanceCm()`).
* **Hooks:** Link events to actions cleanly without tight coupling.

---

## 5. Getting Started Quickstart

### Prerequisites
* Python 3.10+
* Virtual environment (`venv`)

### Running the MOVA Server
```bash
# 1. Activate your virtual environment
source venv/bin/activate        # On Linux / WSL
# or: win_venv\Scripts\activate # On Windows

# 2. Install dependencies
pip install -r requirements.txt # or: pip install mediapipe opencv-python fastapi uvicorn websockets

# 3. Launch the application
python -m src.server.main
# Open http://localhost:8000 in your browser
```

### Running Hardware Simulation
The firmware prototype in `hardware/simulation/` can be opened directly in [Velxio](https://velxio.dev/editor) or Wokwi using the provided `diagram.json` and `sketch.ino`.

---

## 6. Documentation References
* [Full System Architecture & Pipelines](docs/ARCHITECTURE.md)
* [Hardware Developer Guide](docs/hardware/DEVELOPER_GUIDE.md)
* [Hardware Socket & Component User Guide](docs/hardware/USER_GUIDE.md)

```
 [ Motion-Oriented ]   +   [ Versatile ]   +   [ Automation ]
            ↓                       ↓                   ↓
      The Vision / Input     The Modular HW       The Physical Action
      (Hands, Gestures,      (Sockets, Adapters,  (MCU, Devices, Output,
       Human Interaction)     Swappable Sensors)   Hardware Execution)
```