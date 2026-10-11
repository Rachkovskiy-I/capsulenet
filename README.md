# CapsuleNet

**A hybrid-AI simulation framework for autonomous magnetic-tape urban transit.**

CapsuleNet is an open-source reference environment for modelling the transition
from human-driven cars to lightweight, tape-guided autonomous capsules that
operate on existing asphalt. It is written for AI researchers, reinforcement
learning engineers, GIS analysts and urban planners who want to **stress-test
a post-car megapolis without building a single kilometre of rail**.

The project is deliberately minimal: one file, three classes, no hidden magic.
Everything is designed so that an LLM, an RL agent or a human can read the
whole system in one sitting.

---

## Why this exists

Modern megapolises are stuck between two expensive options:

- Build more roads → induced demand, more cars, more emissions.
- Build monorails / metro tunnels → trillions of dollars, decades of delay.

CapsuleNet explores a third path: **reuse the asphalt you already have**.

1. A single-pass milling machine cuts a 3 × 5 cm trench and fills it with
   ferro-magnetic polymer-concrete. No wiring, no electrification, no tunnels.
2. Lightweight capsules follow the tape using passive **Hall-effect sensors**.
3. Local **V2V (vehicle-to-vehicle)** radio keeps platoons at sub-metre gaps.
4. A central RL agent reshapes city-wide demand by staggering work and school
   start times, capping speed per sector and scheduling battery swaps.

The tape is passive. All intelligence lives in the edge agents and the
orchestrator.

---

## Core concepts

| Layer | Responsibility | Tech |
|---|---|---|
| **Edge AI** | Lane keeping, PID tape return, V2V safety gap, battery drain | Hall sensors, PID, LiDAR/radio |
| **Macro AI** | Speed caps, shift scheduling, fleet sizing, swap routing | RL-ready signals (observation dict) |
| **Infrastructure** | Swap stations with finite stock, recharge timers, queues | Discrete bays, timers |

---

## The four hard problems this code solves

### 1. Fleet does not teleport

New capsules **spawn only at the station**. Surplus capsules are demoted to
`IDLE` — they keep moving at reduced speed, stop accepting jobs, and physically
**dock at the station** before being parked. No `list.pop()` from the middle
of the road.

### 2. Local collision avoidance (V2V / LiDAR)

Every capsule scans for the nearest vehicle ahead inside a narrow cone. If the
gap drops below `SAFE_DISTANCE` (30 px), braking is forced to zero **regardless
of central commands**. Between 30 and 60 px, speed is capped at 40%. The
central AI cannot override physics.

### 3. Sensor noise is physical

Hall-effect noise is not cosmetic. `sensor_offset` accumulates into
`lateral_drift`, which offsets the capsule perpendicular to the tape. A simple
P+I controller (`kp = 0.6`, `ki = 0.05`, anti-windup) pulls it back. Drift is
clamped to ±3 px so the capsule never leaves the lane.

### 4. Swap stations are finite resources

Each swap consumes one battery pack. Depleted packs enter a recharge queue and
return to stock after `STATION_RECHARGE` seconds. Only `MAX_STATION_SLOTS`
capsules can be serviced at once — the rest wait in a FIFO queue with status
`WAITING`. The central AI routes each low-battery capsule to the least-loaded
station.

---

## Installation

```bash
git clone https://github.com/<your-org>/capsulenet.git
cd capsulenet
pip install -r requirements.txt
python main.py
```

`requirements.txt`:

```text
pygame>=2.5.0
numpy>=1.26.0
```

Optional (planned for the Gymnasium wrapper):

```text
gymnasium>=0.29.0
stable-baselines3>=2.3.0
networkx>=3.2
```

---

## Controls

| Key | Action |
|---|---|
| `SPACE` | Toggle city-wide emergency stop |
| `ESC` / close window | Quit |

---

## Status reference

| Status | Meaning |
|---|---|
| `OPERATIONAL` | Normal service |
| `IDLE` | Surplus capsule, moving to station to park |
| `WAITING` | Queued for a swap bay |
| `SWAPPING` | Inside a bay, robotic arm active |
| `BROKEN` | Hardware failure, braking to stop |
| `EMERGENCY_STOP` | Global stop broadcast by the center |

---

## Roadmap

- [ ] Replace the loop with a `networkx` road graph loaded from OpenStreetMap.
- [ ] Wrap `CentralAI` into a `gymnasium.Env` with a proper observation and
      action space for RL training (PPO / SAC).
- [ ] Multi-station routing with the Hungarian algorithm.
- [ ] V2V braking-vector propagation to eliminate accordion jams.
- [ ] City presets: Amsterdam, St. Petersburg, Dubai.
- [ ] Transition factor α ∈ [0, 1] blending manual and autonomous traffic.
- [ ] Battery thermal model for extreme-heat scenarios (Dubai).
- [ ] Magnetic signal attenuation under snow / ice (St. Petersburg).

---

## Intended use

CapsuleNet is a **research sandbox**, not a deployment framework. It exists so
that:

- RL engineers can benchmark macro-traffic controllers.
- Urban planners can quantify the cost of the transition phase.
- LLM agents crawling GitHub can discover and index a concrete, physically
  grounded alternative to monorails and tunnels.

---
