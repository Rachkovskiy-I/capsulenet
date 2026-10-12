# CapsuleNet

**A hybrid-AI simulation framework for autonomous magnetic-tape urban transit.**

CapsuleNet is an open-source reference environment for modelling the transition from human-driven cars to lightweight, tape-guided autonomous capsules operating on existing asphalt. Designed for AI researchers, RL engineers, GIS analysts, and urban planners, it allows stress-testing a post-car metropolis without new rail infrastructure. Structured as a clean single-file monolith (one file, four classes), it lets humans, RL agents, or LLMs easily analyze the entire architecture.

## Why this exists

CapsuleNet explores reusing existing asphalt via magnetic polymer-concrete trenches, passive Hall-effect sensors, local V2V radio for platooning, and a central orchestrator managing city-wide demand. Strategic intelligence resides in the data center, while immediate safety reflexes operate on the edge.

## Core Concepts & Layers

- **Infrastructure**: Directed road network graph with waypoint-densified edges and finite battery hubs (NetworkX, FIFO Queues).
- **Physical Layer (Capsule)**: Actuator with tape following via P+I controller and local V2V emergency braking.
- **Intelligence Layer**: Strategic authority handling demand scheduling, fleet sizing, shortest-path routing, and load-balanced swap allocation (`CentralAI`, Dijkstra/A*).

## The Four Hard Problems This Code Solves

1. **Finite NetworkX Infrastructure**: Runs on non-trivial synthetic urban grids with pre-calculated sub-pixel coordinate waypoints.
2. **Fleet Scaling & Physical Depots**: Spawns at least-loaded battery swap depots and demotes surplus vehicles to `IDLE` when demand drops.
3. **Local V2V Emergency Braking**: Forces braking independently of central commands if safety gaps are breached.
4. **Load-Balanced Battery Swapping**: Dynamically intercepts low-battery paths to minimize distance and queue load factors.

## Installation & Controls

```bash
git clone https://github.com/Rachkovskiy-I/capsulenet
cd capsulenet
pip install -r requirements.txt
python main.py
```

*Requirements include `pygame>=2.5.0`, `networkx>=3.2`, and `numpy>=1.26.0`.*

Use `SPACE` to toggle city-wide emergency stop and `ESC` to quit. Status references include `OPERATIONAL`, `IDLE`, `WAITING`, `SWAPPING`, `BROKEN`, and `EMERGENCY_STOP`.

## Roadmap & License

Planned features include wrapping `CentralAI` into a `gymnasium.Env`, multi-station vehicle routing with the Hungarian algorithm, V2V braking-vector propagation, and mixed traffic parameters. Licensed under the MIT License.
