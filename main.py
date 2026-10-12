"""
CapsuleNet — Hybrid AI Simulation Framework for Autonomous Magnetic-Tape
Urban Transit.

Architecture
------------
*   Infrastructure layer: a NetworkX-directed road graph with waypoint-based
    edges (sub-pixel path resolution) and multiple BatterySwapStation hubs.
*   Physical layer: Capsule is a *dumb actuator*. It has no onboard computer,
    no route planning, no strategic logic. Its only local reflexes are a
    P+I lane-keeping controller reacting to magnetic sensor noise and an
    immediate V2V emergency-braking response. Physics cannot wait for a
    network round-trip, so these reflexes live on the capsule.
*   Intelligence layer: CentralAI is the single source of strategic truth.
    It performs fleet sizing, demand scheduling, speed-limit broadcasts,
    route generation (networkx.shortest_path) and dynamic swap-station
    rerouting.

Dependencies: pygame, networkx, numpy, random, math (stdlib).
No non-English text anywhere.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import pygame


# ===========================================================================
# CONFIGURATION CONSTANTS
# ===========================================================================

WIDTH, HEIGHT = 1200, 800
FPS = 60

# Palette
COLOR_BG         = (14, 14, 20)
COLOR_TAPE       = (70, 82, 120)
COLOR_TAPE_IDLE  = (44, 50, 70)
COLOR_NODE       = (180, 180, 200)
COLOR_STATION    = (255, 184, 108)
COLOR_CAPSULE    = (80, 250, 123)
COLOR_IDLE       = (110, 110, 140)
COLOR_WAITING    = (241, 250, 140)
COLOR_SWAPPING   = (255, 184, 108)
COLOR_BROKEN     = (255, 85, 85)
COLOR_HUD        = (220, 220, 220)
COLOR_HUD_ALERT  = (255, 100, 100)

# Physics
SAFE_DISTANCE       = 30.0
WARN_DISTANCE       = 60.0
BATTERY_LOW         = 30.0
BATTERY_CRITICAL    = 15.0
SWAP_TIME           = 1.5
STATION_RECHARGE    = 6.0
MAX_STATION_SLOTS   = 2
IDLE_SPEED_CAP      = 2.5
WAYPOINT_STEP       = 6.0     # px between interpolated waypoints along an edge

# PID
PID_KP              = 0.6
PID_KI              = 0.05
PID_CLAMP           = 1.0
LATERAL_DRIFT_MAX   = 3.0

# Scheduling
SCHEDULING_MODE     = "DECENTRALIZED"   # or "CLASSIC"


# ===========================================================================
# LAYER 1 — INFRASTRUCTURE
# ===========================================================================

@dataclass
class EdgeTape:
    """
    A single directed road segment carrying an embedded magnetic tape.

    The tape is represented as a dense sequence of 2D waypoints so capsules
    can traverse the edge smoothly rather than teleporting between nodes.
    """
    u: int
    v: int
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    length: float = 0.0
    is_idle_corridor: bool = False   # reserved for future dead-end parking


class BatterySwapStation:
    """
    Station hub attached to a specific graph node.

    Stations own a finite battery pack stock and a per-pack recharge queue.
    Strategic routing to a station is handled exclusively by CentralAI.
    """

    def __init__(self, node_id: int, xy: Tuple[float, float], capacity: int = 10):
        self.node_id = node_id
        self.x, self.y = xy
        self.capacity = capacity
        self.available_batteries = capacity
        self.recharging: List[float] = []
        self.queue: List["Capsule"] = []
        self.active_slots = 0
        # Swap registry: capsules currently inside a swap bay.
        self.swapping: List["Capsule"] = []

    def update(self, dt: float) -> None:
        """Advance recharge timers and promote ready packs back into stock."""
        still = []
        for timer in self.recharging:
            timer -= dt
            if timer <= 0:
                if self.available_batteries < self.capacity:
                    self.available_batteries += 1
            else:
                still.append(timer)
        self.recharging = still

    def request_swap(self, capsule: "Capsule") -> bool:
        if capsule in self.queue or capsule in self.swapping:
            return False
        self.queue.append(capsule)
        return True

    def try_serve(self, dt: float) -> List["Capsule"]:
        """Admit new capsules to bays and advance in-progress swaps."""
        served: List["Capsule"] = []

        while (self.active_slots < MAX_STATION_SLOTS
               and self.queue
               and self.available_batteries > 0):
            cap = self.queue.pop(0)
            self.available_batteries -= 1
            self.recharging.append(STATION_RECHARGE)
            self.active_slots += 1
            cap.swap_timer = SWAP_TIME
            cap.status = "SWAPPING"
            self.swapping.append(cap)

        done = []
        for cap in self.swapping:
            cap.swap_timer -= dt
            if cap.swap_timer <= 0:
                cap.battery = 100.0
                cap.needs_swap = False
                cap.status = "OPERATIONAL"
                cap.swap_timer = 0.0
                self.active_slots = max(0, self.active_slots - 1)
                served.append(cap)
                done.append(cap)
        for cap in done:
            self.swapping.remove(cap)
        return served

    def load_factor(self) -> float:
        return (len(self.queue) + self.active_slots * 1.5
                - 0.5 * self.available_batteries)


class UrbanGraph:
    """
    Directed NetworkX road graph with waypoint-densified edges.

    Node positions are 2D pixel coordinates. Every edge is pre-processed into
    an EdgeTape carrying a dense waypoint list, length and unit-segment cache.
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.edge_tapes: dict[Tuple[int, int], EdgeTape] = {}
        self.stations: dict[int, BatterySwapStation] = {}
        self._build_synthetic_city()

    # ------------------------------------------------------------------ build

    def _build_synthetic_city(self) -> None:
        """
        Construct a synthetic but non-trivial urban grid:
          *  6 columns × 4 rows of intersections
          *  Random missing links (dead ends, forks, one-ways)
          *  3 swap stations placed at distinct junctions
        """
        rng = random.Random(20240101)

        cols, rows = 6, 4
        margin_x, margin_y = 120, 120
        step_x = (WIDTH - 2 * margin_x) / (cols - 1)
        step_y = (HEIGHT - 2 * margin_y) / (rows - 1)

        node_id = 0
        grid: dict[Tuple[int, int], int] = {}
        for r in range(rows):
            for c in range(cols):
                x = margin_x + c * step_x
                y = margin_y + r * step_y
                # Small deterministic jitter for organic feel
                x += rng.uniform(-18, 18)
                y += rng.uniform(-18, 18)
                self.graph.add_node(node_id, x=x, y=y)
                grid[(r, c)] = node_id
                node_id += 1

        # Connect horizontally and vertically with some missing links
        for r in range(rows):
            for c in range(cols - 1):
                if rng.random() < 0.9:
                    u, v = grid[(r, c)], grid[(r, c + 1)]
                    self._add_edge_pair(u, v, bidirectional=True)
        for r in range(rows - 1):
            for c in range(cols):
                if rng.random() < 0.85:
                    u, v = grid[(r, c)], grid[(r + 1, c)]
                    self._add_edge_pair(u, v, bidirectional=True)

        # Add two long diagonal "express" corridors
        self._add_edge_pair(grid[(0, 0)], grid[(rows - 1, cols - 1)], bidirectional=False)
        self._add_edge_pair(grid[(0, cols - 1)], grid[(rows - 1, 0)], bidirectional=False)

        # Ensure strong connectivity by adding a fallback ring if needed
        if not nx.is_weakly_connected(self.graph):
            nodes = list(self.graph.nodes)
            for i in range(len(nodes)):
                u, v = nodes[i], nodes[(i + 1) % len(nodes)]
                self._add_edge_pair(u, v, bidirectional=True)

        # ---- Stations at three distinct junctions ----
        station_nodes = [
            grid[(0, 0)],
            grid[(rows - 1, cols - 1)],
            grid[(rows // 2, cols // 2)],
        ]
        for nid in station_nodes:
            xy = (self.graph.nodes[nid]["x"], self.graph.nodes[nid]["y"])
            self.stations[nid] = BatterySwapStation(nid, xy, capacity=10)

        # Precompute edge tapes (waypoints) for all edges
        self._build_edge_tapes()

    def _add_edge_pair(self, u: int, v: int, bidirectional: bool) -> None:
        if self.graph.has_edge(u, v):
            return
        self.graph.add_edge(u, v)
        if bidirectional and not self.graph.has_edge(v, u):
            self.graph.add_edge(v, u)

    def _build_edge_tapes(self) -> None:
        """Populate `edge_tapes` with dense waypoint lists per directed edge."""
        for u, v in self.graph.edges:
            xu, yu = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
            xv, yv = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
            dist = math.hypot(xv - xu, yv - yu)
            n_steps = max(2, int(dist / WAYPOINT_STEP))
            pts = [
                (xu + (xv - xu) * t / n_steps,
                 yu + (yv - yu) * t / n_steps)
                for t in range(n_steps + 1)
            ]
            self.edge_tapes[(u, v)] = EdgeTape(u, v, pts, dist)

    # ----------------------------------------------------------------- helpers

    def node_xy(self, nid: int) -> Tuple[float, float]:
        n = self.graph.nodes[nid]
        return n["x"], n["y"]

    def nearest_station_node(self, x: float, y: float) -> Optional[int]:
        if not self.stations:
            return None
        return min(
            self.stations.keys(),
            key=lambda nid: (self.node_xy(nid)[0] - x) ** 2
                            + (self.node_xy(nid)[1] - y) ** 2,
        )

    def all_station_nodes(self) -> List[int]:
        return list(self.stations.keys())


# ===========================================================================
# LAYER 2 — PHYSICAL (Capsule = dumb actuator)
# ===========================================================================

class Capsule:
    """
    A dumb physical actuator following a tape.

    The capsule receives a *route* (list of node ids) from CentralAI and
    traverses it. It performs only two local computations:
      *  P+I lane-keeping against magnetic sensor noise.
      *  V2V emergency braking against the closest capsule ahead on the
         same edge.

    It owns no route planning, no fleet logic, no traffic cognition.
    """

    __slots__ = (
        "id", "x", "y", "speed", "max_speed", "battery",
        "battery_consumption_rate", "route", "route_step",
        "current_edge", "waypoint_idx", "lateral_drift", "pid_integral",
        "sensor_offset", "is_broken", "needs_swap", "swap_timer",
        "status", "leading_gap", "reserved_station", "spawn_edge",
    )

    OPERATIONAL    = "OPERATIONAL"
    IDLE           = "IDLE"
    WAITING        = "WAITING"
    SWAPPING       = "SWAPPING"
    BROKEN         = "BROKEN"
    EMERGENCY_STOP = "EMERGENCY_STOP"

    def __init__(self, cid: int, x: float, y: float,
                 route: List[int], spawn_edge: Tuple[int, int]):
        self.id = cid
        self.x, self.y = x, y
        self.speed = 0.5
        self.max_speed = 5.0
        self.battery = 100.0
        self.battery_consumption_rate = 0.05

        # Route assigned by CentralAI: sequence of node ids. The capsule
        # follows edge (route[i], route[i+1]).
        self.route = route
        self.route_step = 0
        self.current_edge: Optional[Tuple[int, int]] = spawn_edge
        self.waypoint_idx = 0
        self.spawn_edge = spawn_edge

        # Local reflex state
        self.lateral_drift = 0.0
        self.pid_integral = 0.0
        self.sensor_offset = 0.0
        self.leading_gap = float("inf")

        # Status / fleet flags
        self.is_broken = False
        self.needs_swap = False
        self.swap_timer = 0.0
        self.status = self.OPERATIONAL
        self.reserved_station: Optional[int] = None

    # ------------------------------------------------------------- local PID

    def _pid_lane_keeping(self) -> float:
        """P+I controller returning a lateral correction in [-1, 1]."""
        self.pid_integral += self.sensor_offset
        self.pid_integral = max(-PID_CLAMP, min(PID_CLAMP, self.pid_integral))
        return -(PID_KP * self.sensor_offset + PID_KI * self.pid_integral)

    # ------------------------------------------------------------ V2V reflex

    def _compute_leading_gap(self, all_capsules: List["Capsule"]) -> float:
        """
        Scan for the closest capsule ahead on the same *current edge*.
        Distance is measured in pixel space; only capsules within a lateral
        corridor of ±20 px from our heading are considered.
        """
        if self.current_edge is None:
            return float("inf")

        # Forward direction of the current edge
        pts = CAPSULE_WORLD.edge_tapes[self.current_edge].waypoints
        if self.waypoint_idx >= len(pts) - 1:
            tx, ty = pts[-1]
        else:
            tx, ty = pts[self.waypoint_idx + 1]
        fx, fy = tx - self.x, ty - self.y
        flen = math.hypot(fx, fy) or 1.0
        fx, fy = fx / flen, fy / flen

        min_gap = float("inf")
        for other in all_capsules:
            if other is self or other.current_edge != self.current_edge:
                continue
            dx, dy = other.x - self.x, other.y - self.y
            forward = dx * fx + dy * fy
            if forward <= 0:
                continue
            lateral = abs(dx * (-fy) + dy * fx)
            if lateral > 20:
                continue
            dist = math.hypot(dx, dy)
            if dist < min_gap:
                min_gap = dist
        return min_gap

    # ------------------------------------------------------------ main update

    def update(self,
               central_signals: dict,
               emergency: bool,
               all_capsules: List["Capsule"],
               graph: UrbanGraph,
               dt: float = 1.0) -> None:
        # 0. Parked states
        if self.status in (self.SWAPPING, self.WAITING):
            self.speed = 0.0
            return

        # 1. Emergency / hardware failure
        if self.is_broken or emergency:
            self.speed = max(0.0, self.speed - 0.2)
            self.status = self.BROKEN if self.is_broken else self.EMERGENCY_STOP
            return

        # 2. Target speed from CentralAI
        target_speed = central_signals.get("recommended_speed", self.max_speed)
        if self.status == self.IDLE:
            target_speed = min(target_speed, IDLE_SPEED_CAP)
        elif self.needs_swap:
            target_speed = min(target_speed, self.max_speed * 0.6)

        # 3. Battery drain
        if self.status == self.OPERATIONAL:
            self.battery -= self.battery_consumption_rate * (
                self.speed / self.max_speed + 0.5)
            if self.battery < BATTERY_LOW:
                self.needs_swap = True

        # 4. V2V emergency braking reflex (does not wait for CentralAI)
        self.leading_gap = self._compute_leading_gap(all_capsules)
        if self.leading_gap < SAFE_DISTANCE:
            target_speed = 0.0
        elif self.leading_gap < WARN_DISTANCE:
            target_speed = min(target_speed, self.max_speed * 0.4)

        # 5. Smooth accel / decel
        if self.speed < target_speed:
            self.speed = min(self.speed + 0.1, target_speed)
        elif self.speed > target_speed:
            self.speed = max(self.speed - 0.2, target_speed)

        # 6. Motion along edge waypoints
        if self.current_edge is None or self.speed <= 0:
            return

        tape = graph.edge_tapes[self.current_edge]
        pts = tape.waypoints

        remaining = self.speed
        while remaining > 0 and self.waypoint_idx < len(pts) - 1:
            tx, ty = pts[self.waypoint_idx + 1]
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)

            if dist <= remaining:
                self.x, self.y = tx, ty
                remaining -= dist
                self.waypoint_idx += 1
            else:
                nx_, ny_ = dx / dist, dy / dist
                self.x += nx_ * remaining
                self.y += ny_ * remaining

                # Sensor noise + PID lateral drift
                self.sensor_offset = random.uniform(-0.1, 0.1)
                self.lateral_drift += self.sensor_offset * 0.5
                self.lateral_drift += self._pid_lane_keeping()
                self.lateral_drift = max(-LATERAL_DRIFT_MAX,
                                         min(LATERAL_DRIFT_MAX, self.lateral_drift))

                perp_x, perp_y = -ny_, nx_
                self.x += perp_x * self.lateral_drift * 0.15
                self.y += perp_y * self.lateral_drift * 0.15
                remaining = 0.0

        # 7. Reached end of current edge -> advance to next route segment
        if self.waypoint_idx >= len(pts) - 1 and self.route_step + 1 < len(self.route):
            self.route_step += 1
            u, v = self.route[self.route_step], self.route[self.route_step + 1]
            if graph.graph.has_edge(u, v):
                self.current_edge = (u, v)
                self.waypoint_idx = 0

    # ---------------------------------------------------------------- render

    def draw(self, screen: pygame.Surface, highlight: bool = False) -> None:
        if self.status in (self.BROKEN, self.EMERGENCY_STOP):
            color = COLOR_BROKEN
        elif self.status == self.IDLE:
            color = COLOR_IDLE
        elif self.status == self.WAITING:
            color = COLOR_WAITING
        elif self.status == self.SWAPPING:
            color = COLOR_SWAPPING
        elif self.needs_swap:
            color = (241, 250, 140)
        else:
            color = COLOR_CAPSULE

        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 5)

        if highlight:
            font = pygame.font.SysFont("consolas", 16)
            tag = font.render(
                f"#{self.id} {self.status} Bat:{int(self.battery)}%",
                True, (255, 255, 255)
            )
            screen.blit(tag, (self.x - 60, self.y - 24))


# Global reference so Capsule can read the road graph without a long
# parameter chain. Set once in main().
CAPSULE_WORLD: UrbanGraph = None  # type: ignore


# ===========================================================================
# LAYER 3 — INTELLIGENCE (CentralAI)
# ===========================================================================

class CentralAI:
    """
    Sole strategic authority.

    Owns:
      *  Demand schedule (CLASSIC vs DECENTRALIZED).
      *  Fleet sizing and depot spawning.
      *  Route generation via networkx.shortest_path.
      *  Dynamic rerouting to swap stations based on load factor.
      *  Global speed-limit broadcast.
    """

    def __init__(self, graph: UrbanGraph):
        self.graph = graph
        self.time_of_day = 6.0
        self.demand = 0.2
        self._next_capsule_id = 0
        self._spawn_cooldown = 0.0

    # ------------------------------------------------------ demand schedule

    def update_schedule(self, mode: str, dt_hours: float) -> float:
        self.time_of_day = (self.time_of_day + dt_hours) % 24.0
        h = self.time_of_day
        if mode == "CLASSIC":
            self.demand = 1.0 if 8.0 <= h <= 9.0 else 0.1
        else:  # DECENTRALIZED
            if   7.0 <= h <= 8.0:   self.demand = 0.4
            elif 8.5 <= h <= 9.5:   self.demand = 0.5
            elif 10.0 <= h <= 11.0: self.demand = 0.3
            else:                   self.demand = 0.15
        return self.demand

    def fleet_target(self) -> int:
        return max(2, int(self.demand * 30))

    # ---------------------------------------------------- route generation

    def plan_route(self, src_node: int, dst_node: int) -> Optional[List[int]]:
        """Strategic routing: uses NetworkX shortest_path on the road graph."""
        try:
            return nx.shortest_path(self.graph.graph, src_node, dst_node)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def plan_route_from_edge(self, edge: Tuple[int, int],
                             dst_node: int) -> Optional[List[int]]:
        """Plan a route beginning at the endpoint of the capsule's edge."""
        return self.plan_route(edge[1], dst_node)

    def choose_random_destination(self) -> int:
        nodes = list(self.graph.graph.nodes)
        return random.choice(nodes)

    def pick_spawn_station(self) -> int:
        """Choose the least-loaded station as spawn depot."""
        stations = self.graph.stations
        return min(stations.keys(),
                   key=lambda nid: stations[nid].load_factor())

    def pick_swap_station(self, capsule: Capsule) -> Optional[int]:
        """Pick the least-loaded station, considering distance."""
        if not self.graph.stations:
            return None
        best_nid, best_score = None, float("inf")
        for nid, st in self.graph.stations.items():
            try:
                path = nx.shortest_path(self.graph.graph, capsule.current_edge[1], nid)
                dist = len(path)
            except (nx.NetworkXNoPath, nx.NodeNotFound, TypeError):
                dist = 1000
            score = st.load_factor() + 0.05 * dist
            if score < best_score:
                best_score, best_nid = score, nid
        return best_nid

    def assign_swap(self, capsule: Capsule) -> None:
        """Dynamically reroute a capsule to a swap station."""
        if capsule.reserved_station is not None:
            return
        nid = self.pick_swap_station(capsule)
        if nid is None or capsule.current_edge is None:
            return
        new_route = self.plan_route_from_edge(capsule.current_edge, nid)
        if not new_route:
            return
        # Preserve current edge, replace the tail of the route
        capsule.route = [capsule.current_edge[0]] + new_route
        capsule.route_step = 0
        capsule.reserved_station = nid
        self.graph.stations[nid].request_swap(capsule)

    # -------------------------------------------------------- fleet control

    def maybe_spawn(self, capsules: List[Capsule]) -> Optional[Capsule]:
        """Spawn a new capsule at the least-loaded station depot."""
        if self._spawn_cooldown > 0:
            return None
        nid = self.pick_spawn_station()
        st = self.graph.stations[nid]
        dst = self.choose_random_destination()
        route = self.plan_route(nid, dst)
        if not route or len(route) < 2:
            return None
        spawn_edge = (route[0], route[1])
        x, y = self.graph.node_xy(nid)
        cap = Capsule(self._next_capsule_id, x, y, route, spawn_edge)
        self._next_capsule_id += 1
        self._spawn_cooldown = 0.25
        return cap

    def tick_spawn_cooldown(self, dt: float) -> None:
        self._spawn_cooldown = max(0.0, self._spawn_cooldown - dt)

    def recommend_speed(self, active_count: int) -> float:
        return 4.5 if active_count <= 15 else 2.5


# ===========================================================================
# LAYER 4 — VISUALIZER + MAIN LOOP
# ===========================================================================

def draw_graph(screen: pygame.Surface, graph: UrbanGraph) -> None:
    # Edges as thick tape lines
    for (u, v), tape in graph.edge_tapes.items():
        pts = [(int(x), int(y)) for x, y in tape.waypoints[::2]]
        if len(pts) >= 2:
            pygame.draw.lines(screen, COLOR_TAPE, False, pts, 3)

    # Nodes
    for nid in graph.graph.nodes:
        x, y = graph.node_xy(nid)
        pygame.draw.circle(screen, COLOR_NODE, (int(x), int(y)), 3)

    # Stations as distinct hubs
    for nid, st in graph.stations.items():
        x, y = graph.node_xy(nid)
        pygame.draw.circle(screen, COLOR_STATION, (int(x), int(y)), 12)
        pygame.draw.circle(screen, (255, 255, 255), (int(x), int(y)), 12, 2)
        font = pygame.font.SysFont("consolas", 14)
        label = font.render(
            f"Q{len(st.queue)} P{st.available_batteries}",
            True, (255, 255, 255)
        )
        screen.blit(label, (x - 22, y + 16))


def draw_hud(screen: pygame.Surface,
             ai: CentralAI,
             capsules: List[Capsule],
             target: int,
             emergency: bool) -> None:
    font = pygame.font.SysFont("consolas", 16)
    h = ai.time_of_day
    hh, mm = int(h), int((h - int(h)) * 60)

    idle    = sum(1 for c in capsules if c.status == Capsule.IDLE)
    waiting = sum(1 for c in capsules if c.status == Capsule.WAITING)
    swap    = sum(1 for c in capsules if c.status == Capsule.SWAPPING)
    broken  = sum(1 for c in capsules if c.status == Capsule.BROKEN)

    lines = [
        f"CapsuleNet | {hh:02d}:{mm:02d} | Mode {SCHEDULING_MODE} | Demand {ai.demand:.2f}",
        f"Fleet: {len(capsules):>3} / target {target:>3} | "
        f"idle {idle} | waiting {waiting} | swap {swap} | broken {broken}",
        "Press [SPACE] to toggle city-wide emergency stop",
    ]
    colors = [COLOR_HUD, COLOR_HUD, COLOR_HUD_ALERT if emergency else COLOR_HUD]

    y = 12
    for text, color in zip(lines, colors):
        screen.blit(font.render(text, True, color), (16, y))
        y += 22

    # Station matrix
    station_lines = []
    for nid, st in ai.graph.stations.items():
        station_lines.append(
            f"  Station {nid:>3}: packs {st.available_batteries:>2}/{st.capacity}"
            f" | queue {len(st.queue):>2} | bays {st.active_slots}/{MAX_STATION_SLOTS}"
        )
    for text in station_lines:
        screen.blit(font.render(text, True, COLOR_STATION), (16, y))
        y += 20


def main() -> None:
    global CAPSULE_WORLD

    pygame.init()
    pygame.display.set_caption("CapsuleNet — NetworkX Urban Transit Simulation")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # ---- Infrastructure ----
    graph = UrbanGraph()
    CAPSULE_WORLD = graph

    # ---- Intelligence ----
    central = CentralAI(graph)

    # ---- Active fleet ----
    capsules: List[Capsule] = []

    emergency = False
    dt_seconds = 1.0 / FPS
    running = True

    while running:
        # ---- Events ---------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    emergency = not emergency

        # ---- Central AI: schedule, fleet size, speeds -----------------------
        central.update_schedule(SCHEDULING_MODE, dt_seconds * 0.6)
        central.tick_spawn_cooldown(dt_seconds)
        target_count = central.fleet_target()

        # Spawn new capsules ONLY at stations, only via CentralAI
        if len(capsules) < target_count:
            new_cap = central.maybe_spawn(capsules)
            if new_cap is not None:
                capsules.append(new_cap)

        # Surplus fleet -> demote to IDLE (they finish their loop, then dock)
        elif len(capsules) > target_count:
            for cap in capsules:
                if cap.status == Capsule.OPERATIONAL and not cap.needs_swap:
                    cap.status = Capsule.IDLE
                    break

        # ---- Stations tick --------------------------------------------------
        for st in graph.stations.values():
            st.update(dt_seconds)

        # ---- Random hardware failures --------------------------------------
        if capsules and random.random() < 0.002:
            unlucky = random.choice(capsules)
            if unlucky.status not in (Capsule.BROKEN, Capsule.SWAPPING):
                unlucky.is_broken = True

        # ---- Central speed broadcast ---------------------------------------
        central_signals = {
            "recommended_speed": central.recommend_speed(len(capsules)),
        }

        # ---- Dynamic swap routing (CentralAI-only strategic move) ----------
        for cap in capsules:
            if cap.needs_swap and cap.reserved_station is None \
               and cap.status == Capsule.OPERATIONAL \
               and not cap.is_broken:
                central.assign_swap(cap)

        # ---- Arrival at reserved station -> enter WAITING queue ------------
        for cap in capsules:
            if cap.reserved_station is None:
                continue
            sx, sy = graph.node_xy(cap.reserved_station)
            if math.hypot(cap.x - sx, cap.y - sy) < 20 \
               and cap.status == Capsule.OPERATIONAL:
                cap.status = Capsule.WAITING
                cap.speed = 0.0

        # ---- Stations serve their queues -----------------------------------
        for st in graph.stations.values():
            st.try_serve(dt_seconds)

        # ---- Render ---------------------------------------------------------
        screen.fill(COLOR_BG)
        draw_graph(screen, graph)

        # Update and draw capsules
        for i, cap in enumerate(capsules):
            cap.update(central_signals, emergency, capsules, graph, dt_seconds)
            cap.draw(screen, highlight=(i == 0))

        # ---- Park IDLE capsules once they reach a station node -------------
        to_park = []
        for cap in capsules:
            if cap.status == Capsule.IDLE and cap.current_edge is not None:
                u, v = cap.current_edge
                for nid in (u, v):
                    sx, sy = graph.node_xy(nid)
                    if math.hypot(cap.x - sx, cap.y - sy) < 18:
                        to_park.append(cap)
                        break
        for cap in to_park:
            capsules.remove(cap)

        # ---- HUD ------------------------------------------------------------
        draw_hud(screen, central, capsules, target_count, emergency)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()