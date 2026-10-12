"""
CapsuleNet — Centrally Orchestrated Tape-Guided Urban Transit.

Data-Oriented Design (DOD) simulation with NetworkX infrastructure and
vectorized NumPy physics. Designed to sustain 10,000 active capsules at 60 FPS.

Architecture
------------
*   UrbanGraph       — NetworkX road graph; every directed edge is indexed
                      0..M-1 with a precomputed waypoint cache (W, 2).
*   VectorEngine     — the central state matrix (MAX_CAPSULES=10,000) and
                      the vectorized physics pipeline (no per-capsule loops).
*   CentralAI        — macro scheduler. Fleet sizing, demand, speeds, routing
                      via nx.shortest_path, dynamic swap-station dispatch.
*   Renderer         — bulk blitting via cached sprite + pygame.surfarray.

Dependencies: pygame, networkx, numpy.
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pygame


# ===========================================================================
# GLOBAL CONFIG
# ===========================================================================

WIDTH, HEIGHT = 1200, 800
FPS = 60

# Fleet
MAX_CAPSULES = 10_000
INITIAL_ACTIVE = 2_000

# Physics (per tick at 60 Hz)
MAX_SPEED = 5.0
IDLE_SPEED_CAP = 2.5
WARN_SPEED_FACTOR = 0.4
NEEDS_SWAP_SPEED_FACTOR = 0.6
ACCEL = 0.10
DECEL = 0.20

# Waypoint / tape geometry
WAYPOINT_STEP = 6.0

# Safety
SAFE_DISTANCE = 30.0
WARN_DISTANCE = 60.0
SAFE_DISTANCE_SQ = SAFE_DISTANCE * SAFE_DISTANCE
WARN_DISTANCE_SQ = WARN_DISTANCE * WARN_DISTANCE

# Battery
BATTERY_LOW = 30.0
BATTERY_CRITICAL = 15.0
BATTERY_DRAIN_BASE = 0.02
BATTERY_DRAIN_SPEED = 0.05
BATTERY_RECHARGE_RATE = 0.6
SWAP_DURATION_TICKS = 90

# Local PID (lane keeping)
PID_KP = 0.6
PID_KI = 0.05
PID_INT_CLAMP = 1.0
DRIFT_CLAMP = 3.0
DRIFT_APPLY = 0.15

# Station
STATION_CAPACITY = 10
STATION_RECHARGE_SEC = 6.0
MAX_STATION_SLOTS = 2

# Scheduling
TICKS_PER_HOUR = 900
SCHEDULING_MODE = "DECENTRALIZED"

# Status codes (int8)
ST_OPERATIONAL   = 0
ST_IDLE          = 1
ST_WAITING       = 2
ST_SWAPPING      = 3
ST_BROKEN        = 4
ST_EMERGENCY     = 5

# Palette
COLOR_BG         = (14, 14, 20)
COLOR_TAPE       = (60, 72, 108)
COLOR_NODE       = (160, 160, 180)
COLOR_STATION    = (255, 184, 108)
COLOR_CAPSULE    = (80, 250, 123)
COLOR_IDLE       = (110, 110, 140)
COLOR_WAITING    = (241, 250, 140)
COLOR_SWAPPING   = (255, 184, 108)
COLOR_BROKEN     = (255, 85, 85)
COLOR_HUD        = (220, 220, 220)
COLOR_HUD_ALERT  = (255, 100, 100)


# ===========================================================================
# LAYER 1 — INFRASTRUCTURE (NetworkX + waypoint cache)
# ===========================================================================

class BatterySwapStation:
    """Station hub attached to a specific graph node id."""

    def __init__(self, node_id: int, xy: Tuple[float, float], capacity: int):
        self.node_id = node_id
        self.x, self.y = xy
        self.capacity = capacity
        self.available_batteries = capacity
        self.recharging: List[float] = []
        self.queue_count = 0          # macro counter — capsules tracked by mask
        self.active_slots = 0
        self.swap_slots: List[int] = []   # capsule row ids currently swapping

    def update(self, dt: float) -> None:
        still = []
        for t in self.recharging:
            t -= dt
            if t <= 0:
                if self.available_batteries < self.capacity:
                    self.available_batteries += 1
            else:
                still.append(t)
        self.recharging = still

    def load_factor(self) -> float:
        return (self.queue_count + self.active_slots * 1.5
                - 0.5 * self.available_batteries)


class UrbanGraph:
    """
    NetworkX road graph with O(1) mapping from edge-index to waypoint cache.

    edge_id -> (u, v)
    waypoint_cache[eid] -> np.ndarray (W, 2) float32
    edge_length_px[eid] -> float
    """

    def __init__(self):
        self.graph: nx.DiGraph = nx.DiGraph()
        self.edges: List[Tuple[int, int]] = []
        self.edge_to_id: Dict[Tuple[int, int], int] = {}
        self.station_nodes: Dict[int, BatterySwapStation] = {}

        self._build_synthetic_city()

        # ---- Waypoint caches per edge ----
        self.edge_waypoints: List[np.ndarray] = []
        self.edge_len_px: np.ndarray = np.zeros(len(self.edges), dtype=np.float32)
        self._build_waypoint_caches()

    # ------------------------------------------------------------------ build

    def _build_synthetic_city(self) -> None:
        rng = random.Random(20240101)
        cols, rows = 6, 4
        mx, my = 120, 120
        sx = (WIDTH - 2 * mx) / (cols - 1)
        sy = (HEIGHT - 2 * my) / (rows - 1)

        nid = 0
        grid: Dict[Tuple[int, int], int] = {}
        for r in range(rows):
            for c in range(cols):
                x = mx + c * sx + rng.uniform(-18, 18)
                y = my + r * sy + rng.uniform(-18, 18)
                self.graph.add_node(nid, x=x, y=y)
                grid[(r, c)] = nid
                nid += 1

        for r in range(rows):
            for c in range(cols - 1):
                if rng.random() < 0.90:
                    self._add_pair(grid[(r, c)], grid[(r, c + 1)])
        for r in range(rows - 1):
            for c in range(cols):
                if rng.random() < 0.85:
                    self._add_pair(grid[(r, c)], grid[(r + 1, c)])

        # Express diagonals
        self._add_pair(grid[(0, 0)], grid[(rows - 1, cols - 1)], bi=False)
        self._add_pair(grid[(0, cols - 1)], grid[(rows - 1, 0)], bi=False)

        # Fallback ring for weak connectivity
        if not nx.is_weakly_connected(self.graph):
            nodes = list(self.graph.nodes)
            for i in range(len(nodes)):
                self._add_pair(nodes[i], nodes[(i + 1) % len(nodes)])

        # Three stations at distinct junctions
        station_nodes = [
            grid[(0, 0)],
            grid[(rows - 1, cols - 1)],
            grid[(rows // 2, cols // 2)],
        ]
        for s in station_nodes:
            xy = (self.graph.nodes[s]["x"], self.graph.nodes[s]["y"])
            self.station_nodes[s] = BatterySwapStation(s, xy, STATION_CAPACITY)

        # Build the flat edge list + index
        self.edges = list(self.graph.edges)
        for i, e in enumerate(self.edges):
            self.edge_to_id[e] = i

    def _add_pair(self, u: int, v: int, bi: bool = True) -> None:
        if not self.graph.has_edge(u, v):
            self.graph.add_edge(u, v)
        if bi and not self.graph.has_edge(v, u):
            self.graph.add_edge(v, u)

    def _build_waypoint_caches(self) -> None:
        for (u, v) in self.edges:
            xu, yu = self.graph.nodes[u]["x"], self.graph.nodes[u]["y"]
            xv, yv = self.graph.nodes[v]["x"], self.graph.nodes[v]["y"]
            dist = math.hypot(xv - xu, yv - yu)
            n = max(2, int(dist / WAYPOINT_STEP))
            t = np.linspace(0.0, 1.0, n + 1, dtype=np.float32)
            xs = xu + (xv - xu) * t
            ys = yu + (yv - yu) * t
            self.edge_waypoints.append(
                np.stack([xs, ys], axis=1).astype(np.float32)
            )
            self.edge_len_px[len(self.edge_waypoints) - 1] = float(n)

    # --------------------------------------------------------------- lookup

    def node_xy(self, nid: int) -> Tuple[float, float]:
        n = self.graph.nodes[nid]
        return n["x"], n["y"]

    def edge_start_node(self, eid: int) -> int:
        return self.edges[eid][0]

    def edge_end_node(self, eid: int) -> int:
        return self.edges[eid][1]

    def edge_id(self, u: int, v: int) -> Optional[int]:
        return self.edge_to_id.get((u, v))

    def station_node_list(self) -> List[int]:
        return list(self.station_nodes.keys())


# ===========================================================================
# LAYER 2 — VECTORIZED PHYSICS (Data-Oriented State Matrix)
# ===========================================================================

class VectorEngine:
    """
    Central state matrix + vectorized physics pipeline.

    All per-capsule physics runs on contiguous NumPy arrays of size
    MAX_CAPSULES. No Python loop ever touches individual rows in the
    frame update; only boolean masks + fancy indexing.
    """

    def __init__(self, graph: UrbanGraph):
        self.graph = graph
        self.n = MAX_CAPSULES
        M = len(graph.edges)

        # ---- Kinematic state ----
        self.x           = np.zeros(self.n, dtype=np.float32)
        self.y           = np.zeros(self.n, dtype=np.float32)
        self.speed       = np.zeros(self.n, dtype=np.float32)
        self.target      = np.zeros(self.n, dtype=np.float32)
        self.edge_idx    = np.zeros(self.n, dtype=np.int32)
        self.wp_idx      = np.zeros(self.n, dtype=np.int32)

        # ---- Battery / status ----
        self.battery       = np.full(self.n, 100.0, dtype=np.float32)
        self.needs_swap    = np.zeros(self.n, dtype=np.int8)
        self.status        = np.zeros(self.n, dtype=np.int8)
        self.swap_timer    = np.zeros(self.n, dtype=np.int32)

        # ---- Local PID scratch ----
        self.sensor_offset = np.zeros(self.n, dtype=np.float32)
        self.pid_integral  = np.zeros(self.n, dtype=np.float32)
        self.drift         = np.zeros(self.n, dtype=np.float32)

        # ---- V2V scratch ----
        self.gap_sq        = np.full(self.n, np.inf, dtype=np.float32)

        # ---- Routing (route chains) ----
        # We store, per capsule, its current target station or destination node.
        # At each junction CentralAI rewrites edge_idx and wp_idx.
        self.dest_node     = np.full(self.n, -1, dtype=np.int32)
        self.reserved_station = np.full(self.n, -1, dtype=np.int32)

        # ---- Bookkeeping ----
        self.active_mask   = np.zeros(self.n, dtype=np.int8)
        self.next_free_row = 0
        self.rng           = np.random.default_rng(0xC0FFEE)

        # ---- Precomputed edge constants as flat arrays for O(1) lookup ----
        self.edge_wp_count = np.array(
            [wp.shape[0] for wp in graph.edge_waypoints], dtype=np.int32
        )
        # Concatenated waypoint buffer + per-edge offset for flat indexing
        offsets = [0]
        for wp in graph.edge_waypoints:
            offsets.append(offsets[-1] + wp.shape[0])
        self.wp_offsets = np.array(offsets, dtype=np.int32)
        self.wp_flat = np.concatenate(graph.edge_waypoints, axis=0).astype(np.float32)

        # Scratch buffers reused each tick
        self._scratch_bool   = np.zeros(self.n, dtype=bool)
        self._scratch_i32    = np.zeros(self.n, dtype=np.int32)
        self._scratch_f32    = np.zeros(self.n, dtype=np.float32)

    # ------------------------------------------------------------ activation

    def activate_row(self, i: int, edge_id: int, wp_start: int = 0) -> None:
        """Initialize a fresh capsule row at (edge_id, wp_start)."""
        wp = self.graph.edge_waypoints[edge_id]
        wp_start = int(np.clip(wp_start, 0, wp.shape[0] - 1))
        x, y = wp[wp_start]
        self.x[i] = x
        self.y[i] = y
        self.speed[i] = 1.0
        self.target[i] = MAX_SPEED
        self.edge_idx[i] = edge_id
        self.wp_idx[i] = wp_start
        self.battery[i] = 100.0
        self.needs_swap[i] = 0
        self.status[i] = ST_OPERATIONAL
        self.swap_timer[i] = 0
        self.sensor_offset[i] = 0.0
        self.pid_integral[i] = 0.0
        self.drift[i] = 0.0
        self.dest_node[i] = -1
        self.reserved_station[i] = -1
        self.active_mask[i] = 1

    def deactivate_row(self, i: int) -> None:
        self.active_mask[i] = 0
        self.status[i] = ST_OPERATIONAL

    # --------------------------------------------------------------- physics

    def step(self, central_signals: dict, dt: float = 1.0) -> None:
        """
        One vectorized physics tick. No Python loops over capsules.
        CentralAI may rewrite edge_idx / wp_idx after this call via the
        junction gateway.
        """
        act = self.active_mask.astype(bool)  # bool mask for boolean indexing

        # ---------------------------------------------------------------
        # 1. Target speed base
        # ---------------------------------------------------------------
        base = central_signals.get("recommended_speed", MAX_SPEED)
        self.target[:] = base

        # IDLE rows slow down
        idle = act & (self.status == ST_IDLE)
        np.minimum(self.target, IDLE_SPEED_CAP, out=self.target, where=idle)

        # Low-battery rows slow down
        low = act & (self.needs_swap == 1)
        scaled = (self.target * NEEDS_SWAP_SPEED_FACTOR).astype(np.float32)
        self.target = np.where(low, scaled, self.target).astype(np.float32)

        # Emergency / broken rows -> target 0
        em = act & ((self.status == ST_BROKEN) | (self.status == ST_EMERGENCY))
        self.target[em] = 0.0

        # Parked rows (WAITING / SWAPPING) -> speed 0, target 0
        parked = act & ((self.status == ST_WAITING) | (self.status == ST_SWAPPING))
        self.speed[parked] = 0.0
        self.target[parked] = 0.0

        # ---------------------------------------------------------------
        # 2. Battery drain (only OPERATIONAL rows)
        # ---------------------------------------------------------------
        op = act & (self.status == ST_OPERATIONAL)
        sr = (self.speed / MAX_SPEED).astype(np.float32)
        drain = (BATTERY_DRAIN_BASE + BATTERY_DRAIN_SPEED * sr).astype(np.float32)
        # Only apply to operational rows
        np.subtract(self.battery, np.where(op, drain, 0.0).astype(np.float32),
                    out=self.battery)
        np.clip(self.battery, 0.0, 100.0, out=self.battery)

        # Flag rows that need a swap
        low_bat = op & (self.battery < BATTERY_LOW)
        self.needs_swap[low_bat] = 1

        # ---------------------------------------------------------------
        # 3. V2V safety reflex (vectorized, per edge)
        # ---------------------------------------------------------------
        self._compute_gaps_v2v(act)

        hard = act & (self.gap_sq < SAFE_DISTANCE_SQ)
        warn = act & (self.gap_sq >= SAFE_DISTANCE_SQ) & (self.gap_sq < WARN_DISTANCE_SQ)
        self.target[hard] = 0.0
        self.target[warn] = np.minimum(
            self.target[warn], (MAX_SPEED * WARN_SPEED_FACTOR)
        ).astype(np.float32)

        # ---------------------------------------------------------------
        # 4. Kinematics (accel / decel + speed integration)
        # ---------------------------------------------------------------
        # Accel branch
        acc = act & (self.speed < self.target)
        self.speed[acc] = np.minimum(self.speed[acc] + ACCEL, self.target[acc])
        # Decel branch
        dec = act & (self.speed > self.target)
        self.speed[dec] = np.maximum(self.speed[dec] - DECEL, self.target[dec])
        np.clip(self.speed, 0.0, MAX_SPEED, out=self.speed)

        # ---------------------------------------------------------------
        # 5. Waypoint advance: wp_idx += round(speed * dt)
        #    Capsules may advance by more than one waypoint per tick.
        # ---------------------------------------------------------------
        step = np.rint(self.speed * dt).astype(np.int32)
        self.wp_idx = np.where(act, self.wp_idx + step, self.wp_idx).astype(np.int32)

        # Clamp wp_idx against per-edge waypoint count; values >= count signal
        # "junction reached" and will be handled by the CentralAI gateway.
        max_wp = self.edge_wp_count[self.edge_idx]
        # Do NOT clamp here — the gateway needs the overflow to detect arrival.

        # ---------------------------------------------------------------
        # 6. Position lookup: flat index into wp_flat
        # ---------------------------------------------------------------
        wp_lookup = np.clip(self.wp_idx, 0, max_wp - 1)
        flat = self.wp_offsets[self.edge_idx] + wp_lookup
        pos = self.wp_flat[flat]         # (n, 2)
        self.x = pos[:, 0].copy()
        self.y = pos[:, 1].copy()

        # ---------------------------------------------------------------
        # 7. Hall sensor noise + PID lateral drift
        # ---------------------------------------------------------------
        noise = self.rng.normal(0.0, 0.08, size=self.n).astype(np.float32)
        self.sensor_offset = np.where(act, noise, 0.0).astype(np.float32)

        self.pid_integral[act] += self.sensor_offset[act]
        np.clip(self.pid_integral, -PID_INT_CLAMP, PID_INT_CLAMP,
                out=self.pid_integral)

        pid_out = (-(PID_KP * self.sensor_offset
                     + PID_KI * self.pid_integral)).astype(np.float32)
        self.drift = np.where(
            act,
            self.drift + self.sensor_offset * 0.5 + pid_out,
            0.0
        ).astype(np.float32)
        np.clip(self.drift, -DRIFT_CLAMP, DRIFT_CLAMP, out=self.drift)

        # Perpendicular offset: rotate the local direction of travel by 90°
        # We approximate the direction using a finite difference along wp_flat.
        next_flat = np.clip(flat + 1, 0, self.wp_flat.shape[0] - 1)
        nxt = self.wp_flat[next_flat]
        dx = nxt[:, 0] - self.x
        dy = nxt[:, 1] - self.y
        norm = np.hypot(dx, dy)
        norm[norm < 1e-5] = 1.0
        px = -dy / norm
        py = dx / norm
        self.x = self.x + px * self.drift * DRIFT_APPLY
        self.y = self.y + py * self.drift * DRIFT_APPLY

        # ---------------------------------------------------------------
        # 8. Swap timers (vectorized)
        # ---------------------------------------------------------------
        swapping = act & (self.status == ST_SWAPPING)
        self.swap_timer[swapping] -= 1
        done = swapping & (self.swap_timer <= 0)
        self.status[done] = ST_OPERATIONAL
        self.battery[done] = 100.0
        self.needs_swap[done] = 0
        self.reserved_station[done] = -1

    # ------------------------------------------------------------- V2V gaps

    def _compute_gaps_v2v(self, act: np.ndarray) -> None:
        """
        Vectorized leading-gap computation.

        Approach: for each edge, gather capsule indices sharing that edge,
        sort them by wp_idx, compute the squared distance between consecutive
        capsules, and scatter the result back to gap_sq. This is O(N log N)
        globally but only O(k log k) per edge with small k in practice.
        """
        self.gap_sq.fill(np.inf)

        if not np.any(act):
            return

        # Group active rows by edge_idx
        active_rows = np.flatnonzero(act)
        if active_rows.size == 0:
            return

        edges_of_active = self.edge_idx[active_rows]
        # Sort by (edge, wp_idx) — a single vectorized lexsort pass
        wp_of_active = self.wp_idx[active_rows]
        order = np.lexsort((wp_of_active, edges_of_active))
        sorted_rows = active_rows[order]
        sorted_edges = edges_of_active[order]
        sorted_wp = wp_of_active[order]

        # Boundaries between edges
        # edge changes: keep only pairs where the current edge equals the next
        if sorted_rows.size >= 2:
            same_edge = sorted_edges[:-1] == sorted_edges[1:]
            lead_rows = sorted_rows[:-1][same_edge]
            trail_rows = sorted_rows[1:][same_edge]

            dx = self.x[lead_rows] - self.x[trail_rows]
            dy = self.y[lead_rows] - self.y[trail_rows]
            gap_sq = dx * dx + dy * dy

            # Scatter back — write per trailing row (no duplicates by design)
            self.gap_sq[trail_rows] = gap_sq.astype(np.float32)

    # ------------------------------------------------------------ render data

    def active_xy_status(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (x, y, status) slices for active rows (bulk render)."""
        mask = self.active_mask.astype(bool)
        return self.x[mask], self.y[mask], self.status[mask]

    def active_count(self) -> int:
        return int(self.active_mask.sum())


# ===========================================================================
# LAYER 3 — CENTRAL AI (Strategic Only)
# ===========================================================================

class CentralAI:
    """
    Sole strategic authority. Operates on the engine's NumPy state through
    a small, event-driven interface (junction gateway + demand pulse).
    """

    def __init__(self, graph: UrbanGraph, engine: VectorEngine):
        self.graph = graph
        self.engine = engine
        self.time_of_day = 6.0
        self.demand = 0.2
        self._spawn_cooldown = 0.0

    # ---------------------------------------------------------- demand pulse

    def update_schedule(self, mode: str, dt_hours: float) -> float:
        self.time_of_day = (self.time_of_day + dt_hours) % 24.0
        h = self.time_of_day
        if mode == "CLASSIC":
            self.demand = 1.0 if 8.0 <= h <= 9.0 else 0.1
        else:
            if   7.0 <= h <= 8.0:   self.demand = 0.4
            elif 8.5 <= h <= 9.5:   self.demand = 0.5
            elif 10.0 <= h <= 11.0: self.demand = 0.3
            else:                   self.demand = 0.15
        return self.demand

    def fleet_target(self) -> int:
        return max(2, int(self.demand * 30))

    def recommended_speed(self, active: int) -> float:
        return 4.5 if active <= 15 else 2.5

    def tick(self, dt: float) -> None:
        self._spawn_cooldown = max(0.0, self._spawn_cooldown - dt)

    # ------------------------------------------------------------ spawn fleet

    def maybe_spawn(self) -> bool:
        if self._spawn_cooldown > 0:
            return False
        # Choose least-loaded station as depot
        st_nodes = self.graph.station_node_list()
        if not st_nodes:
            return False
        best = min(st_nodes, key=lambda n: self.graph.station_nodes[n].load_factor())
        # Spawn on one of the outgoing edges of that station node
        outgoing = [eid for eid, (u, v) in enumerate(self.graph.edges) if u == best]
        if not outgoing:
            return False
        eid = random.choice(outgoing)
        row = self.engine.next_free_row
        # Find next free row (bounded, initialization-time only)
        inactive = np.flatnonzero(self.engine.active_mask == 0)
        if inactive.size == 0:
            return False
        row = int(inactive[0])
        self.engine.activate_row(row, eid, wp_start=0)
        # Random destination node
        self.engine.dest_node[row] = random.choice(list(self.graph.graph.nodes))
        self.engine.next_free_row = row + 1
        self._spawn_cooldown = 0.25
        return True

    # ---------------------------------------------------------- junction gate

    def process_junctions(self, emergency: bool) -> None:
        """
        Vectorized junction event gateway.

        Detects rows whose wp_idx has overflowed their current edge's
        waypoint count, freezes speed, and rewrites their edge_idx to the
        next segment in their route. Route recomputation uses NetworkX.
        """
        eng = self.engine
        act = eng.active_mask.astype(bool)
        if not np.any(act):
            return

        max_wp = eng.edge_wp_count[eng.edge_idx]
        at_junction = act & (eng.wp_idx >= max_wp)
        if not np.any(at_junction):
            return

        # Freeze speed for rows at junction
        eng.speed[at_junction] = 0.0

        # Enumerate the (usually small) set of junction rows
        rows = np.flatnonzero(at_junction)
        for row in rows:
            self._resolve_junction(int(row), emergency)

    def _resolve_junction(self, row: int, emergency: bool) -> None:
        """Per-junction route resolution. Small population, called rarely."""
        eng = self.engine
        cur_eid = int(eng.edge_idx[row])
        u, v = self.graph.edges[cur_eid]

        # Determine the strategic destination for this capsule
        if eng.needs_swap[row] == 1:
            # Dynamic swap routing: reroute to the least-loaded station
            dest = self._pick_swap_station_node()
            eng.reserved_station[row] = dest
        else:
            dest = int(eng.dest_node[row])
            if dest < 0 or dest == v:
                # Assign a new random destination and continue
                dest = random.choice(list(self.graph.graph.nodes))
                eng.dest_node[row] = dest

        # Plan the next hop using NetworkX shortest_path
        try:
            path = nx.shortest_path(self.graph.graph, v, dest, weight=None)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            path = None

        if not path or len(path) < 2:
            # No route: pick a random outgoing edge as a fallback
            outgoing = [eid for eid, (a, b) in enumerate(self.graph.edges) if a == v]
            if not outgoing:
                eng.deactivate_row(row)
                return
            new_eid = random.choice(outgoing)
        else:
            new_eid = self.graph.edge_id(path[0], path[1])
            if new_eid is None:
                eng.deactivate_row(row)
                return

        # Rewrite state matrix in O(1)
        eng.edge_idx[row] = new_eid
        eng.wp_idx[row] = 0
        # Snap position to the new edge start
        x0, y0 = self.graph.edge_waypoints[new_eid][0]
        eng.x[row] = x0
        eng.y[row] = y0

    def _pick_swap_station_node(self) -> int:
        """Pick the station node with the lowest current load factor."""
        nodes = self.graph.station_node_list()
        return min(nodes, key=lambda n: self.graph.station_nodes[n].load_factor())

    # ------------------------------------------------------- station serving

    def serve_stations(self, dt: float) -> None:
        """
        Admit WAITING rows to stations with free slots and available packs,
        and advance recharge timers.
        """
        eng = self.engine
        for st in self.graph.station_nodes.values():
            st.update(dt)
            if st.active_slots >= MAX_STATION_SLOTS or st.available_batteries <= 0:
                continue
            # Vectorized selection: WAITING rows whose reserved_station matches
            waiting = (eng.active_mask.astype(bool)
                       & (eng.status == ST_WAITING)
                       & (eng.reserved_station == st.node_id))
            rows = np.flatnonzero(waiting)
            free_bays = MAX_STATION_SLOTS - st.active_slots
            take = rows[:min(free_bays, st.available_batteries)]
            if take.size == 0:
                continue
            eng.status[take] = ST_SWAPPING
            eng.swap_timer[take] = SWAP_DURATION_TICKS
            eng.speed[take] = 0.0
            st.active_slots += take.size
            st.available_batteries -= take.size
            for _ in take:
                st.recharging.append(STATION_RECHARGE_SEC)

    def flag_waiting_at_stations(self) -> None:
        """
        Vectorized: rows that reached their reserved station become WAITING.
        """
        eng = self.engine
        act = eng.active_mask.astype(bool)
        # Only rows whose current edge ends at their reserved station
        for st_node, st in self.graph.station_nodes.items():
            ending_here = np.array(
                [self.graph.edges[e][1] == st_node for e in range(len(self.graph.edges))],
                dtype=bool
            ) if len(self.graph.edges) < 200 else None  # small graph: cheap mask

            if ending_here is None:
                continue
            on_edge = ending_here[eng.edge_idx]
            near_end = eng.wp_idx >= (eng.edge_wp_count[eng.edge_idx] - 1)
            ready = (act
                     & (eng.reserved_station == st_node)
                     & on_edge
                     & near_end
                     & (eng.status == ST_OPERATIONAL))
            rows = np.flatnonzero(ready)
            if rows.size == 0:
                continue
            eng.status[rows] = ST_WAITING
            eng.speed[rows] = 0.0
            st.queue_count += rows.size

    def refresh_station_queues(self) -> None:
        """Recompute station queue counters from engine state."""
        eng = self.engine
        for st_node, st in self.graph.station_nodes.items():
            st.queue_count = int(np.sum(
                (eng.active_mask.astype(bool))
                & (eng.status == ST_WAITING)
                & (eng.reserved_station == st_node)
            ))

    def demote_surplus(self, target: int) -> None:
        """Vectorized: demote surplus OPERATIONAL rows to IDLE."""
        eng = self.engine
        act = eng.active_mask.astype(bool)
        op = act & (eng.status == ST_OPERATIONAL)
        count = int(op.sum())
        if count <= target:
            return
        excess = count - target
        rows = np.flatnonzero(op)[:excess]
        eng.status[rows] = ST_IDLE

    def trigger_random_failures(self, p: float = 2.0e-5) -> None:
        eng = self.engine
        act = eng.active_mask.astype(bool)
        draw = eng.rng.random(MAX_CAPSULES) < p
        rows = np.flatnonzero(act & draw & (eng.status != ST_BROKEN))
        if rows.size:
            eng.status[rows] = ST_BROKEN
            eng.speed[rows] = 0.0


# ===========================================================================
# LAYER 4 — RENDERER (bulk blitting)
# ===========================================================================

class Renderer:
    """High-throughput Pygame renderer using cached sprites + bulk blits."""

    def __init__(self, screen: pygame.Surface, graph: UrbanGraph):
        self.screen = screen
        self.graph = graph

        # Cached 4x4 dot sprites keyed by status
        self.sprites: Dict[int, pygame.Surface] = {}
        self._build_sprites()

        # Pre-render the static tape graph once onto a background surface
        self.static_bg = pygame.Surface((WIDTH, HEIGHT))
        self.static_bg.fill(COLOR_BG)
        self._draw_static_graph(self.static_bg)

    def _build_sprites(self) -> None:
        palette = {
            ST_OPERATIONAL: COLOR_CAPSULE,
            ST_IDLE:        COLOR_IDLE,
            ST_WAITING:     COLOR_WAITING,
            ST_SWAPPING:    COLOR_SWAPPING,
            ST_BROKEN:      COLOR_BROKEN,
            ST_EMERGENCY:   COLOR_BROKEN,
        }
        for code, col in palette.items():
            surf = pygame.Surface((6, 6), pygame.SRCALPHA)
            pygame.draw.circle(surf, col, (3, 3), 3)
            self.sprites[code] = surf

    def _draw_static_graph(self, surface: pygame.Surface) -> None:
        # Edges as thick tape lines
        for wp in self.graph.edge_waypoints:
            pts = [(int(p[0]), int(p[1])) for p in wp[::3]]
            if len(pts) >= 2:
                pygame.draw.lines(surface, COLOR_TAPE, False, pts, 3)
        # Nodes
        for nid in self.graph.graph.nodes:
            x, y = self.graph.node_xy(nid)
            pygame.draw.circle(surface, COLOR_NODE, (int(x), int(y)), 3)
        # Stations
        for nid, st in self.graph.station_nodes.items():
            x, y = self.graph.node_xy(nid)
            pygame.draw.circle(surface, COLOR_STATION, (int(x), int(y)), 12)
            pygame.draw.circle(surface, (255, 255, 255), (int(x), int(y)), 12, 2)

    def draw(self, engine: VectorEngine) -> None:
        # Blit static background (graph + tape)
        self.screen.blit(self.static_bg, (0, 0))

        # Extract active slices in one vectorized pass
        x, y, status = engine.active_xy_status()
        if x.size == 0:
            return

        # Cast to ints; build blit list grouped by status so we use one
        # bulk screen.blits() call per status group (few calls total).
        xi = x.astype(np.int32)
        yi = y.astype(np.int32)

        blit_seq = []
        for code, sprite in self.sprites.items():
            sel = status == code
            if not np.any(sel):
                continue
            xs = xi[sel] - 3
            ys = yi[sel] - 3
            # Zip into the (surface, dest) tuples pygame.blits expects
            blit_seq.extend(zip([sprite] * xs.size,
                                zip(xs.tolist(), ys.tolist())))
        if blit_seq:
            self.screen.blits(blit_seq, doreturn=False)

    # --------------------------------------------------------------- HUD

    def draw_hud(self, central: CentralAI, engine: VectorEngine,
                 target: int, emergency: bool) -> None:
        font = pygame.font.SysFont("consolas", 16)
        h = central.time_of_day
        hh, mm = int(h), int((h - int(h)) * 60)

        active = engine.active_count()
        idle = int(np.sum((engine.active_mask.astype(bool))
                          & (engine.status == ST_IDLE)))
        waiting = int(np.sum((engine.active_mask.astype(bool))
                             & (engine.status == ST_WAITING)))
        swap = int(np.sum((engine.active_mask.astype(bool))
                          & (engine.status == ST_SWAPPING)))
        broken = int(np.sum((engine.active_mask.astype(bool))
                            & (engine.status == ST_BROKEN)))

        lines = [
            (f"CapsuleNet | {hh:02d}:{mm:02d} | Mode {SCHEDULING_MODE}"
             f" | Demand {central.demand:.2f}", COLOR_HUD),
            (f"Fleet {active:>5} / target {target:>5} | "
             f"idle {idle:>5} | waiting {waiting:>4} | swap {swap:>4} | broken {broken:>3}",
             COLOR_HUD),
            ("Press [SPACE] to toggle city-wide emergency stop",
             COLOR_HUD_ALERT if emergency else COLOR_HUD),
        ]
        y = 12
        for text, col in lines:
            self.screen.blit(font.render(text, True, col), (16, y))
            y += 22

        # Station matrix
        for nid, st in self.graph.station_nodes.items():
            text = (f"  Station {nid:>3} | packs {st.available_batteries:>2}/"
                    f"{st.capacity} | queue {st.queue_count:>3} | "
                    f"bays {st.active_slots}/{MAX_STATION_SLOTS}")
            self.screen.blit(font.render(text, True, COLOR_STATION), (16, y))
            y += 20


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    pygame.init()
    pygame.display.set_caption("CapsuleNet — DOD Vectorized Transit Simulation")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()

    # Infrastructure
    graph = UrbanGraph()
    # Physics / state matrix
    engine = VectorEngine(graph)
    # Strategic intelligence
    central = CentralAI(graph, engine)
    # Renderer
    renderer = Renderer(screen, graph)

    # Initialize the first INITIAL_ACTIVE rows at stations
    station_nodes = graph.station_node_list()
    for i in range(INITIAL_ACTIVE):
        st_node = random.choice(station_nodes)
        outgoing = [eid for eid, (u, v) in enumerate(graph.edges) if u == st_node]
        if not outgoing:
            continue
        eid = random.choice(outgoing)
        engine.activate_row(i, eid, wp_start=0)
        engine.dest_node[i] = random.choice(list(graph.graph.nodes))
    engine.next_free_row = INITIAL_ACTIVE

    emergency = False
    dt_seconds = 1.0 / FPS
    running = True

    while running:
        # ---- Events -------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    emergency = not emergency
                    if emergency:
                        eng = engine
                        act = eng.active_mask.astype(bool)
                        eng.status[act] = ST_EMERGENCY
                        eng.target[act] = 0.0
                    else:
                        eng = engine
                        em = eng.active_mask.astype(bool) & (eng.status == ST_EMERGENCY)
                        eng.status[em] = ST_OPERATIONAL

        # ---- Central AI (strategic) ---------------------------------------
        central.update_schedule(SCHEDULING_MODE, dt_seconds * 0.6)
        central.tick(dt_seconds)

        target_fleet = central.fleet_target()
        current_active = engine.active_count()

        # Spawn new rows via CentralAI (bounded, at most 1 per frame)
        if current_active < target_fleet:
            central.maybe_spawn()

        # Demote surplus rows (vectorized)
        central.demote_surplus(target_fleet)

        # Random hardware failures (vectorized)
        central.trigger_random_failures()

        # ---- Central speed broadcast --------------------------------------
        signals = {
            "recommended_speed": central.recommended_speed(engine.active_count()),
        }

        # ---- Vectorized physics tick --------------------------------------
        engine.step(signals, dt=1.0)

        # ---- Junction gateway (route resolution, mostly idle) -------------
        central.process_junctions(emergency)

        # ---- Station macro logic ------------------------------------------
        central.refresh_station_queues()
        central.flag_waiting_at_stations()
        central.serve_stations(dt_seconds)

        # ---- Render --------------------------------------------------------
        renderer.draw(engine)
        renderer.draw_hud(central, engine, target_fleet, emergency)

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()