"""
CapsuleNet — Hybrid AI Simulation of Autonomous Magnetic-Tape Urban Transit.

Single-file reference implementation:
  - Edge AI: Hall-effect lane keeping + PID return + V2V / LiDAR safety gap.
  - Macro AI: dynamic speed caps, staggered shift scheduling, battery-swap queues.
  - Infrastructure: swap stations with finite battery stock and recharge timers.

No Russian text in code, logs or UI — English only.
"""

import pygame
import random
import math

# --- SIMULATION CONSTANTS ---
WIDTH, HEIGHT = 1200, 800
FPS = 60

# Palette
COLOR_BG        = (20, 20, 25)
COLOR_TAPE      = (98, 114, 164)
COLOR_CAPSULE   = (80, 250, 123)
COLOR_IDLE      = (120, 120, 140)
COLOR_STATION   = (255, 184, 108)
COLOR_EMERGENCY = (255, 85, 85)
COLOR_WAITING   = (241, 250, 140)

# Physics / policy constants
SAFE_DISTANCE      = 30.0     # px — hard V2V braking threshold
WARN_DISTANCE      = 60.0     # px — soft V2V speed reduction
BATTERY_LOW        = 30.0     # % — threshold to request a swap slot
SWAP_TIME          = 1.5      # s  — robotic arm swap duration
STATION_RECHARGE   = 6.0      # s  — time to recharge one depleted pack
MAX_STATION_SLOTS  = 2        # concurrent swap bays per station
IDLE_SPEED_CAP     = 2.5      # px/frame — cap for IDLE capsules

# Shared registry of capsules currently in a swap bay (used by stations)
_SWAPPING_REGISTRY = []


# ---------------------------------------------------------------------------
# INFRASTRUCTURE
# ---------------------------------------------------------------------------

class BatterySwapStation:
    """Swap station with finite battery stock and per-pack recharge timers."""

    def __init__(self, x, y, capacity=10):
        self.x, self.y = x, y
        self.radius = 25
        self.capacity = capacity

        self.available_batteries = capacity
        self.recharging = []          # list of remaining timers (seconds)
        self.queue = []               # capsules waiting for a bay
        self.active_slots = 0         # bays currently occupied

    def update(self, dt):
        """Advance recharge timers; promote ready packs back into stock."""
        still_charging = []
        for timer in self.recharging:
            timer -= dt
            if timer <= 0:
                if self.available_batteries < self.capacity:
                    self.available_batteries += 1
                # else: pack is discarded (storage is full)
            else:
                still_charging.append(timer)
        self.recharging = still_charging

    def request_swap(self, capsule):
        """Enqueue a capsule if it is not already queued."""
        if capsule in self.queue:
            return False
        self.queue.append(capsule)
        return True

    def try_serve(self, dt):
        """
        Admit new capsules into free bays and advance in-progress swaps.
        Returns a list of capsules that completed their swap this tick.
        """
        served = []

        # Admit new capsules into free bays while stock is available
        while (self.active_slots < MAX_STATION_SLOTS
               and self.queue
               and self.available_batteries > 0):
            cap = self.queue.pop(0)
            self.available_batteries -= 1
            self.recharging.append(STATION_RECHARGE)
            self.active_slots += 1
            cap.swap_timer = SWAP_TIME
            cap.status = "SWAPPING"

        # Advance in-progress swaps
        for cap in list(_SWAPPING_REGISTRY):
            cap.swap_timer -= dt
            if cap.swap_timer <= 0:
                cap.battery = 100.0
                cap.needs_swap = False
                cap.status = "OPERATIONAL"
                cap.swap_timer = 0.0
                self.active_slots = max(0, self.active_slots - 1)
                served.append(cap)

        return served

    def draw(self, screen):
        pygame.draw.circle(screen, COLOR_STATION, (int(self.x), int(self.y)), self.radius)
        pygame.draw.circle(screen, (255, 255, 255), (int(self.x), int(self.y)), self.radius, 2)

        font = pygame.font.SysFont(None, 18)
        label = font.render(
            f"PACKS {self.available_batteries}/{self.capacity} | Q {len(self.queue)}",
            True, (255, 255, 255)
        )
        screen.blit(label, (self.x - 60, self.y + self.radius + 4))


# ---------------------------------------------------------------------------
# EDGE AGENT
# ---------------------------------------------------------------------------

class Capsule:
    """Autonomous capsule with Hall-sensor lane keeping and V2V safety."""

    def __init__(self, cid, start_x, start_y, path, spawn_idx=0):
        self.id = cid
        self.x, self.y = start_x, start_y
        self.path = path
        self.current_target_idx = (spawn_idx + 1) % len(path)

        self.speed = 0.0
        self.max_speed = 5.0
        self.battery = 100.0
        self.battery_consumption_rate = 0.05

        self.is_broken = False
        self.needs_swap = False
        self.swap_timer = 0.0

        # Status values:
        #   OPERATIONAL | IDLE | WAITING | SWAPPING | BROKEN | EMERGENCY_STOP
        self.status = "OPERATIONAL"

        # Hall sensors / PID state
        self.sensor_offset = 0.0      # instantaneous deviation (-1..1)
        self.lateral_drift = 0.0      # accumulated physical offset (px)
        self._pid_integral = 0.0

        # V2V / LiDAR
        self.leading_gap = float("inf")

    # ------------------------- Local intelligence -------------------------

    def _compute_leading_gap(self, all_capsules):
        """Find the nearest capsule ahead within a ~120 px cone."""
        min_gap = float("inf")
        tx, ty = self.path[self.current_target_idx]
        fx, fy = tx - self.x, ty - self.y
        flen = math.hypot(fx, fy) or 1.0
        fx, fy = fx / flen, fy / flen

        for other in all_capsules:
            if other is self:
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

    def _pid_lane_keeping(self):
        """Simple P+I controller that pulls the capsule back to tape center."""
        kp, ki = 0.6, 0.05
        self._pid_integral += self.sensor_offset
        self._pid_integral = max(-1.0, min(1.0, self._pid_integral))
        return -(kp * self.sensor_offset + ki * self._pid_integral)

    # ----------------------------- Main tick ------------------------------

    def update(self, central_signals, emergency_event, all_capsules, dt=1.0):
        # 0. Parked states — no motion, no consumption
        if self.status in ("SWAPPING", "WAITING"):
            self.speed = 0.0
            return

        # 1. Emergency / failure
        if self.is_broken or emergency_event:
            self.speed = max(0.0, self.speed - 0.2)
            self.status = "BROKEN" if self.is_broken else "EMERGENCY_STOP"
            return

        # 2. Target speed
        if self.status == "IDLE":
            target_speed = min(IDLE_SPEED_CAP, self.max_speed * 0.5)
        else:
            target_speed = central_signals.get("recommended_speed", self.max_speed)
            if self.needs_swap:
                target_speed = min(target_speed, self.max_speed * 0.6)

        # 3. Battery drain
        if self.status == "OPERATIONAL":
            self.battery -= self.battery_consumption_rate * (self.speed / self.max_speed + 0.5)
            if self.battery < BATTERY_LOW:
                self.needs_swap = True
                central_signals.setdefault("swap_requests", []).append(self)

        # 4. V2V / LiDAR emergency braking (independent of central AI)
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

        # 6. Motion along the magnetic tape
        if self.current_target_idx < len(self.path) and self.speed > 0:
            tx, ty = self.path[self.current_target_idx]
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)

            if dist < 1e-3:
                self.current_target_idx = (self.current_target_idx + 1) % len(self.path)
                return

            step = min(self.speed, dist)
            nx, ny = dx / dist, dy / dist
            self.x += nx * step
            self.y += ny * step

            if dist - step < 1e-3:
                self.current_target_idx = (self.current_target_idx + 1) % len(self.path)

            # 6a. Fresh magnetic noise
            self.sensor_offset = random.uniform(-0.1, 0.1)

            # 6b. Accumulate drift
            self.lateral_drift += self.sensor_offset * 0.5

            # 6c. PID correction pulls back toward tape center
            self.lateral_drift += self._pid_lane_keeping()
            self.lateral_drift = max(-3.0, min(3.0, self.lateral_drift))

            # 6d. Apply perpendicular physical offset
            perp_x, perp_y = -ny, nx
            self.x += perp_x * self.lateral_drift * 0.15
            self.y += perp_y * self.lateral_drift * 0.15

    # ------------------------------ Rendering -----------------------------

    def draw(self, screen):
        if self.status in ("BROKEN", "EMERGENCY_STOP"):
            color = COLOR_EMERGENCY
        elif self.status == "IDLE":
            color = COLOR_IDLE
        elif self.status in ("SWAPPING", "WAITING"):
            color = COLOR_WAITING
        elif self.needs_swap:
            color = (241, 250, 140)
        else:
            color = COLOR_CAPSULE

        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), 12)

        if self.id == 0:
            font = pygame.font.SysFont(None, 20)
            img = font.render(
                f"Cap 0 | Bat {int(self.battery)}% | {self.status}",
                True, (255, 255, 255)
            )
            screen.blit(img, (self.x - 60, self.y - 28))


# ---------------------------------------------------------------------------
# MACRO AGENT
# ---------------------------------------------------------------------------

class CentralAI:
    """Macro orchestrator: schedules, speed caps, fleet sizing, swap routing."""

    def __init__(self):
        self.time_of_day = 6.0
        self.traffic_demand_modifier = 0.2
        self._next_capsule_id = 0

    def optimize_traffic_schedule(self, scheduling_type, dt_hours=0.01):
        """Advance city clock and return the current demand modifier."""
        self.time_of_day = (self.time_of_day + dt_hours) % 24.0
        h = self.time_of_day

        if scheduling_type == "CLASSIC":
            self.traffic_demand_modifier = 1.0 if 8.0 <= h <= 9.0 else 0.1
        elif scheduling_type == "DECENTRALIZED":
            if   7.0 <= h <= 8.0:   self.traffic_demand_modifier = 0.4
            elif 8.5 <= h <= 9.5:   self.traffic_demand_modifier = 0.5
            elif 10.0 <= h <= 11.0: self.traffic_demand_modifier = 0.3
            else:                   self.traffic_demand_modifier = 0.15

        return self.traffic_demand_modifier

    def assign_swap_slot(self, capsule, stations):
        """Route a low-battery capsule to the least-loaded station."""
        if not stations:
            return None
        best = min(
            stations,
            key=lambda s: (len(s.queue), s.active_slots, -s.available_batteries)
        )
        best.request_swap(capsule)
        capsule.status = "WAITING"
        return best

    def fleet_target_size(self, demand):
        return max(2, int(demand * 30))


# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("CapsuleNet — Hybrid AI Urban Transit Simulation")
    clock = pygame.time.Clock()

    # Closed loop induction tape
    road_tape = []
    for angle in range(0, 360, 2):
        rad = math.radians(angle)
        road_tape.append((
            WIDTH  // 2 + 400 * math.cos(rad),
            HEIGHT // 2 + 250 * math.sin(rad)
        ))
    road_tape.append(road_tape[0])

    station = BatterySwapStation(WIDTH // 2, HEIGHT // 2 - 250, capacity=10)
    stations = [station]

    central_brain = CentralAI()
    capsules = []

    SCHEDULING_MODE = "DECENTRALIZED"
    global_emergency = False
    dt_seconds = 1.0 / FPS

    running = True
    while running:
        # ---- Event handling -------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                global_emergency = not global_emergency
                print(f"[CENTER] Emergency state: {global_emergency}")

        screen.fill(COLOR_BG)

        # ---- Macro AI: schedule, demand, fleet target -----------------------
        demand = central_brain.optimize_traffic_schedule(
            SCHEDULING_MODE, dt_seconds * 0.6
        )
        target_count = central_brain.fleet_target_size(demand)

        # Spawn new capsules ONLY from the station (never mid-tape).
        if len(capsules) < target_count and random.random() < 0.15:
            start_x, start_y = road_tape[0]
            cap = Capsule(
                central_brain._next_capsule_id,
                start_x, start_y,
                road_tape, spawn_idx=0
            )
            central_brain._next_capsule_id += 1
            capsules.append(cap)
            print(f"[SPAWN] Capsule #{cap.id} departed the station.")

        # Surplus fleet -> mark as IDLE (still moves, no new jobs).
        elif len(capsules) > target_count:
            for cap in capsules:
                if cap.status == "OPERATIONAL" and not cap.needs_swap:
                    cap.status = "IDLE"
                    print(f"[IDLE] Capsule #{cap.id} switched to idle mode.")
                    break

        # ---- Stations tick (recharge timers) --------------------------------
        for st in stations:
            st.update(dt_seconds)

        # ---- Random hardware failures ---------------------------------------
        if capsules and random.random() < 0.002:
            unlucky = random.choice(capsules)
            if unlucky.status not in ("BROKEN", "SWAPPING"):
                unlucky.is_broken = True
                print(f"[FAULT] Capsule #{unlucky.id} reported a hardware failure.")

        # ---- Central speed recommendation -----------------------------------
        recommended_speed = 4.5 if len(capsules) <= 15 else 2.5
        central_signals = {
            "recommended_speed": recommended_speed,
            "swap_requests": [],
        }

        # ---- Draw tape and stations -----------------------------------------
        if len(road_tape) > 1:
            pygame.draw.lines(screen, COLOR_TAPE, False, road_tape, 6)
        for st in stations:
            st.draw(screen)

        # ---- Swap routing: low battery -> request slot ----------------------
        for cap in capsules:
            if cap.needs_swap and cap.status == "OPERATIONAL" and not cap.is_broken:
                central_brain.assign_swap_slot(cap, stations)

        # ---- Refresh the swap registry used by stations ---------------------
        _SWAPPING_REGISTRY.clear()
        for cap in capsules:
            if cap.status == "SWAPPING" and cap not in _SWAPPING_REGISTRY:
                _SWAPPING_REGISTRY.append(cap)

        # ---- Stations advance their queues ----------------------------------
        for st in stations:
            served = st.try_serve(dt_seconds)
            for cap in served:
                print(f"[SWAP DONE] Capsule #{cap.id} returned to service.")

        # ---- Update + draw capsules -----------------------------------------
        for cap in capsules:
            cap.update(central_signals, global_emergency, capsules, dt_seconds)
            cap.draw(screen)

        # ---- IDLE capsules dock at the station and are parked ---------------
        to_remove = []
        for cap in capsules:
            if cap.status == "IDLE" and \
               math.hypot(cap.x - station.x, cap.y - station.y) < 40:
                to_remove.append(cap)
        for cap in to_remove:
            capsules.remove(cap)
            print(f"[PARKED] Capsule #{cap.id} docked at the station.")

        # ---- HUD ------------------------------------------------------------
        font = pygame.font.SysFont(None, 24)
        hud1 = font.render(
            f"Time {int(central_brain.time_of_day):02d}:00 | Mode {SCHEDULING_MODE}",
            True, (255, 255, 255)
        )
        hud2 = font.render(
            f"Capsules {len(capsules)} / target {target_count} | Demand {demand:.2f}",
            True, (255, 255, 255)
        )
        hud3 = font.render(
            "Press [SPACE] to toggle city-wide emergency stop",
            True, COLOR_EMERGENCY
        )
        hud4 = font.render(
            f"Station | packs {station.available_batteries}/{station.capacity} "
            f"| queue {len(station.queue)} "
            f"| bays {station.active_slots}/{MAX_STATION_SLOTS}",
            True, COLOR_STATION
        )
        screen.blit(hud1, (20, 20))
        screen.blit(hud2, (20, 50))
        screen.blit(hud3, (20, 80))
        screen.blit(hud4, (20, 110))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()