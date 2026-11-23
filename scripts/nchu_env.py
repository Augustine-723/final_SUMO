"""Single-agent Gymnasium environment for the NCHU main gate."""
from collections import Counter
import hashlib
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
from xml.etree import ElementTree as ET

import gymnasium as gym
import numpy as np
import traci
import traci.constants as tc

from nchu_demand import ARMS, load_counts, routes
from nchu_scenario import network


class Signal:
    """Two compatible green groups, each with protected transition stages."""

    def __init__(self, net):
        tls = net.find("tlLogic[@id='gate']")
        phases = tls.findall("phase")
        self.states = [p.get("state") for p in phases]
        self.durations = [int(p.get("duration")) for p in phases]
        self.crossings = [e.get("id") for e in net.findall("edge[@function='crossing']")]
        links = net.findall("connection[@tl='gate']")
        if self.durations != [20, 5, 3, 2] * 2 or len(self.crossings) != 4:
            raise ValueError("Unrecognized NCHU clearance program")
        # Group 0/1 are defined by the generated program; verify the incoming arms.
        self.arms = []
        for base in (0, 4):
            green, clear, yellow, red = self.states[base:base + 4]
            active = set()
            for link in links:
                i = int(link.get("linkIndex"))
                src = link.get("from")
                if src.endswith("_in"):
                    if green[i] in "Gg":
                        active.add(src[0])
                    valid = clear[i] == green[i] and yellow[i] == ("y" if green[i] in "gG" else "r")
                else:
                    valid = clear[i] == yellow[i] == "r"
                if not valid or red[i] != "r":
                    raise ValueError("Invalid green-to-red clearance sequence")
            self.arms.append(active)
        if self.arms != [set("EW"), set("NS")]:
            raise ValueError("Unexpected action-to-arm mapping")
        self.phase = 0
        self.elapsed = 0

    def advance(self, action, occupied):
        """Choose the display for the upcoming second; elapsed counts prior full seconds."""
        stage = self.phase % 4
        if stage == 0:
            change = self.elapsed >= 45 or (action != self.phase // 4 and self.elapsed >= 10)
        else:
            change = self.elapsed >= self.durations[self.phase] and (stage != 3 or not occupied)
        if change:
            self.phase = (self.phase + 1) % 8
            self.elapsed = 0
        self.elapsed += 1
        return change


class GateEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, counts):
        super().__init__()
        self.counts = Path(counts)
        self.bins = load_counts(self.counts)
        self.horizon = 900 * len(self.bins) + 900
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(0, 1, (19,), dtype=np.float32)
        self.tmp = None
        self.conn = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.traffic_seed = int(seed if seed is not None else self.np_random.integers(0, 2**31 - 1))
        self.tmp = TemporaryDirectory(prefix="nchu-env-")
        self.directory = Path(self.tmp.name)
        try:
            network(self.directory)
            net_file = self.directory / "gate.net.xml"
            route_file = self.directory / "gate.rou.xml"
            self.signal = Signal(ET.parse(net_file))
            routes(self.bins, self.traffic_seed, route_file, net_file)
            self.route_hash = hashlib.sha256(route_file.read_bytes()).hexdigest()
            label = uuid4().hex
            traci.start(["sumo", "-n", str(net_file), "-r", str(route_file), "--seed", str(self.traffic_seed),
                         "--tripinfo-output", str(self.directory / "trips.xml"),
                         "--statistic-output", str(self.directory / "stats.xml"),
                         "--collision.action", "warn", "--no-step-log", "true"], label=label)
            self.conn = traci.getConnection(label)
            self.conn.trafficlight.setRedYellowGreenState("gate", self.signal.states[0])
            for arm in ARMS:
                self.conn.edge.subscribe(arm + "_in", [tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_VEHICLE_HALTING_NUMBER])
            for crossing in self.signal.crossings:
                self.conn.edge.subscribe(crossing, [tc.LAST_STEP_PERSON_ID_LIST])
        except BaseException:
            self.close()
            raise
        self.time = self.queue_sum = self.queue_peak = 0
        self.occupied = False
        self.queues = [0] * 4
        self.vehicles = [0] * 4
        return self.observation(), {"traffic_seed": self.traffic_seed, "route_sha256": self.route_hash}

    def observation(self):
        values = [min(x / 40, 1) for x in self.queues + self.vehicles]
        values.extend(int(i == self.signal.phase) for i in range(8))
        values.extend((min(self.signal.elapsed / 45, 1), self.time / self.horizon, float(self.occupied)))
        return np.asarray(values, dtype=np.float32)

    def step(self, action):
        if self.conn is None:
            raise RuntimeError("Call reset() before stepping")
        if not self.action_space.contains(action):
            raise ValueError("Action must be 0 (EW) or 1 (NS)")
        reward = 0
        for _ in range(5):
            if self.signal.advance(action, self.occupied):
                self.conn.trafficlight.setRedYellowGreenState("gate", self.signal.states[self.signal.phase])
            self.conn.simulationStep()
            self.time += 1
            edges = self.conn.edge.getAllSubscriptionResults()
            self.queues = [edges[a + "_in"][tc.LAST_STEP_VEHICLE_HALTING_NUMBER] for a in ARMS]
            self.vehicles = [edges[a + "_in"][tc.LAST_STEP_VEHICLE_NUMBER] for a in ARMS]
            self.occupied = any(edges[c][tc.LAST_STEP_PERSON_ID_LIST] for c in self.signal.crossings)
            queue = sum(self.queues)
            self.queue_peak = max(queue, self.queue_peak)
            self.queue_sum += queue
            reward -= queue
        done = self.time >= self.horizon
        info = {}
        if done:
            self.conn.close()
            self.conn = None
            info = self.metrics()
        return self.observation(), float(reward), False, done, info

    def metrics(self):
        trips = ET.parse(self.directory / "trips.xml")
        stats = ET.parse(self.directory / "stats.xml")
        vehicles = trips.findall("tripinfo")
        walks = trips.findall("personinfo/walk")
        counts = Counter()
        for bucket in self.bins.values():
            for (_, _, mode), count in bucket.items():
                counts[mode] += count

        def mean(elements, attribute):
            return sum(float(e.get(attribute)) for e in elements) / len(elements) if elements else None

        return {"traffic_seed": self.traffic_seed, "route_sha256": self.route_hash,
                "seconds": self.time, "vehicle_arrivals": len(vehicles), "pedestrian_arrivals": len(walks),
                "unfinished_vehicles": counts["car"] + counts["motorcycle"] - len(vehicles),
                "unfinished_pedestrians": counts["pedestrian"] - len(walks),
                "mean_vehicle_wait_s": mean(vehicles, "waitingTime"), "mean_vehicle_trip_s": mean(vehicles, "duration"),
                "mean_pedestrian_wait_s": mean(walks, "waitingTime"), "mean_total_queue": self.queue_sum / self.time,
                "max_total_queue": self.queue_peak, "episode_reward": -self.queue_sum,
                "collisions": int(stats.find("safety").get("collisions")),
                "vehicle_teleports": int(stats.find("teleports").get("total")),
                "pedestrian_teleports": int(stats.find("personTeleports").get("total")) if stats.find("personTeleports") is not None else None}

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None
        if self.tmp is not None:
            self.tmp.cleanup()
            self.tmp = None
