import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from xml.etree import ElementTree as ET

from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from nchu_demand import ROOT
from nchu_env import GateEnv, Signal
from nchu_scenario import network


class GateTests(unittest.TestCase):
    def test_transitions_with_pedestrian_interlock(self):
        with TemporaryDirectory() as tmp:
            network(Path(tmp))
            signal = Signal(ET.parse(Path(tmp) / "gate.net.xml"))
        # Repeated switch requests cannot cut the green or clearance short.
        seen = []
        for second in range(30):
            signal.advance(1, occupied=second < 24)
            seen.append(signal.phase)
        self.assertEqual(seen, [0]*10 + [1]*5 + [2]*3 + [3]*6 + [4]*6)
        # Holding green indefinitely must still serve the other group.
        for _ in range(39):
            signal.advance(1, occupied=False)
        self.assertEqual(signal.elapsed, 45)
        for phase, duration in ((5, 5), (6, 3), (7, 2), (0, 1)):
            for _ in range(duration):
                signal.advance(1, occupied=False)
                self.assertEqual(signal.phase, phase)

    def test_gym_and_replay(self):
        env = GateEnv(ROOT / "data/nchu/morning_synthetic.csv")
        try:
            check_env(env, skip_render_check=True)
            trajectories = []
            for _ in range(2):
                obs, info = env.reset(seed=42)
                trajectory = [info["route_sha256"]]
                for action in [1, 1, 1, 0, 0, 1] * 2:
                    obs, reward, terminated, truncated, _ = env.step(action)
                    self.assertFalse(terminated or truncated)
                    self.assertEqual(env.conn.trafficlight.getRedYellowGreenState("gate"),
                                     env.signal.states[env.signal.phase])
                    trajectory.append((obs.tolist(), reward))
                trajectories.append(trajectory)
            self.assertEqual(*trajectories)
            _, info = env.reset(seed=43)
            self.assertNotEqual(trajectories[0][0], info["route_sha256"])
            with self.assertRaises(ValueError):
                env.step(2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
