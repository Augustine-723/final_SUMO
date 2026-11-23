"""Exercise the single-agent Gymnasium API for one short synthetic episode."""
import math

from build_scenario import NET, ROUTES, build


def main():
    from sumo_rl import SumoEnvironment

    build()
    env = SumoEnvironment(net_file=str(NET), route_file=str(ROUTES), single_agent=True,
                          sumo_seed=42, num_seconds=60, delta_time=5, min_green=5, yellow_time=3)
    try:
        obs, _ = env.reset(seed=42)
        if not env.observation_space.contains(obs) or env.action_space.n < 2:
            raise RuntimeError('Unexpected initial observation or action space')
        phases = set()
        for step in range(1, 13):
            action = ((step - 1) // 3) % env.action_space.n
            obs, reward, terminated, truncated, _ = env.step(int(action))
            phases.add(int(obs[:env.action_space.n].argmax()))
            if not env.observation_space.contains(obs) or not math.isfinite(reward):
                raise RuntimeError(f'Invalid transition at step {step}')
            if terminated or truncated:
                break
        if step != 12 or not (terminated or truncated) or len(phases) < 2:
            raise RuntimeError('Expected a 60-second episode with an actual green-phase change')
        print(f'decisions={step} phases={sorted(phases)} obs_shape={obs.shape} last_reward={reward}')
    finally:
        env.close()


if __name__ == '__main__':
    main()
