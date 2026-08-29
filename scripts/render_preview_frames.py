"""Render milestone frames from the selected learned PPO policy."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import imageio.v2 as imageio
import mujoco
from stable_baselines3 import PPO

from himalaya_env import G1SelfArrestEnv


def render_snapshots() -> None:
    output = Path("videos/snapshots")
    output.mkdir(parents=True, exist_ok=True)
    model = PPO.load("models/ppo_self_arrest/g1_self_arrest_final.zip", device="cpu")
    env = G1SelfArrestEnv(randomize_reset=False)
    observation, _ = env.reset(options={"randomize": False, "speed": 4.5})
    renderer = mujoco.Renderer(env.model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = env.pelvis_body_id
    camera.distance = 1.75
    camera.azimuth = 55
    camera.elevation = -18

    milestones = {0, 100, 200}
    for step in range(env.max_episode_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, _, terminated, truncated, info = env.step(action)
        if step in milestones or terminated or truncated:
            renderer.update_scene(env.data, camera=camera)
            imageio.imwrite(output / f"learned_policy_{step:03d}.png", renderer.render())
            print(f"step={step:03d} speed={info['v_slope']:.3f} success={info['success']}")
        if terminated or truncated:
            break
    renderer.close()
    env.close()


if __name__ == "__main__":
    render_snapshots()
