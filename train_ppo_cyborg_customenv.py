import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


SCENARIO = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"


class CybORGPPOEnv(gym.Env):
    """
    Minimal Gymnasium wrapper for CybORG that works with Stable-Baselines3 PPO.

    - Discrete action space: index into CybORG get_action_space() dict
    - Observation: simple numeric vector (you can improve later)
    - Reward: uses CybORG reward (you can reward-shape later)
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_path=SCENARIO, agent="Blue", max_steps=200):
        super().__init__()
        from CybORG import CybORG

        self.agent = agent
        self.max_steps = max_steps

        self._cyborg = CybORG(scenario_path, "sim")
        self._t = 0

        # --- Build action mapping (dict -> discrete ids) ---
        action_dict = self._cyborg.get_action_space(agent=self.agent)
        self.action_names = list(action_dict.keys())
        self.action_objs = [action_dict[k] for k in self.action_names]

        self.action_space = spaces.Discrete(len(self.action_names))

        # --- Observation space ---
        # CybORG observations are complex objects, so we create a simple feature vector.
        # Start very simple: [timestep, last_reward, last_done_flag]
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=(3,), dtype=np.float32
        )

        self._last_reward = 0.0
        self._last_done = 0.0

    def _obs_to_vec(self):
        return np.array([self._t, self._last_reward, self._last_done], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        _ = self._cyborg.reset()
        self._t = 0
        self._last_reward = 0.0
        self._last_done = 0.0

        obs = self._obs_to_vec()
        info = {}
        return obs, info

    def step(self, action_idx):
        self._t += 1

        action_obj = self.action_objs[int(action_idx)]
        results = self._cyborg.step(agent=self.agent, action=action_obj)

        reward = float(results.reward)
        done = bool(results.done)

        # force truncate at horizon
        truncated = self._t >= self.max_steps
        terminated = done

        self._last_reward = reward
        self._last_done = 1.0 if (terminated or truncated) else 0.0

        obs = self._obs_to_vec()

        info = {
            "action_name": self.action_names[int(action_idx)],
            "raw_observation": results.observation,
            "raw_info": results.info,
        }

        return obs, reward, terminated, truncated, info


def make_env():
    env = CybORGPPOEnv()
    env = Monitor(env)
    return env


def main():
    venv = DummyVecEnv([make_env])

    model = PPO(
        "MlpPolicy",
        venv,
        verbose=1,
        device="auto",
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
    )

    model.learn(total_timesteps=200_000)

    out_dir = r"D:\Vasanth\RL_CYB\ppo_out"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ppo_cyborg_customenv.zip")
    model.save(model_path)

    print("\n✅ PPO saved:", model_path)


if __name__ == "__main__":
    main()
