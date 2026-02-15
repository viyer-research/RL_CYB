import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


SCENARIO = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"


# ----------------------------
# ONLINE reward shaping function
# ----------------------------
def cyber_event_reward(obs_text, next_obs_text, info_text, step_cost=-0.1):
    """
    Online shaped reward based on event keywords in observation/info.
    """
    s = f"{obs_text} {next_obs_text} {info_text}".lower()

    # expanded keywords (better coverage)
    ev_detect = any(k in s for k in ["detect", "alert", "alarm", "ids", "suspicious", "anomal"])
    ev_block  = any(k in s for k in ["block", "blocked", "deny", "dropped", "firewall", "quarantine", "isolat"])
    ev_patch  = any(k in s for k in ["patch", "patched", "update"])

    ev_cred = any(k in s for k in ["credential", "password", "hash", "login", "bruteforce"])
    ev_comp = any(k in s for k in ["compromise", "compromised", "owned", "root", "breach", "session"])
    ev_exfil = any(k in s for k in ["exfil", "exfiltration", "steal", "leak"])
    ev_exploit = any(k in s for k in ["exploit", "payload", "rce", "execute", "injection"])

    # start with step cost
    reward = step_cost

    # defender positive rewards
    if ev_detect:
        reward += 2.0
    if ev_block:
        reward += 5.0
    if ev_patch:
        reward += 3.0

    # attacker negative outcomes
    if ev_exploit:
        reward -= 3.0
    if ev_cred:
        reward -= 2.0
    if ev_comp:
        reward -= 10.0
    if ev_exfil:
        reward -= 20.0

    events = {
        "ev_detect": ev_detect,
        "ev_block": ev_block,
        "ev_patch": ev_patch,
        "ev_cred": ev_cred,
        "ev_compromise": ev_comp,
        "ev_exfil": ev_exfil,
        "ev_exploit": ev_exploit,
    }
    return reward, events


class CybORGFixedFlatShapedEnv(gym.Env):
    """
    PPO environment for CybORG Scenario1b:
    - FixedFlatWrapper observation vector
    - Discrete action mapping
    - ONLINE shaped reward
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_path=SCENARIO, agent="Blue", max_steps=200):
        super().__init__()

        from CybORG import CybORG
        from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper

        self.agent = agent
        self.max_steps = max_steps

        self._cyborg = CybORG(scenario_path, "sim")
        self._flat = FixedFlatWrapper(env=self._cyborg, agent=self.agent)


        # action dictionary -> discrete action ids
        action_dict = self._cyborg.get_action_space(agent=self.agent)
        self.action_names = list(action_dict.keys())
        self.action_objs = [action_dict[k] for k in self.action_names]

        self.action_space = spaces.Discrete(len(self.action_names))

        # determine observation shape
        _ = self._cyborg.reset()
        flat_obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=flat_obs.shape, dtype=np.float32
        )

        self._t = 0
        self._last_events = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        _ = self._cyborg.reset()
        self._t = 0
        self._last_events = {}

        obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()
        info = {}
        return obs, info

    def step(self, action_idx):
        self._t += 1

        action_obj = self.action_objs[int(action_idx)]

        # take CybORG step
        results = self._cyborg.step(agent=self.agent, action=action_obj)

        # raw signals for shaping
        obs_text = str(self._flat.get_observation(self.agent))
        next_obs_text = str(results.observation)
        info_text = str(results.info)

        reward_shaped, events = cyber_event_reward(obs_text, next_obs_text, info_text, step_cost=-0.1)

        terminated = bool(results.done)
        truncated = self._t >= self.max_steps

        obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

        info = {
            "action_name": self.action_names[int(action_idx)],
            "reward_env": float(results.reward),
            "reward_shaped": float(reward_shaped),
            **events
        }

        return obs, float(reward_shaped), terminated, truncated, info


def make_env():
    env = CybORGFixedFlatShapedEnv()
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
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.05,
    )

    total_timesteps = 1_000_000
    model.learn(total_timesteps=total_timesteps)

    out_dir = r"D:\Vasanth\RL_CYB\ppo_out"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "ppo_cyborg_fixedflat_shaped.zip")
    model.save(model_path)

    print("\n✅ PPO saved:", model_path)


if __name__ == "__main__":
    main()
