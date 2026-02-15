import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


#SCENARIO = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"
SCENARIO = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml"


# ----------------------------
# ONLINE reward shaping function (improved)
# ----------------------------
def cyber_event_reward(obs_text, next_obs_text, info_text, step_cost=0.0):
    """
    Improved shaped reward based on event keywords in observation/info.
    Positive events are more rewarding, negative events less punishing.
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

    # defender positive rewards (increased)
    if ev_detect:
        reward += 20.0
    if ev_block:
        reward += 40.0
    if ev_patch:
        reward += 30.0

    # attacker negative outcomes (reduced)
    if ev_exploit:
        reward -= 1.0
    if ev_cred:
        reward -= 0.5
    if ev_comp:
        reward -= 2.0
    if ev_exfil:
        reward -= 5.0

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
    - Discrete action mapping using EnumActionWrapper for proper Blue actions
    - ONLINE shaped reward
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_path=SCENARIO, agent="Blue", max_steps=200):
        super().__init__()

        from CybORG import CybORG
        from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
        from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
        from CybORG.Agents.Wrappers.BlueTableWrapper import BlueTableWrapper

        self.agent = agent
        self.max_steps = max_steps

        self._cyborg = CybORG(scenario_path, "sim")
        
        # Chain wrappers properly: BlueTableWrapper -> EnumActionWrapper for proper action space
        self._table = BlueTableWrapper(env=self._cyborg, output_mode='vector')
        self._enum = EnumActionWrapper(env=self._table)
        self._flat = FixedFlatWrapper(env=self._cyborg, agent=self.agent)

        # Get the properly enumerated actions from EnumActionWrapper
        action_space_size = self._enum.action_space_change(
            self._cyborg.get_action_space(agent=self.agent)
        )
        self.possible_actions = self._enum.possible_actions
        
        # Build readable action names
        self.action_names = [str(act) for act in self.possible_actions]
        print(f"[INFO] Blue agent has {len(self.possible_actions)} discrete actions:")
        for i, name in enumerate(self.action_names[:10]):
            print(f"  [{i}] {name}")
        if len(self.action_names) > 10:
            print(f"  ... and {len(self.action_names) - 10} more")

        self.action_space = spaces.Discrete(len(self.possible_actions))

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

        # Use the properly enumerated action object
        action_obj = self.possible_actions[int(action_idx)]

        # take CybORG step
        results = self._cyborg.step(agent=self.agent, action=action_obj)

        # raw signals for shaping
        obs_text = str(self._flat.get_observation(self.agent))
        next_obs_text = str(results.observation)
        info_text = str(results.info)

        reward_shaped, events = cyber_event_reward(obs_text, next_obs_text, info_text, step_cost=-0.1)

        terminated = bool(results.done)
        truncated = self._t >= self.max_steps

        # >>> ADD THIS BLOCK <<<
        if terminated or truncated:
           reward_shaped += 5.0  # episode completion bonus

        obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

        info = {
            "action_name": self.action_names[int(action_idx)],
            "reward_env": float(results.reward),
            "reward_shaped": float(reward_shaped),
            **events
        }

        return obs, float(reward_shaped), terminated, truncated, info
    
'''
def make_env():
    env = CybORGFixedFlatShapedEnv()
    env = Monitor(env)
    return env
'''
def make_env():
    def _init():
        env = CybORGFixedFlatShapedEnv()
        env = Monitor(env)
        return env
    return _init

def main():
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([make_env() for _ in range(4)])  # 4 parallel environments

    model = PPO(
        "MlpPolicy",
        venv,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.05,  # Increased from 0.01 for better exploration
        verbose=1,
        tensorboard_log=r"D:\Vasanth\RL_CYB\cage-challenge-1\ppo_cyborg_tensorboard"
    )
    model.learn(total_timesteps=1_000_000, tb_log_name="PPO")

    out_dir = r"D:\Vasanth\RL_CYB\ppo_out"
    os.makedirs(out_dir, exist_ok=True)

    model_path = os.path.join(out_dir, "ppo_cyborg_fixedflat_shaped_vulnerable.zip")
    model.save(model_path)

    print("\n✅ PPO saved:", model_path)


if __name__ == "__main__":
    main()
