import numpy as np
from stable_baselines3 import PPO

SCENARIO = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"
MODEL_PATH = r"D:\Vasanth\RL_CYB\ppo_out\ppo_cyborg_fixedflat_shaped.zip"


def make_eval_env(max_steps=200):
    import gymnasium as gym
    from CybORG import CybORG
    from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
    from gymnasium import spaces

    class EvalEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.agent = "Blue"
            self.max_steps = max_steps
            self._t = 0

            self._cyborg = CybORG(SCENARIO, "sim")

            # ✅ your repo uses agent= not agent_name=
            self._flat = FixedFlatWrapper(env=self._cyborg, agent=self.agent)

            action_dict = self._cyborg.get_action_space(agent=self.agent)
            self.action_names = list(action_dict.keys())
            self.action_objs = [action_dict[k] for k in self.action_names]

            self.action_space = spaces.Discrete(len(self.action_names))

            _ = self._cyborg.reset()
            obs0 = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()
            self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=obs0.shape, dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            _ = self._cyborg.reset()
            self._t = 0
            obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()
            return obs, {}

        def step(self, action_idx):
            self._t += 1
            action_obj = self.action_objs[int(action_idx)]
            results = self._cyborg.step(agent=self.agent, action=action_obj)

            obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()
            reward = float(results.reward)  # NOTE: this is env reward, not shaped
            terminated = bool(results.done)
            truncated = self._t >= self.max_steps

            info = {
                "action_name": self.action_names[int(action_idx)],
                "raw_info": results.info,
                "raw_obs": results.observation,
            }

            return obs, reward, terminated, truncated, info

    return EvalEnv()


def main():
    env = make_eval_env()
    model = PPO.load(MODEL_PATH)

    episodes = 50
    returns = []
    action_counts = {}
    compromise_hits = 0
    cred_hits = 0

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_return += r
            done = terminated or truncated

            a = info.get("action_name", "unknown")
            action_counts[a] = action_counts.get(a, 0) + 1

            # crude keyword scan (consistent with your shaping idea)
            txt = (str(info.get("raw_obs", "")) + " " + str(info.get("raw_info", ""))).lower()
            if "credential" in txt or "password" in txt or "hash" in txt:
                cred_hits += 1
            if "compromise" in txt or "compromised" in txt or "breach" in txt or "root" in txt or "owned" in txt:
                compromise_hits += 1

        returns.append(ep_return)

    print("\n================= PPO EVAL =================")
    print("Episodes:", episodes)
    print("Avg Return:", float(np.mean(returns)))
    print("Std Return:", float(np.std(returns)))
    print("Min/Max Return:", float(np.min(returns)), float(np.max(returns)))
    print("Credential hits:", cred_hits)
    print("Compromise hits:", compromise_hits)

    print("\nTop actions used:")
    top_actions = sorted(action_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    for name, c in top_actions:
        print(f"{name:15s}  {c}")


if __name__ == "__main__":
    main()

