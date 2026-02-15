import numpy as np
from stable_baselines3 import PPO

SCENARIO = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"
MODEL_PATH = r"D:\Vasanth\RL_CYB\ppo_out\ppo_cyborg_fixedflat_shaped.zip"

# Set True to force event flags during evaluation for debugging.
FORCE_EVENTS = False


def cyber_event_reward(obs_text, next_obs_text, info_text, step_cost=-0.1):
    s = f"{obs_text} {next_obs_text} {info_text}".lower()

    ev_detect = any(k in s for k in ["detect", "alert", "alarm", "ids", "suspicious", "anomal"])
    ev_block  = any(k in s for k in ["block", "blocked", "deny", "dropped", "firewall", "quarantine", "isolat"])
    ev_patch  = any(k in s for k in ["patch", "patched", "update"])

    ev_cred = any(k in s for k in ["credential", "password", "hash", "login", "bruteforce"])
    ev_comp = any(k in s for k in ["compromise", "compromised", "owned", "root", "breach"])
    ev_exfil = any(k in s for k in ["exfil", "exfiltration", "steal", "leak"])
    ev_exploit = any(k in s for k in ["exploit", "payload", "rce", "execute", "injection"])

    reward = step_cost

    # defender positive
    if ev_detect: reward += 2.0
    if ev_block:  reward += 5.0
    if ev_patch:  reward += 3.0

    # attacker negative
    if ev_exploit: reward -= 3.0
    if ev_cred:    reward -= 2.0
    if ev_comp:    reward -= 10.0
    if ev_exfil:   reward -= 20.0

    events = {
        "cred": ev_cred,
        "comp": ev_comp,
        "exfil": ev_exfil,
        "detect": ev_detect,
        "block": ev_block,
        "patch": ev_patch,
        "exploit": ev_exploit,
    }
    return reward, events


def make_env(max_steps=200):
    import gymnasium as gym
    from gymnasium import spaces
    from CybORG import CybORG
    from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper

    class EvalEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self.agent = "Blue"
            self.max_steps = max_steps
            self._t = 0

            self._cyborg = CybORG(SCENARIO, "sim")
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
        '''
        def step(self, action_idx):
            self._t += 1
            action_obj = self.action_objs[int(action_idx)]
            results = self._cyborg.step(agent=self.agent, action=action_obj)

            obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

            terminated = bool(results.done)
            truncated = self._t >= self.max_steps

            info = {
                "action_name": self.action_names[int(action_idx)],
                "raw_obs": results.observation,
                "raw_info": results.info,
                "env_reward": float(results.reward),
                "success": getattr(results, "success", None),
            }
            return obs, float(reward_shaped), terminated, truncated, info

 
        '''
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

            # Debug-only: force a rotating set of events to validate logging/counts.
            if FORCE_EVENTS:
                cycle = ["detect", "block", "patch", "exploit", "cred", "comp", "exfil"]
                forced = cycle[(self._t - 1) % len(cycle)]
                events = {k: (k == forced) for k in events.keys()}

            terminated = bool(results.done)
            truncated = self._t >= self.max_steps

            obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

            info = {
                "action_name": self.action_names[int(action_idx)],
                "reward_env": float(results.reward),
                "reward_shaped": float(reward_shaped),
                "raw_obs": results.observation,
                "raw_info": results.info,
                "success": getattr(results, "success", None),
                **events,
                **{f"ev_{k}": v for k, v in events.items()},
            }
            #print("Action:", self.action_names[int(action_idx)])
            #print("obs_text:", obs_text)
            #print("next_obs_text:", next_obs_text)
            #print("info_text:", info_text)
            #print("reward_shaped:", reward_shaped)
            #print("results:", results)

            return obs, float(reward_shaped), terminated, truncated, info

    return EvalEnv()

def reward_from_success(info, step_cost=-0.1):
    """Shaping using the CybORG Results.success value (stored in info['success'])."""
    r = step_cost
    s = info.get("success", None)
    if s is None:
        return r

    s_str = str(s).lower()
    if "true" in s_str:
        r += 1.0   # success reward
    elif "false" in s_str:
        r -= 0.2   # failed action penalty
    return r

    s_str = str(s).lower()

    if "true" in s_str:
        r += 1.0          # success reward
    elif "false" in s_str:
        r -= 0.2          # failed action penalty
    return r
    
    s = raw_obs.get("success", None)
    if s is None:
       return r

    s_str = str(s).lower()

    if "true" in s_str:
        r += 1.0          # success reward
    elif "false" in s_str:
        r -= 0.2          # failed action penalty
    else:
        r += 0.0          # unknown
    return r


def run_policy(env, policy_name="ppo", model=None, episodes=50):
    shaped_returns = []
    comp_count = 0
    cred_count = 0
    action_hist = {}

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_shaped = 0.0

        while not done:
            if policy_name == "ppo":
                action, _ = model.predict(obs, deterministic=True)
                action = int(action)
            else:
                action = int(env.action_space.sample())

            next_obs, _, terminated, truncated, info = env.step(action)

            # Use env-provided event flags and success shaping (no double-parsing)
            r_success = reward_from_success(info, step_cost=-0.1)
            r_event = 0.0
            events = {
                "cred": info.get("cred", False),
                "comp": info.get("comp", False),
                "exfil": info.get("exfil", False),
                "detect": info.get("detect", False),
                "block": info.get("block", False),
                "patch": info.get("patch", False),
                "exploit": info.get("exploit", False),
            }
            if events.get("detect"):
                r_event += 2.0
            if events.get("block"):
                r_event += 5.0
            if events.get("patch"):
                r_event += 3.0
            if events.get("exploit"):
                r_event -= 3.0
            if events.get("cred"):
                r_event -= 2.0
            if events.get("comp"):
                r_event -= 10.0
            if events.get("exfil"):
                r_event -= 20.0

            r_shaped = r_success + r_event
            ep_shaped += r_shaped

            # event counters
            if events.get("comp", False):
                comp_count += 1
            if events.get("cred", False):
                cred_count += 1

            done = terminated or truncated

            # action histogram
            a = info.get("action_name", "unknown")
            action_hist[a] = action_hist.get(a, 0) + 1

            obs = next_obs
            
            #Debug
            '''
            if ep == 0 and env._t < 3:
                print("\n--- DEBUG STEP ---")
                print("action:", info.get("action_name"))
                print("raw_obs:", info.get("raw_obs"))
                print("raw_info:", info.get("raw_info"))
            '''

        shaped_returns.append(ep_shaped)

    return {
        "avg_shaped_return": float(np.mean(shaped_returns)),
        "std_shaped_return": float(np.std(shaped_returns)),
        "min_return": float(np.min(shaped_returns)),
        "max_return": float(np.max(shaped_returns)),
        "compromise_hits": int(comp_count),
        "credential_hits": int(cred_count),
        "top_actions": sorted(action_hist.items(), key=lambda x: x[1], reverse=True)[:5]
    }

from stable_baselines3 import PPO

def evaluate_policy(model_path, env, n_episodes=5):
    model = PPO.load(model_path)
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        print(f"\n--- Episode {ep+1} ---")
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step}: Action={info['action_name']}, Reward={reward}, Events={[(k,v) for k,v in info.items() if k.startswith('ev_') and v]}")
            done = terminated or truncated
            step += 1

def evaluate_random_policy(env, n_episodes=3):
    for ep in range(n_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        print(f"\n--- Random Policy Episode {ep+1} ---")
        while not done:
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {step}: Action={info['action_name']}, Reward={reward}, Events={[(k,v) for k,v in info.items() if k.startswith('ev_') and v]}")
            done = terminated or truncated
            step += 1

def main():
    env = make_env(max_steps=200)

    print("\nLoading PPO model:", MODEL_PATH)
    model = PPO.load(MODEL_PATH)

    ppo_stats = run_policy(env, policy_name="ppo", model=model, episodes=50)
    rnd_stats = run_policy(env, policy_name="random", model=None, episodes=50)

    print("\n==================== SHAPED RETURN COMPARISON ====================")
    print("PPO  avg/std:", ppo_stats["avg_shaped_return"], ppo_stats["std_shaped_return"])
    print("PPO  min/max:", ppo_stats["min_return"], ppo_stats["max_return"])
    print("PPO  cred/comp hits:", ppo_stats["credential_hits"], ppo_stats["compromise_hits"])
    print("PPO  top actions:", ppo_stats["top_actions"])

    print("\nRND  avg/std:", rnd_stats["avg_shaped_return"], rnd_stats["std_shaped_return"])
    print("RND  min/max:", rnd_stats["min_return"], rnd_stats["max_return"])
    print("RND  cred/comp hits:", rnd_stats["credential_hits"], rnd_stats["compromise_hits"])
    print("RND  top actions:", rnd_stats["top_actions"])
    print("===================================================================")
  
    # Recreate the environment as in training
    evaluate_policy("D:\\Vasanth\\RL_CYB\\ppo_out\\ppo_cyborg_fixedflat_shaped.zip", env, n_episodes=3)
 
    # Evaluate random policy step-by-step
    evaluate_random_policy(env, n_episodes=3)


if __name__ == "__main__":
    main()