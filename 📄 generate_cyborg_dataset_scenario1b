import json
import re
import numpy as np
import pandas as pd
from tqdm import tqdm

from CybORG import CybORG


# ----------------------------
# Helpers
# ----------------------------
TRINARY_RE = re.compile(r"TrinaryEnum\.([A-Z]+)")

def safe_str(x):
    try:
        return str(x)
    except:
        return "<unprintable>"

def to_json(obj):
    """
    CybORG observations contain enums that are not valid JSON.
    We store them as a STRING wrapped in JSON-safe form.
    """
    try:
        return json.dumps(obj)
    except TypeError:
        return json.dumps(safe_str(obj))

def extract_trinary(text):
    """
    Convert TrinaryEnum.TRUE/FALSE/UNKNOWN to numeric.
    TRUE=1, FALSE=0, UNKNOWN=-1, not found=-2
    """
    if text is None:
        return -2
    m = TRINARY_RE.search(safe_str(text))
    if not m:
        return -2
    return {"TRUE": 1, "FALSE": 0, "UNKNOWN": -1}.get(m.group(1), -2)

def keyword_counts(text):
    """
    Cheap "richer signal" features from observation/info text.
    """
    s = safe_str(text).lower()
    keys = [
        "success", "fail",
        "scan", "exploit", "compromise",
        "malware", "phish", "credential",
        "alert", "detect",
        "ssh", "ftp", "http", "dns",
        "host", "process", "port",
    ]
    return {f"kw_{k}": s.count(k) for k in keys}


def main():
    # ----------------------------
    # Config (tune these)
    # ----------------------------
    num_episodes = 100     # increase later: 200 / 500
    max_steps = 200        # longer horizon = more meaningful RL data
    agent_name = "Blue"

    scenario_path = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"

    # ----------------------------
    # Create environment
    # ----------------------------
    env = CybORG(scenario_path, "sim")

    rows = []

    for ep in tqdm(range(num_episodes), desc="Episodes"):
        obs = env.reset()
        done = False

        for t in range(max_steps):
            action_space = env.get_action_space(agent=agent_name)
            action_names = list(action_space.keys())

            act_name = np.random.choice(action_names)
            action = action_space[act_name]

            # ✅ CybORG returns a Results object
            results = env.step(agent=agent_name, action=action)

            next_obs = results.observation
            reward = results.reward
            done = results.done
            info = results.info

            # --- richer signals ---
            obs_str = safe_str(obs)
            next_obs_str = safe_str(next_obs)
            info_str = safe_str(info)

            obs_success = extract_trinary(obs_str)
            next_obs_success = extract_trinary(next_obs_str)

            obs_kw = keyword_counts(obs_str)
            next_obs_kw = keyword_counts(next_obs_str)
            info_kw = keyword_counts(info_str)

            row = {
                "episode": ep,
                "timestep": t,
                "action_name": act_name,
                "reward": float(reward),
                "done": bool(done),

                # raw observation storage
                "obs_json": to_json(obs),
                "next_obs_json": to_json(next_obs),
                "info_json": to_json(info),

                # numeric success flags
                "obs_success": int(obs_success),
                "next_obs_success": int(next_obs_success),

                # basic length signals
                "obs_len": len(obs_str),
                "next_obs_len": len(next_obs_str),
                "info_len": len(info_str),
            }

            # add keyword counts
            for k, v in obs_kw.items():
                row[f"obs_{k}"] = v
            for k, v in next_obs_kw.items():
                row[f"next_obs_{k}"] = v
            for k, v in info_kw.items():
                row[f"info_{k}"] = v

            rows.append(row)

            obs = next_obs
            if done:
                break

    df = pd.DataFrame(rows)

    out_csv = "cyborg_rl_dataset_scenario1b.csv"
    df.to_csv(out_csv, index=False)

    print("\n✅ Dataset saved:", out_csv)
    print("✅ Transitions collected:", len(df))
    print("✅ Unique actions:", df["action_name"].nunique())
    print("✅ Done rate:", df["done"].mean())
    print(df.head(5))


if __name__ == "__main__":
    main()
