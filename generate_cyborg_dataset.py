import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from CybORG import CybORG


def to_json(obj):
    try:
        return json.dumps(obj)
    except TypeError:
        return json.dumps(str(obj))


def main():
    # -------- Config --------
    num_episodes = 30
    max_steps = 50
    agent_name = "Blue"

    scenario_path = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1.yaml"
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

            rows.append({
                "episode": ep,
                "timestep": t,
                "action_name": act_name,
                "reward": float(reward),
                "done": bool(done),
                "obs_json": to_json(obs),
                "next_obs_json": to_json(next_obs),
                "info_json": to_json(info),
            })

            obs = next_obs
            if done:
                break

    df = pd.DataFrame(rows)
    df.to_csv("cyborg_rl_dataset.csv", index=False)

    print("\n✅ Dataset saved: cyborg_rl_dataset.csv")
    print("✅ Transitions collected:", len(df))
    print(df.head())


if __name__ == "__main__":
    main()
