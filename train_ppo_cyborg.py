import os
import numpy as np

# --- Stable Baselines 3 ---
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

SCENARIO = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"

def make_env():
    from CybORG import CybORG

    # ✅ Use your scenario
    scenario_path = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"

    cyborg = CybORG(scenario_path, "sim")

    # ✅ Import wrappers from your repo
    from CybORG.Agents.Wrappers.OpenAIGymWrapper import OpenAIGymWrapper
    from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper

    # ✅ Important: OpenAIGymWrapper expects env=cyborg argument (not positional)
    env = OpenAIGymWrapper(env=cyborg, agent_name="Blue")

    # ✅ Flatten observation into vector
    env = FixedFlatWrapper(env=env, agent_name="Blue")

    # ✅ SB3 monitor wrapper
    from stable_baselines3.common.monitor import Monitor
    env = Monitor(env)

    return env


def main():
    # Vectorize env (SB3 expects VecEnv)
    venv = DummyVecEnv([make_env])

    # PPO policy: MLP works since obs is flat vector
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gamma=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        device="auto",
    )

    # Train
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    # Save
    out_dir = r"D:\Vasanth\RL_CYB\ppo_out"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ppo_cyborg_scenario1b.zip")
    model.save(model_path)
    print(f"\n✅ Saved model: {model_path}")

    # Quick evaluation rollouts
    obs = venv.reset()
    ep_reward = 0.0
    for _ in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = venv.step(action)
        ep_reward += float(reward[0])
        if bool(done[0]):
            print("Episode ended. Return:", ep_reward)
            ep_reward = 0.0
            obs = venv.reset()

    print("✅ Eval finished.")


if __name__ == "__main__":
    main()
