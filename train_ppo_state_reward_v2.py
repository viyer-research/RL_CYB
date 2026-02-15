"""
PPO Training with STATE-BASED reward and strong diversity enforcement.
Uses custom gym.Env class (not wrapper) for compatibility.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Configuration - use absolute paths
CYBORG_PATH = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG"

#CYBORG_PATH = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG"
sys.path.insert(0, CYBORG_PATH)

SCENARIO_PATH = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml"
OUTPUT_DIR = r"D:\Vasanth\RL_CYB\ppo_out"


class CybORGDiverseEnv(gym.Env):
    """
    CybORG environment with strong diversity enforcement:
    - Uses CybORG's native reward
    - Heavy penalty for repeating same action
    - Bonus for using different action types
    - Exploration bonus for rarely-used actions
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_path, agent="Blue", max_steps=200):
        super().__init__()

        from CybORG import CybORG
        from CybORG.Agents import B_lineAgent
        from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
        from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
        from CybORG.Agents.Wrappers.BlueTableWrapper import BlueTableWrapper

        self.agent = agent
        self.max_steps = max_steps
        self.scenario_path = scenario_path

        # Create CybORG with B_lineAgent as Red
        self._cyborg = CybORG(scenario_path, "sim", agents={'Red': B_lineAgent})
        
        # Chain wrappers properly
        self._table = BlueTableWrapper(env=self._cyborg, output_mode='vector')
        self._enum = EnumActionWrapper(env=self._table)
        self._flat = FixedFlatWrapper(env=self._cyborg, agent=self.agent)

        # Get the properly enumerated actions
        action_space_size = self._enum.action_space_change(
            self._cyborg.get_action_space(agent=self.agent)
        )
        self.possible_actions = self._enum.possible_actions
        
        # Build readable action names
        self.action_names = [str(act) for act in self.possible_actions]
        print(f"[INFO] Blue agent has {len(self.possible_actions)} discrete actions")
        
        # Show first few actions
        for i, name in enumerate(self.action_names[:10]):
            print(f"  Action {i}: {name}")

        self.action_space = spaces.Discrete(len(self.possible_actions))

        # Determine observation shape
        _ = self._cyborg.reset()
        flat_obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=flat_obs.shape, dtype=np.float32
        )

        self._t = 0
        self._last_action = None
        self._last_action_name = ""
        self._action_history = []
        self._action_type_history = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        _ = self._cyborg.reset()
        self._t = 0
        self._last_action = None
        self._last_action_name = ""
        self._action_history = []
        self._action_type_history = []

        obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()
        return obs, {}

    def step(self, action):
        action = int(action)
        action_obj = self.possible_actions[action]
        action_name = str(action_obj)
        action_type = action_name.split()[0]  # e.g., "Restore", "Monitor", etc.

        # Execute in CybORG
        result = self._cyborg.step(agent=self.agent, action=action_obj)
        
        # Get CybORG's native reward (state-based)
        native_reward = result.reward

        # === DIVERSITY ENFORCEMENT ===
        diversity_bonus = 0.0
        
        # 1. HEAVY penalty for repeating exact same action
        if self._last_action is not None and action == self._last_action:
            diversity_bonus -= 0.5
        
        # 2. Bonus for using a different action TYPE than recent history
        recent_types = self._action_type_history[-5:] if self._action_type_history else []
        if action_type not in recent_types:
            diversity_bonus += 0.3
        
        # 3. Exploration bonus for rarely-used actions in this episode
        action_count = self._action_history.count(action)
        if action_count < 2:
            diversity_bonus += 0.2
        
        # === FINAL REWARD ===
        total_reward = native_reward + diversity_bonus

        # Update state
        self._last_action = action
        self._last_action_name = action_name
        self._action_history.append(action)
        self._action_type_history.append(action_type)
        
        # Keep history bounded
        if len(self._action_history) > 100:
            self._action_history = self._action_history[-100:]
            self._action_type_history = self._action_type_history[-100:]

        self._t += 1
        obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()
        
        terminated = False
        truncated = self._t >= self.max_steps

        info = {
            'native_reward': native_reward,
            'diversity_bonus': diversity_bonus,
            'action_name': action_name,
            'action_type': action_type,
        }

        return obs, total_reward, terminated, truncated, info


class DiversityCallback(BaseCallback):
    """Monitor action diversity during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_counts = {}
        self.total_actions = 0
        
    def _on_step(self) -> bool:
        infos = self.locals.get('infos', [{}])
        if infos and 'action_name' in infos[0]:
            action = infos[0]['action_name']
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            self.total_actions += 1
        
        # Log every 10000 steps
        if self.num_timesteps % 10000 == 0 and self.action_counts:
            unique = len(self.action_counts)
            most_common = max(self.action_counts.items(), key=lambda x: x[1])
            most_common_pct = most_common[1] / self.total_actions * 100
            
            print(f"\n[Step {self.num_timesteps}] Unique actions: {unique}, "
                  f"Most common: {most_common[0]} ({most_common_pct:.1f}%)")
            
            self.logger.record("diversity/unique_actions", unique)
            self.logger.record("diversity/most_common_pct", most_common_pct)
            
        return True


def main():
    print("=" * 60)
    print("PPO Training with Diversity Enforcement")
    print("=" * 60)
    print("\nDiversity mechanisms:")
    print("  1. -0.5 penalty for repeating same action")
    print("  2. +0.3 bonus for new action type")
    print("  3. +0.2 bonus for rarely-used actions")
    print("  4. HIGH entropy coefficient (0.1) - constant")
    print("=" * 60)
    
    # Create environment
    env = DummyVecEnv([lambda: CybORGDiverseEnv(SCENARIO_PATH)])
    
    # PPO with HIGH constant entropy
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.1,  # HIGH entropy, never decays
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=OUTPUT_DIR,
    )
    
    print("\nStarting training (1M steps)...")
    callback = DiversityCallback()
    
    model.learn(
        total_timesteps=1_000_000,
        callback=callback,
        tb_log_name="PPO_diverse_v1",
        progress_bar=True
    )
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "ppo_diverse_v1.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Quick evaluation
    print("\n" + "=" * 60)
    print("Quick Evaluation (3 episodes, first 50 steps each)")
    print("=" * 60)
    
    eval_env = CybORGDiverseEnv(SCENARIO_PATH)
    action_counts = {}
    
    for ep in range(3):
        obs, _ = eval_env.reset()
        total_reward = 0
        ep_actions = []
        
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            
            action_name = info.get('action_name', f'Action_{action}')
            ep_actions.append(action_name)
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            if done or truncated:
                break
        
        unique = len(set(ep_actions))
        print(f"Episode {ep+1}: Reward={total_reward:.1f}, Unique actions={unique}")
    
    print("\nAction distribution (top 10):")
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {action}: {count} ({count/total*100:.1f}%)")
    
    env.close()
    print("\n" + "=" * 60)
    print(f"Training complete! Model: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
