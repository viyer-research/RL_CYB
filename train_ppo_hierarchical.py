"""
Hierarchical PPO: 
1. High-level policy picks ACTION TYPE (Monitor, Analyse, Remove, Restore)
2. Low-level policy picks TARGET HOST based on action type

This reduces action space from 54 to (4 types + 13 hosts) = more learnable.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict

# Add CybORG to path
sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper

SCENARIO_PATH = r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml'
OUTPUT_DIR = r"D:\Vasanth\RL_CYB\ppo_out"

# Define action types and hosts
ACTION_TYPES = ['Monitor', 'Analyse', 'Remove', 'Restore']  # Skip Sleep, Misinform
HOSTS = [
    'Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2',
    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
    'User0', 'User1', 'User2', 'User3', 'User4'
]


class HierarchicalCybORGEnv(gym.Env):
    """
    Hierarchical action space:
    - Action is a tuple: (action_type_idx, host_idx)
    - Encoded as single int: action_type_idx * len(HOSTS) + host_idx
    - Special case: Monitor has no host target (use any host_idx)
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, scenario_path, max_steps=200):
        super().__init__()
        
        self.max_steps = max_steps
        self.n_action_types = len(ACTION_TYPES)
        self.n_hosts = len(HOSTS)
        
        # Create CybORG
        self._cyborg = CybORG(scenario_path, "sim", agents={'Red': B_lineAgent})
        self._table = BlueTableWrapper(env=self._cyborg, output_mode='vector')
        self._enum = EnumActionWrapper(env=self._table)
        self._flat = FixedFlatWrapper(env=self._cyborg, agent='Blue')
        
        # Get all possible actions
        self._enum.action_space_change(self._cyborg.get_action_space('Blue'))
        self._all_actions = self._enum.possible_actions
        
        # Build mapping from (type, host) -> action object
        self._action_map = {}
        self._build_action_map()
        
        # Hierarchical action space: type * hosts 
        # For Monitor, host doesn't matter but we still include it
        self.action_space = spaces.Discrete(self.n_action_types * self.n_hosts)
        
        # Get observation shape
        self._cyborg.reset()
        flat_obs = np.array(self._flat.get_observation('Blue'), dtype=np.float32).flatten()
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=flat_obs.shape, dtype=np.float32
        )
        
        print(f"[INFO] Hierarchical action space: {self.n_action_types} types x {self.n_hosts} hosts = {self.action_space.n} actions")
        print(f"  Action types: {ACTION_TYPES}")
        print(f"  Hosts: {HOSTS}")
        
        self._t = 0
        self._last_action_type = None
        self._last_host = None
        
    def _build_action_map(self):
        """Map (action_type, host) -> CybORG action object."""
        for action in self._all_actions:
            action_str = str(action)
            
            for type_idx, action_type in enumerate(ACTION_TYPES):
                if action_str.startswith(action_type):
                    if action_type == 'Monitor':
                        # Monitor has no host - map all host indices to it
                        for host_idx in range(self.n_hosts):
                            self._action_map[(type_idx, host_idx)] = action
                    else:
                        # Find which host
                        for host_idx, host in enumerate(HOSTS):
                            if host in action_str:
                                self._action_map[(type_idx, host_idx)] = action
                                break
                    break
        
        # Fill gaps with Sleep action (fallback)
        sleep_action = None
        for action in self._all_actions:
            if 'Sleep' in str(action):
                sleep_action = action
                break
        
        for type_idx in range(self.n_action_types):
            for host_idx in range(self.n_hosts):
                if (type_idx, host_idx) not in self._action_map:
                    self._action_map[(type_idx, host_idx)] = sleep_action
    
    def _decode_action(self, action):
        """Decode flat action into (type_idx, host_idx)."""
        type_idx = action // self.n_hosts
        host_idx = action % self.n_hosts
        return type_idx, host_idx
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._cyborg.reset()
        self._t = 0
        self._last_action_type = None
        self._last_host = None
        
        obs = np.array(self._flat.get_observation('Blue'), dtype=np.float32).flatten()
        return obs, {}
    
    def step(self, action):
        action = int(action)
        type_idx, host_idx = self._decode_action(action)
        
        # Get actual CybORG action
        action_obj = self._action_map.get((type_idx, host_idx))
        action_name = str(action_obj)
        action_type = ACTION_TYPES[type_idx]
        host_name = HOSTS[host_idx]
        
        # Execute in CybORG
        result = self._cyborg.step(agent='Blue', action=action_obj)
        native_reward = result.reward
        
        # Small bonus for using different action types
        type_bonus = 0.0
        if self._last_action_type is not None and type_idx != self._last_action_type:
            type_bonus = 0.1  # Small bonus for switching action type
        
        total_reward = native_reward + type_bonus
        
        # Update state
        self._last_action_type = type_idx
        self._last_host = host_idx
        self._t += 1
        
        obs = np.array(self._flat.get_observation('Blue'), dtype=np.float32).flatten()
        terminated = False
        truncated = self._t >= self.max_steps
        
        info = {
            'native_reward': native_reward,
            'action_type': action_type,
            'host': host_name,
            'action_name': action_name,
        }
        
        return obs, total_reward, terminated, truncated, info


class HierarchicalCallback(BaseCallback):
    """Track action type and host distributions."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.type_counts = defaultdict(int)
        self.host_counts = defaultdict(int)
        self.total = 0
        
    def _on_step(self):
        infos = self.locals.get('infos', [{}])
        if infos and 'action_type' in infos[0]:
            self.type_counts[infos[0]['action_type']] += 1
            self.host_counts[infos[0]['host']] += 1
            self.total += 1
        
        if self.num_timesteps % 20000 == 0 and self.total > 0:
            print(f"\n[Step {self.num_timesteps}] Action Type Distribution:")
            for t in ACTION_TYPES:
                pct = self.type_counts[t] / self.total * 100
                print(f"  {t}: {pct:.1f}%")
            
            # Log to tensorboard
            for t in ACTION_TYPES:
                self.logger.record(f"action_type/{t}", self.type_counts[t] / self.total * 100)
        
        return True


def evaluate_policy(model, n_episodes=5, deterministic=True):
    """Evaluate the hierarchical policy."""
    
    env = HierarchicalCybORGEnv(SCENARIO_PATH)
    
    mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
    print(f"\n{'='*60}")
    print(f"Evaluating Hierarchical Policy ({mode})")
    print(f"{'='*60}")
    
    type_counts = defaultdict(int)
    host_counts = defaultdict(int)
    action_counts = defaultdict(int)
    total_rewards = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_actions = []
        
        for step in range(200):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, truncated, info = env.step(action)
            
            ep_reward += info['native_reward']  # Use native reward for comparison
            type_counts[info['action_type']] += 1
            host_counts[info['host']] += 1
            action_counts[info['action_name']] += 1
            ep_actions.append(info['action_name'])
            
            if done or truncated:
                break
        
        unique = len(set(ep_actions))
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward={ep_reward:.1f}, Unique actions={unique}")
    
    print(f"\nAverage Reward: {np.mean(total_rewards):.1f} (+/- {np.std(total_rewards):.1f})")
    
    print("\nAction TYPE distribution:")
    total = sum(type_counts.values())
    for t in ACTION_TYPES:
        print(f"  {t}: {type_counts[t]} ({type_counts[t]/total*100:.1f}%)")
    
    print("\nAction distribution (top 15):")
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {action}: {count} ({count/total*100:.1f}%)")
    
    return np.mean(total_rewards)


def main():
    print("=" * 60)
    print("Hierarchical PPO Training")
    print("=" * 60)
    print("\nAction decomposition:")
    print("  1. Pick ACTION TYPE: Monitor, Analyse, Remove, Restore")
    print("  2. Pick TARGET HOST: 13 possible hosts")
    print("  Total: 4 x 13 = 52 actions (vs 54 flat)")
    print("=" * 60)
    
    # Create environment
    env = DummyVecEnv([lambda: HierarchicalCybORGEnv(SCENARIO_PATH)])
    
    # PPO with moderate entropy
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
        ent_coef=0.05,  # Moderate entropy
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=OUTPUT_DIR,
    )
    
    print("\nStarting training (500K steps)...")
    callback = HierarchicalCallback()
    
    model.learn(
        total_timesteps=500_000,
        callback=callback,
        tb_log_name="PPO_hierarchical_v1",
        progress_bar=True
    )
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "ppo_hierarchical_v1.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Evaluate
    reward_det = evaluate_policy(model, n_episodes=5, deterministic=True)
    reward_stoch = evaluate_policy(model, n_episodes=5, deterministic=False)
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Deterministic reward: {reward_det:.1f}")
    print(f"  Stochastic reward: {reward_stoch:.1f}")
    print(f"  Model: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
