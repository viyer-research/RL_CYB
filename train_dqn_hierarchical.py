"""
DQN Training for CybORG Blue Agent with Hierarchical Action Space.

DQN learns Q(s,a) directly - the expected return for each action in each state.
This should produce state-dependent action selection without the flat distribution
problem seen with PPO.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
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
ACTION_TYPES = ['Monitor', 'Analyse', 'Remove', 'Restore']
HOSTS = [
    'Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2',
    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
    'User0', 'User1', 'User2', 'User3', 'User4'
]


class HierarchicalCybORGEnv(gym.Env):
    """
    Hierarchical action space for CybORG.
    Action = action_type_idx * n_hosts + host_idx
    """
    
    metadata = {"render_modes": []}
    
    def __init__(self, scenario_path, max_steps=100):
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
        
        # Hierarchical action space
        self.action_space = spaces.Discrete(self.n_action_types * self.n_hosts)
        
        # Get observation shape
        self._cyborg.reset()
        flat_obs = np.array(self._flat.get_observation('Blue'), dtype=np.float32).flatten()
        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=flat_obs.shape, dtype=np.float32
        )
        
        print(f"[INFO] Hierarchical action space: {self.n_action_types} types x {self.n_hosts} hosts = {self.action_space.n} actions")
        
        self._t = 0
        
    def _build_action_map(self):
        """Map (action_type, host) -> CybORG action object."""
        for action in self._all_actions:
            action_str = str(action)
            
            for type_idx, action_type in enumerate(ACTION_TYPES):
                if action_str.startswith(action_type):
                    if action_type == 'Monitor':
                        for host_idx in range(self.n_hosts):
                            self._action_map[(type_idx, host_idx)] = action
                    else:
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
        obs = np.array(self._flat.get_observation('Blue'), dtype=np.float32).flatten()
        return obs, {}
    
    def step(self, action):
        action = int(action)
        type_idx, host_idx = self._decode_action(action)
        
        action_obj = self._action_map.get((type_idx, host_idx))
        
        # Execute in CybORG
        result = self._cyborg.step(agent='Blue', action=action_obj)
        reward = result.reward  # Use native CybORG reward
        
        self._t += 1
        done = self._t >= self.max_steps
        
        obs = np.array(self._flat.get_observation('Blue'), dtype=np.float32).flatten()
        return obs, reward, done, False, {'action_name': str(action_obj)}


class DQNLoggingCallback(BaseCallback):
    """Callback for logging DQN training progress."""
    
    def __init__(self, eval_env, log_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._evaluate()
        return True
    
    def _evaluate(self):
        """Run evaluation episodes."""
        action_counts = defaultdict(int)
        total_rewards = []
        
        for _ in range(5):
            obs, _ = self.eval_env.reset()
            ep_reward = 0
            
            for step in range(100):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _, info = self.eval_env.step(action)
                ep_reward += reward
                
                action_name = info.get('action_name', str(action))
                action_counts[action_name] += 1
                
                if done:
                    break
            
            total_rewards.append(ep_reward)
        
        mean_reward = np.mean(total_rewards)
        unique_actions = len(action_counts)
        
        # Get top action
        if action_counts:
            top_action = max(action_counts.items(), key=lambda x: x[1])
            top_pct = top_action[1] / sum(action_counts.values()) * 100
        else:
            top_action = ("None", 0)
            top_pct = 0
        
        print(f"\n[Step {self.n_calls}] Eval: Reward={mean_reward:.1f}, "
              f"Unique={unique_actions}, Top={top_action[0]} ({top_pct:.1f}%)")
        
        # Log action type distribution
        type_counts = defaultdict(int)
        for action_name, count in action_counts.items():
            for action_type in ACTION_TYPES:
                if action_name.startswith(action_type):
                    type_counts[action_type] += count
                    break
        
        total = sum(type_counts.values()) or 1
        print(f"  Action types: " + ", ".join(
            f"{t}:{type_counts[t]/total*100:.0f}%" for t in ACTION_TYPES
        ))


def make_env(scenario_path, max_steps=100):
    def _init():
        return HierarchicalCybORGEnv(scenario_path, max_steps=max_steps)
    return _init


def main():
    print("="*60)
    print("  DQN Training for CybORG Blue Agent")
    print("="*60)
    
    # Create training environment
    print("\nCreating training environment...")
    env = DummyVecEnv([make_env(SCENARIO_PATH, max_steps=100)])
    
    # Create eval environment
    eval_env = HierarchicalCybORGEnv(SCENARIO_PATH, max_steps=100)
    
    # DQN hyperparameters
    # Key differences from PPO:
    # - Uses epsilon-greedy exploration (not entropy)
    # - Learns Q-values directly
    # - More sample efficient with replay buffer
    
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=100000,           # Replay buffer size
        learning_starts=10000,        # Steps before learning starts
        batch_size=64,
        tau=0.005,                    # Soft update coefficient
        gamma=0.99,                   # Discount factor
        train_freq=4,                 # Update every 4 steps
        gradient_steps=1,
        target_update_interval=1000,  # Hard update target network
        exploration_fraction=0.3,     # Fraction of training for exploration decay
        exploration_initial_eps=1.0,  # Start with 100% random
        exploration_final_eps=0.05,   # End with 5% random
        policy_kwargs=dict(
            net_arch=[256, 256, 128]   # Larger network
        ),
        verbose=1,
        tensorboard_log=os.path.join(OUTPUT_DIR, "dqn_logs")
    )
    
    print(f"\nModel architecture: MlpPolicy with net_arch=[256, 256, 128]")
    print(f"Exploration: eps 1.0 -> 0.05 over 30% of training")
    print(f"Replay buffer: 100K transitions")
    
    # Callback
    callback = DQNLoggingCallback(eval_env, log_freq=20000)
    
    # Train
    total_timesteps = 500000
    print(f"\nStarting DQN training for {total_timesteps} steps...")
    print("="*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save
    model_path = os.path.join(OUTPUT_DIR, "dqn_hierarchical_v1.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("  Final Evaluation")
    print("="*60)
    
    action_counts = defaultdict(int)
    total_rewards = []
    
    for ep in range(10):
        obs, _ = eval_env.reset()
        ep_reward = 0
        ep_actions = []
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, info = eval_env.step(action)
            ep_reward += reward
            
            action_name = info.get('action_name', str(action))
            action_counts[action_name] += 1
            ep_actions.append(action_name)
            
            if done:
                break
        
        total_rewards.append(ep_reward)
        unique = len(set(ep_actions))
        print(f"Episode {ep+1}: Reward={ep_reward:.1f}, Unique actions={unique}")
    
    print(f"\nMean Reward: {np.mean(total_rewards):.1f} (+/- {np.std(total_rewards):.1f})")
    print(f"Total unique actions: {len(action_counts)}")
    
    print("\nTop 15 actions:")
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {action}: {count} ({count/total*100:.1f}%)")


if __name__ == "__main__":
    main()
