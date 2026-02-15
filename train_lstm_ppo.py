"""
Recurrent PPO (LSTM) Training for CybORG Blue Agent.

Uses LSTM to capture sequential patterns in observations.
This helps because:
1. B_lineAgent follows a predictable attack sequence
2. Blue can learn to anticipate Red's next move
3. LSTM maintains memory of past observations

Requires: pip install sb3-contrib
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import RecurrentPPO
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
    """Hierarchical action space for CybORG."""
    
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
        
        # Build mapping
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
        print(f"[INFO] Observation shape: {self.observation_space.shape}")
        
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
        
        # Fill gaps with Sleep action
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
        
        result = self._cyborg.step(agent='Blue', action=action_obj)
        reward = result.reward
        
        self._t += 1
        done = self._t >= self.max_steps
        
        obs = np.array(self._flat.get_observation('Blue'), dtype=np.float32).flatten()
        return obs, reward, done, False, {'action_name': str(action_obj)}


class LSTMLoggingCallback(BaseCallback):
    """Callback for logging LSTM-PPO training progress."""
    
    def __init__(self, eval_env, log_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._evaluate()
        return True
    
    def _evaluate(self):
        """Run evaluation with LSTM state maintained across episode."""
        action_counts = defaultdict(int)
        total_rewards = []
        
        for _ in range(5):
            obs, _ = self.eval_env.reset()
            ep_reward = 0
            
            # Initialize LSTM states
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            
            for step in range(100):
                # RecurrentPPO needs lstm_states and episode_starts
                action, lstm_states = self.model.predict(
                    obs.reshape(1, -1), 
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True
                )
                episode_start = np.zeros((1,), dtype=bool)
                
                obs, reward, done, _, info = self.eval_env.step(int(action))
                ep_reward += reward
                
                action_name = info.get('action_name', str(action))
                action_counts[action_name] += 1
                
                if done:
                    break
            
            total_rewards.append(ep_reward)
        
        mean_reward = np.mean(total_rewards)
        unique_actions = len(action_counts)
        
        if action_counts:
            top_action = max(action_counts.items(), key=lambda x: x[1])
            top_pct = top_action[1] / sum(action_counts.values()) * 100
        else:
            top_action = ("None", 0)
            top_pct = 0
        
        print(f"\n[Step {self.n_calls}] LSTM Eval: Reward={mean_reward:.1f}, "
              f"Unique={unique_actions}, Top={top_action[0]} ({top_pct:.1f}%)")
        
        # Action type distribution
        type_counts = defaultdict(int)
        for action_name, count in action_counts.items():
            for action_type in ACTION_TYPES:
                if action_name.startswith(action_type):
                    type_counts[action_type] += count
                    break
        
        total = sum(type_counts.values()) or 1
        print(f"  Types: " + ", ".join(
            f"{t}:{type_counts[t]/total*100:.0f}%" for t in ACTION_TYPES
        ))


def make_env(scenario_path, max_steps=100):
    def _init():
        return HierarchicalCybORGEnv(scenario_path, max_steps=max_steps)
    return _init


def main():
    print("="*60)
    print("  Recurrent PPO (LSTM) Training for CybORG Blue Agent")
    print("="*60)
    
    # Create training environment
    print("\nCreating training environment...")
    env = DummyVecEnv([make_env(SCENARIO_PATH, max_steps=100)])
    
    # Create eval environment
    eval_env = HierarchicalCybORGEnv(SCENARIO_PATH, max_steps=100)
    
    # RecurrentPPO with LSTM
    # Key: LSTM maintains hidden state across timesteps within an episode
    model = RecurrentPPO(
        "MlpLstmPolicy",
        env,
        learning_rate=3e-4,
        n_steps=128,                   # Steps per rollout (shorter for LSTM)
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,                 # Lower entropy - let LSTM learn patterns
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            lstm_hidden_size=128,      # LSTM hidden state size
            n_lstm_layers=1,           # Number of LSTM layers
            shared_lstm=True,          # Share LSTM between policy and value
            enable_critic_lstm=False,  # False when shared_lstm=True
            net_arch=[128, 64]         # Network after LSTM
        ),
        verbose=1,
        tensorboard_log=os.path.join(OUTPUT_DIR, "lstm_ppo_logs")
    )
    
    print(f"\nModel: RecurrentPPO with LSTM")
    print(f"  LSTM hidden size: 128")
    print(f"  LSTM layers: 1")
    print(f"  Policy net: [128, 64]")
    print(f"  Entropy coefficient: 0.01 (low - encourage commitment)")
    
    # Callback
    callback = LSTMLoggingCallback(eval_env, log_freq=20000)
    
    # Train
    total_timesteps = 500000
    print(f"\nStarting LSTM-PPO training for {total_timesteps} steps...")
    print("="*60)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save
    model_path = os.path.join(OUTPUT_DIR, "lstm_ppo_hierarchical_v1.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Final evaluation
    print("\n" + "="*60)
    print("  Final LSTM Evaluation (Deterministic)")
    print("="*60)
    
    action_counts = defaultdict(int)
    total_rewards = []
    
    for ep in range(10):
        obs, _ = eval_env.reset()
        ep_reward = 0
        ep_actions = []
        
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        
        for step in range(100):
            action, lstm_states = model.predict(
                obs.reshape(1, -1),
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)
            
            obs, reward, done, _, info = eval_env.step(int(action))
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
    
    # Action type distribution
    print("\nAction Type Distribution:")
    type_counts = defaultdict(int)
    for action_name, count in action_counts.items():
        for action_type in ACTION_TYPES:
            if action_name.startswith(action_type):
                type_counts[action_type] += count
                break
    
    for action_type in ACTION_TYPES:
        count = type_counts.get(action_type, 0)
        pct = count / total * 100 if total > 0 else 0
        print(f"  {action_type}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
