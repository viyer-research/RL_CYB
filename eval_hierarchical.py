"""
Evaluation script for hierarchical PPO model.
Tests both deterministic and stochastic inference.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from collections import defaultdict

# Add CybORG to path
sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper

SCENARIO_PATH = r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml'
MODEL_PATH = r'D:\Vasanth\RL_CYB\ppo_out\ppo_hierarchical_v1.zip'

# Same constants as training
ACTION_TYPES = ['Monitor', 'Analyse', 'Remove', 'Restore']
HOSTS = [
    'Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2',
    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
    'User0', 'User1', 'User2', 'User3', 'User4'
]


class HierarchicalCybORGEnv(gym.Env):
    """Same wrapper as training - needed to decode actions correctly."""
    
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
    
    def get_action_name(self, action):
        """Get human-readable action name."""
        type_idx, host_idx = self._decode_action(int(action))
        action_obj = self._action_map.get((type_idx, host_idx))
        return str(action_obj)
    
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


def evaluate(model, env, n_episodes=10, deterministic=True):
    """Evaluate model and return stats."""
    action_counts = defaultdict(int)
    type_counts = defaultdict(int)
    host_counts = defaultdict(int)
    episode_rewards = []
    episode_unique_actions = []
    
    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        ep_actions = []
        
        for step in range(100):
            action, _ = model.predict(obs, deterministic=deterministic)
            action = int(action)
            
            # Decode action
            type_idx = action // len(HOSTS)
            host_idx = action % len(HOSTS)
            action_type = ACTION_TYPES[type_idx]
            host = HOSTS[host_idx]
            action_name = env.get_action_name(action)
            
            obs, reward, done, _, _ = env.step(action)
            ep_reward += reward
            
            action_counts[action_name] += 1
            type_counts[action_type] += 1
            host_counts[host] += 1
            ep_actions.append(action_name)
            
            if done:
                break
        
        episode_rewards.append(ep_reward)
        episode_unique_actions.append(len(set(ep_actions)))
    
    return {
        'action_counts': dict(action_counts),
        'type_counts': dict(type_counts),
        'host_counts': dict(host_counts),
        'episode_rewards': episode_rewards,
        'episode_unique_actions': episode_unique_actions
    }


def print_results(results, mode_name):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  {mode_name}")
    print(f"{'='*60}")
    
    rewards = results['episode_rewards']
    unique = results['episode_unique_actions']
    
    print(f"\nEpisode Rewards: mean={np.mean(rewards):.1f}, std={np.std(rewards):.1f}")
    print(f"  Per episode: {[f'{r:.0f}' for r in rewards]}")
    
    print(f"\nUnique Actions per Episode: mean={np.mean(unique):.1f}")
    print(f"  Per episode: {unique}")
    
    print(f"\nAction Type Distribution:")
    total_types = sum(results['type_counts'].values())
    for action_type in ACTION_TYPES:
        count = results['type_counts'].get(action_type, 0)
        pct = count / total_types * 100 if total_types > 0 else 0
        print(f"  {action_type:12s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nHost Distribution:")
    total_hosts = sum(results['host_counts'].values())
    for host in HOSTS:
        count = results['host_counts'].get(host, 0)
        pct = count / total_hosts * 100 if total_hosts > 0 else 0
        print(f"  {host:15s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nTop 15 Actions:")
    total_actions = sum(results['action_counts'].values())
    sorted_actions = sorted(results['action_counts'].items(), key=lambda x: -x[1])
    for action, count in sorted_actions[:15]:
        pct = count / total_actions * 100 if total_actions > 0 else 0
        print(f"  {action:40s}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\nTotal unique actions used: {len(results['action_counts'])}")


def main():
    print("Loading model from:", MODEL_PATH)
    model = PPO.load(MODEL_PATH)
    
    print("Creating environment...")
    env = HierarchicalCybORGEnv(SCENARIO_PATH, max_steps=100)
    
    n_episodes = 10
    
    # Deterministic evaluation (the real test!)
    print("\nRunning DETERMINISTIC evaluation...")
    det_results = evaluate(model, env, n_episodes=n_episodes, deterministic=True)
    print_results(det_results, "DETERMINISTIC (deterministic=True)")
    
    # Stochastic evaluation (for comparison)
    print("\nRunning STOCHASTIC evaluation...")
    stoch_results = evaluate(model, env, n_episodes=n_episodes, deterministic=False)
    print_results(stoch_results, "STOCHASTIC (deterministic=False)")
    
    # Summary comparison
    print("\n" + "="*60)
    print("  SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Metric':<30} {'Deterministic':>15} {'Stochastic':>15}")
    print("-"*60)
    print(f"{'Mean Reward':<30} {np.mean(det_results['episode_rewards']):>15.1f} {np.mean(stoch_results['episode_rewards']):>15.1f}")
    print(f"{'Mean Unique Actions/Episode':<30} {np.mean(det_results['episode_unique_actions']):>15.1f} {np.mean(stoch_results['episode_unique_actions']):>15.1f}")
    print(f"{'Total Unique Actions':<30} {len(det_results['action_counts']):>15d} {len(stoch_results['action_counts']):>15d}")
    
    det_most_common = max(det_results['action_counts'].values()) / sum(det_results['action_counts'].values()) * 100
    stoch_most_common = max(stoch_results['action_counts'].values()) / sum(stoch_results['action_counts'].values()) * 100
    print(f"{'Most Common Action %':<30} {det_most_common:>14.1f}% {stoch_most_common:>14.1f}%")


if __name__ == "__main__":
    main()
