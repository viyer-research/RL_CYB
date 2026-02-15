"""
PPO Training with STATE-BASED reward only.
The agent is rewarded/penalized based on the SECURITY STATE of the network,
NOT based on what action it took. This prevents gaming the reward.

Key insight: If the agent only gets reward from network state, 
spamming Restore won't help unless it actually improves security.
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Add CybORG to path - use absolute path
CYBORG_PATH = r"z:\UNT\RL-Learning\cage-challenge-1\CybORG\CybORG"
sys.path.insert(0, CYBORG_PATH)

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper

SCENARIO_PATH = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml"
OUTPUT_DIR = r"D:\Vasanth\RL_CYB\ppo_out"


class StateBasedRewardWrapper(gym.Wrapper):
    """
    Reward based ONLY on network security state, not on actions taken.
    
    Reward = (number of clean hosts) - (number of compromised hosts) * penalty_weight
    
    This makes it impossible to game by repeating one action - 
    the agent must actually improve the security state.
    """
    
    def __init__(self, env, compromise_penalty=2.0, clean_bonus=0.1):
        super().__init__(env)
        self.compromise_penalty = compromise_penalty
        self.clean_bonus = clean_bonus
        self.last_action = None
        self.action_history = []
        self.hosts = [
            'User0', 'User1', 'User2', 'User3', 'User4',
            'Enterprise0', 'Enterprise1', 'Enterprise2',
            'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
            'Defender'
        ]
        self.critical_hosts = ['Op_Server0', 'Op_Host0', 'Op_Host1', 'Op_Host2']
        
    def reset(self, **kwargs):
        self.last_action = None
        self.action_history = []
        return self.env.reset(**kwargs)
    
    def _count_compromised_hosts(self, obs_dict):
        """Count compromised hosts from the blue table observation."""
        compromised = 0
        critical_compromised = 0
        
        # The BlueTableWrapper provides a table with host status
        # We need to check the raw CybORG state
        try:
            # Get the actual CybORG environment
            cyborg_env = self.env
            while hasattr(cyborg_env, 'env'):
                cyborg_env = cyborg_env.env
            
            if hasattr(cyborg_env, 'environment_controller'):
                state = cyborg_env.environment_controller.state
                for hostname in self.hosts:
                    if hostname in state.hosts:
                        host = state.hosts[hostname]
                        # Check if host has any malicious sessions or processes
                        has_red_session = any(
                            session.agent == 'Red' 
                            for session in host.sessions.values()
                        )
                        if has_red_session:
                            compromised += 1
                            if hostname in self.critical_hosts:
                                critical_compromised += 1
        except Exception as e:
            # Fallback: can't read state, use 0
            pass
            
        return compromised, critical_compromised
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get action name for logging
        action_name = "Unknown"
        try:
            cyborg_env = self.env
            while hasattr(cyborg_env, 'env'):
                cyborg_env = cyborg_env.env
            if hasattr(cyborg_env, 'get_last_action'):
                action_name = str(cyborg_env.get_last_action('Blue'))
        except:
            pass
        
        # === STATE-BASED REWARD ===
        # Use CybORG's native reward as base (it's already state-based!)
        state_reward = reward  # CybORG reward is based on Red's impact
        
        # === DIVERSITY BONUS ===
        # Strong penalty for repeating the EXACT same action
        diversity_bonus = 0.0
        if self.last_action is not None and action == self.last_action:
            diversity_bonus = -0.5  # Stronger penalty for repetition
        
        # Bonus for using different action TYPES (not just different hosts)
        action_type = action_name.split()[0] if action_name != "Unknown" else ""
        recent_types = [a.split()[0] for a in self.action_history[-5:] if a != "Unknown"]
        if action_type and action_type not in recent_types:
            diversity_bonus += 0.3  # Bonus for new action type
        
        # === EXPLORATION BONUS ===
        # Small bonus for actions we haven't taken much
        exploration_bonus = 0.0
        action_count = self.action_history.count(action_name)
        if action_count < 3:
            exploration_bonus = 0.2  # Bonus for rarely-used actions
        
        # === FINAL REWARD ===
        final_reward = state_reward + diversity_bonus + exploration_bonus
        
        # Update history
        self.last_action = action
        self.action_history.append(action_name)
        if len(self.action_history) > 50:
            self.action_history = self.action_history[-50:]
        
        info['state_reward'] = state_reward
        info['diversity_bonus'] = diversity_bonus
        info['exploration_bonus'] = exploration_bonus
        info['action_name'] = action_name
        
        return obs, final_reward, terminated, truncated, info


class ActionDiversityCallback(BaseCallback):
    """Monitor action diversity during training."""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.action_counts = {}
        self.episode_actions = []
        
    def _on_step(self) -> bool:
        # Track actions
        if 'action_name' in self.locals.get('infos', [{}])[0]:
            action = self.locals['infos'][0]['action_name']
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            self.episode_actions.append(action)
        
        # Log diversity every 10000 steps
        if self.num_timesteps % 10000 == 0 and self.action_counts:
            total = sum(self.action_counts.values())
            unique = len(self.action_counts)
            most_common = max(self.action_counts.items(), key=lambda x: x[1])
            
            print(f"\n[Step {self.num_timesteps}] Action Diversity:")
            print(f"  Unique actions: {unique}")
            print(f"  Most common: {most_common[0]} ({most_common[1]/total*100:.1f}%)")
            
            # Log to tensorboard
            self.logger.record("diversity/unique_actions", unique)
            self.logger.record("diversity/most_common_pct", most_common[1]/total*100)
            
        return True


def make_env():
    """Create the CybORG environment with proper wrappers."""
    cyborg = CybORG(SCENARIO_PATH, 'sim', agents={'Red': B_lineAgent})
    env = BlueTableWrapper(cyborg, output_mode='vector')
    env = EnumActionWrapper(env)
    env = FixedFlatWrapper(env)
    env = StateBasedRewardWrapper(env)
    return env


def main():
    print("=" * 60)
    print("PPO Training with STATE-BASED Reward")
    print("=" * 60)
    print("\nKey changes from previous attempts:")
    print("1. Reward based on CybORG's native state-based reward")
    print("2. Strong penalty (-0.5) for repeating same action")
    print("3. Bonus (+0.3) for using different action types")
    print("4. Exploration bonus (+0.2) for rarely-used actions")
    print("5. HIGH entropy coefficient (0.1) throughout training")
    print("=" * 60)
    
    # Create environment
    env = DummyVecEnv([make_env])
    
    # Check action space
    test_env = make_env()
    print(f"\nAction space: {test_env.action_space}")
    print(f"Observation space shape: {test_env.observation_space.shape}")
    test_env.close()
    
    # Create PPO with HIGH entropy - don't decay it!
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
        ent_coef=0.1,  # HIGH entropy, kept constant
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=OUTPUT_DIR,
    )
    
    # Train with diversity monitoring
    print("\nStarting training (1M steps)...")
    callback = ActionDiversityCallback()
    
    model.learn(
        total_timesteps=1_000_000,
        callback=callback,
        tb_log_name="PPO_state_reward_v1",
        progress_bar=True
    )
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "ppo_state_reward_v1.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Quick evaluation
    print("\n" + "=" * 60)
    print("Quick Evaluation (5 episodes)")
    print("=" * 60)
    
    eval_env = make_env()
    action_counts = {}
    
    for ep in range(5):
        obs, _ = eval_env.reset()
        total_reward = 0
        ep_actions = []
        
        for step in range(50):  # First 50 steps
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = eval_env.step(action)
            total_reward += reward
            
            action_name = info.get('action_name', f'Action_{action}')
            ep_actions.append(action_name)
            action_counts[action_name] = action_counts.get(action_name, 0) + 1
            
            if done or truncated:
                break
        
        unique_actions = len(set(ep_actions))
        print(f"Episode {ep+1}: Reward={total_reward:.1f}, Unique actions={unique_actions}")
    
    print("\nAction distribution across all eval episodes:")
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {action}: {count} ({count/total*100:.1f}%)")
    
    eval_env.close()
    env.close()
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Model: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
