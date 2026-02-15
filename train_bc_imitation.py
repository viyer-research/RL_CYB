"""
Imitation Learning via Behavioral Cloning.
Train a policy to imitate a diverse "expert" policy (random or heuristic).
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import defaultdict

# Add CybORG to path
sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper

SCENARIO_PATH = r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml'
OUTPUT_DIR = r"D:\Vasanth\RL_CYB\ppo_out"


class PolicyNetwork(nn.Module):
    """Simple MLP policy for behavioral cloning."""
    
    def __init__(self, obs_dim, action_dim, hidden_sizes=[256, 256]):
        super().__init__()
        
        layers = []
        prev_size = obs_dim
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, action_dim))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, obs):
        return self.network(obs)
    
    def get_action(self, obs, deterministic=True):
        with torch.no_grad():
            logits = self.forward(obs)
            if deterministic:
                return torch.argmax(logits, dim=-1)
            else:
                probs = torch.softmax(logits, dim=-1)
                return torch.multinomial(probs, 1).squeeze(-1)


def create_env():
    """Create CybORG environment with wrappers."""
    cyborg = CybORG(SCENARIO_PATH, 'sim', agents={'Red': B_lineAgent})
    table = BlueTableWrapper(cyborg, output_mode='vector')
    enum = EnumActionWrapper(table)
    flat = FixedFlatWrapper(cyborg, agent='Blue')
    
    enum.action_space_change(cyborg.get_action_space('Blue'))
    possible_actions = enum.possible_actions
    
    return cyborg, flat, enum, possible_actions


def smart_random_action(step, possible_actions, action_history):
    """
    Smart random policy that:
    1. Prefers Monitor early in episode
    2. Prefers Analyse after Monitor detects something
    3. Prefers Remove/Restore for known compromised hosts
    4. Avoids repeating same action twice in a row
    """
    n_actions = len(possible_actions)
    
    # Get action names
    action_names = [str(a) for a in possible_actions]
    
    # Weight different action types
    weights = np.ones(n_actions)
    
    for i, name in enumerate(action_names):
        # Slightly prefer active defense actions
        if 'Monitor' in name:
            weights[i] = 1.5
        elif 'Analyse' in name:
            weights[i] = 1.3
        elif 'Remove' in name:
            weights[i] = 1.4
        elif 'Restore' in name:
            weights[i] = 1.2
        elif 'Sleep' in name:
            weights[i] = 0.3  # Discourage sleep
        elif 'Misinform' in name:
            weights[i] = 0.8  # Slightly discourage misinform
    
    # Discourage repeating last action
    if action_history:
        last_action = action_history[-1]
        weights[last_action] *= 0.3
    
    # Normalize and sample
    probs = weights / weights.sum()
    action = np.random.choice(n_actions, p=probs)
    
    return action


def collect_demonstrations(n_episodes=100, steps_per_episode=200):
    """Collect demonstrations from smart random policy."""
    print(f"Collecting {n_episodes} episodes of demonstrations...")
    
    cyborg, flat, enum, possible_actions = create_env()
    n_actions = len(possible_actions)
    
    observations = []
    actions = []
    
    for ep in range(n_episodes):
        cyborg.reset()
        obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
        action_history = []
        
        for step in range(steps_per_episode):
            # Get smart random action
            action = smart_random_action(step, possible_actions, action_history)
            
            # Store transition
            observations.append(obs)
            actions.append(action)
            action_history.append(action)
            
            # Execute action
            action_obj = possible_actions[action]
            result = cyborg.step('Blue', action_obj)
            obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
        
        if (ep + 1) % 20 == 0:
            print(f"  Collected {ep + 1}/{n_episodes} episodes")
    
    observations = np.array(observations)
    actions = np.array(actions)
    
    print(f"Total transitions: {len(observations)}")
    
    # Print action distribution in demonstrations
    action_counts = defaultdict(int)
    for a in actions:
        action_counts[str(possible_actions[a])] += 1
    
    print("\nDemonstration action distribution (top 15):")
    total = len(actions)
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {action}: {count} ({count/total*100:.1f}%)")
    
    return observations, actions, n_actions


def train_behavioral_cloning(observations, actions, n_actions, 
                              epochs=50, batch_size=256, lr=1e-3):
    """Train policy via behavioral cloning (supervised learning)."""
    
    obs_dim = observations.shape[1]
    
    # Create dataset
    obs_tensor = torch.FloatTensor(observations)
    action_tensor = torch.LongTensor(actions)
    dataset = TensorDataset(obs_tensor, action_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nTraining on: {device}")
    
    policy = PolicyNetwork(obs_dim, n_actions).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\nTraining for {epochs} epochs...")
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_obs, batch_actions in dataloader:
            batch_obs = batch_obs.to(device)
            batch_actions = batch_actions.to(device)
            
            optimizer.zero_grad()
            logits = policy(batch_obs)
            loss = criterion(logits, batch_actions)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_actions).sum().item()
            total += len(batch_actions)
        
        accuracy = correct / total * 100
        avg_loss = total_loss / len(dataloader)
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: Loss={avg_loss:.4f}, Accuracy={accuracy:.1f}%")
    
    return policy, device


def evaluate_policy(policy, device, n_episodes=5, deterministic=True):
    """Evaluate the learned policy."""
    
    cyborg, flat, enum, possible_actions = create_env()
    
    mode = "DETERMINISTIC" if deterministic else "STOCHASTIC"
    print(f"\n{'='*60}")
    print(f"Evaluating BC Policy ({mode})")
    print(f"{'='*60}")
    
    action_counts = defaultdict(int)
    total_rewards = []
    
    for ep in range(n_episodes):
        cyborg.reset()
        obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
        ep_reward = 0
        ep_actions = []
        
        for step in range(200):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            action = policy.get_action(obs_tensor, deterministic=deterministic).item()
            
            action_obj = possible_actions[action]
            action_name = str(action_obj)
            
            result = cyborg.step('Blue', action_obj)
            ep_reward += result.reward
            obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
            
            ep_actions.append(action_name)
            action_counts[action_name] += 1
        
        unique = len(set(ep_actions))
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}: Reward={ep_reward:.1f}, Unique actions={unique}")
    
    print(f"\nAverage Reward: {np.mean(total_rewards):.1f} (+/- {np.std(total_rewards):.1f})")
    
    print("\nAction distribution (top 15):")
    total = sum(action_counts.values())
    for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {action}: {count} ({count/total*100:.1f}%)")
    
    return np.mean(total_rewards)


def main():
    print("=" * 60)
    print("Behavioral Cloning from Smart Random Policy")
    print("=" * 60)
    
    # Step 1: Collect demonstrations
    observations, actions, n_actions = collect_demonstrations(
        n_episodes=100, 
        steps_per_episode=200
    )
    
    # Step 2: Train via behavioral cloning
    policy, device = train_behavioral_cloning(
        observations, actions, n_actions,
        epochs=100,
        batch_size=256,
        lr=1e-3
    )
    
    # Step 3: Evaluate
    print("\n" + "=" * 60)
    reward_det = evaluate_policy(policy, device, n_episodes=5, deterministic=True)
    reward_stoch = evaluate_policy(policy, device, n_episodes=5, deterministic=False)
    
    # Save model
    model_path = os.path.join(OUTPUT_DIR, "bc_policy.pth")
    torch.save({
        'model_state_dict': policy.state_dict(),
        'obs_dim': observations.shape[1],
        'n_actions': n_actions,
    }, model_path)
    print(f"\nModel saved to: {model_path}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Deterministic reward: {reward_det:.1f}")
    print(f"  Stochastic reward: {reward_stoch:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
