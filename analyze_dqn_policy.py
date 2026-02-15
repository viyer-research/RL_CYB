"""
Analyze what policy DQN has learned - 
examine Q-values and action selection patterns.
"""

import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from stable_baselines3 import DQN
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper
import torch

SCENARIO_PATH = r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml'
MODEL_PATH = r'D:\Vasanth\RL_CYB\ppo_out\dqn_hierarchical_v1.zip'

ACTION_TYPES = ['Monitor', 'Analyse', 'Remove', 'Restore']
HOSTS = ['Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2', 
         'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
         'User0', 'User1', 'User2', 'User3', 'User4']


def decode_action(action):
    type_idx = action // len(HOSTS)
    host_idx = action % len(HOSTS)
    return ACTION_TYPES[type_idx], HOSTS[host_idx]


def get_action_name(action, action_map):
    type_idx = action // len(HOSTS)
    host_idx = action % len(HOSTS)
    action_obj = action_map.get((type_idx, host_idx))
    return str(action_obj)


def main():
    print("="*70)
    print("  DQN Policy Analysis")
    print("="*70)
    
    # Load model
    model = DQN.load(MODEL_PATH)
    
    # Create environment
    cyborg = CybORG(SCENARIO_PATH, "sim", agents={'Red': B_lineAgent})
    table = BlueTableWrapper(env=cyborg, output_mode='vector')
    enum = EnumActionWrapper(env=table)
    flat = FixedFlatWrapper(env=cyborg, agent='Blue')
    
    enum.action_space_change(cyborg.get_action_space('Blue'))
    all_actions = enum.possible_actions
    
    # Build action map
    action_map = {}
    for action in all_actions:
        action_str = str(action)
        for type_idx, action_type in enumerate(ACTION_TYPES):
            if action_str.startswith(action_type):
                if action_type == 'Monitor':
                    for host_idx in range(len(HOSTS)):
                        action_map[(type_idx, host_idx)] = action
                else:
                    for host_idx, host in enumerate(HOSTS):
                        if host in action_str:
                            action_map[(type_idx, host_idx)] = action
                            break
                break
    
    # Track action selection by step number
    step_actions = defaultdict(list)
    
    print("\n" + "="*70)
    print("  Running detailed episode analysis")
    print("="*70)
    
    for ep in range(3):
        print(f"\n--- Episode {ep+1} ---")
        cyborg.reset()
        obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
        
        for step in range(30):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
            with torch.no_grad():
                q_values = model.q_net(obs_tensor).cpu().numpy().flatten()
            
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            action_name = get_action_name(action, action_map)
            action_type, host = decode_action(action)
            
            step_actions[step].append(action_name)
            
            top_actions = np.argsort(q_values)[-5:][::-1]
            
            if step < 10 or step % 5 == 0:
                print(f"\nStep {step}: Selected '{action_name}'")
                print(f"  Top 5 Q-values:")
                for idx in top_actions:
                    a_type, a_host = decode_action(idx)
                    q_val = q_values[idx]
                    marker = " <-- SELECTED" if idx == action else ""
                    print(f"    {a_type:8s} {a_host:15s}: Q={q_val:7.2f}{marker}")
            
            action_obj = action_map.get((action // len(HOSTS), action % len(HOSTS)))
            result = cyborg.step(agent='Blue', action=action_obj)
            obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
    
    print("\n" + "="*70)
    print("  Step-wise Action Selection Pattern")
    print("="*70)
    print("\nMost common action at each step (first 20 steps):")
    
    for step in range(20):
        actions = step_actions[step]
        if actions:
            from collections import Counter
            most_common = Counter(actions).most_common(1)[0]
            print(f"  Step {step:2d}: {most_common[0]} ({most_common[1]}/{len(actions)} episodes)")
    
    print("\n" + "="*70)
    print("  Q-Value Analysis for Initial State")
    print("="*70)
    
    cyborg.reset()
    obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy().flatten()
    
    print(f"\nQ-value statistics (initial state):")
    print(f"  Mean: {q_values.mean():.2f}")
    print(f"  Std:  {q_values.std():.2f}")
    print(f"  Min:  {q_values.min():.2f}")
    print(f"  Max:  {q_values.max():.2f}")
    print(f"  Gap (max-2nd): {np.sort(q_values)[-1] - np.sort(q_values)[-2]:.2f}")
    
    print("\nQ-values by action type (initial state):")
    for type_idx, action_type in enumerate(ACTION_TYPES):
        type_q_values = []
        for host_idx in range(len(HOSTS)):
            action_idx = type_idx * len(HOSTS) + host_idx
            type_q_values.append(q_values[action_idx])
        print(f"  {action_type:8s}: mean={np.mean(type_q_values):7.2f}, "
              f"max={np.max(type_q_values):7.2f}, min={np.min(type_q_values):7.2f}")
    
    print("\nTop 10 actions by Q-value (initial state):")
    top_10 = np.argsort(q_values)[-10:][::-1]
    for rank, idx in enumerate(top_10):
        a_type, a_host = decode_action(idx)
        print(f"  {rank+1:2d}. {a_type:8s} {a_host:15s}: Q={q_values[idx]:7.2f}")
    
    print("\n" + "="*70)
    print("  State-Dependency Check")
    print("="*70)
    
    print("\nTracking action changes over episode:")
    cyborg.reset()
    obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
    prev_action = None
    action_changes = 0
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        
        if prev_action is not None and action != prev_action:
            action_changes += 1
            prev_name = get_action_name(prev_action, action_map)
            curr_name = get_action_name(action, action_map)
            if action_changes <= 10:
                print(f"  Step {step}: Changed from '{prev_name}' to '{curr_name}'")
        
        prev_action = action
        action_obj = action_map.get((action // len(HOSTS), action % len(HOSTS)))
        result = cyborg.step(agent='Blue', action=action_obj)
        obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
    
    print(f"\nTotal action changes in 50 steps: {action_changes}")
    print(f"This means the policy is {'STATE-DEPENDENT' if action_changes > 5 else 'MOSTLY STATIC'}")


if __name__ == "__main__":
    main()