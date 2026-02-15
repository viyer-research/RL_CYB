"""Quick stochastic evaluation of PPO diverse model."""
import sys
sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from stable_baselines3 import PPO
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper
import numpy as np

# Load model
model = PPO.load(r'D:\Vasanth\RL_CYB\ppo_out\ppo_diverse_v1.zip')

# Create env
cyborg = CybORG(r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml','sim', agents={'Red': B_lineAgent})
table = BlueTableWrapper(cyborg, output_mode='vector')
enum = EnumActionWrapper(table)
flat = FixedFlatWrapper(cyborg, agent='Blue')

enum.action_space_change(cyborg.get_action_space('Blue'))
possible_actions = enum.possible_actions

print('=' * 60)
print('STOCHASTIC Evaluation (deterministic=False)')
print('=' * 60)
action_counts = {}

for ep in range(3):
    cyborg.reset()
    obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
    ep_actions = []
    total_reward = 0
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=False)  # STOCHASTIC
        action_obj = possible_actions[int(action)]
        action_name = str(action_obj)
        
        result = cyborg.step('Blue', action_obj)
        total_reward += result.reward
        obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
        
        ep_actions.append(action_name)
        action_counts[action_name] = action_counts.get(action_name, 0) + 1
    
    unique = len(set(ep_actions))
    print(f'Episode {ep+1}: Reward={total_reward:.1f}, Unique actions={unique}')

print()
print('Action distribution (top 15):')
total = sum(action_counts.values())
for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:15]:
    print(f'  {action}: {count} ({count/total*100:.1f}%)')

print()
print('=' * 60)
print('DETERMINISTIC Evaluation (for comparison)')
print('=' * 60)
action_counts_det = {}

for ep in range(3):
    cyborg.reset()
    obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
    ep_actions = []
    total_reward = 0
    
    for step in range(50):
        action, _ = model.predict(obs, deterministic=True)  # DETERMINISTIC
        action_obj = possible_actions[int(action)]
        action_name = str(action_obj)
        
        result = cyborg.step('Blue', action_obj)
        total_reward += result.reward
        obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
        
        ep_actions.append(action_name)
        action_counts_det[action_name] = action_counts_det.get(action_name, 0) + 1
    
    unique = len(set(ep_actions))
    print(f'Episode {ep+1}: Reward={total_reward:.1f}, Unique actions={unique}')

print()
print('Action distribution (top 15):')
total = sum(action_counts_det.values())
for action, count in sorted(action_counts_det.items(), key=lambda x: -x[1])[:15]:
    print(f'  {action}: {count} ({count/total*100:.1f}%)')
