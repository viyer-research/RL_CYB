import sys
sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from stable_baselines3 import DQN
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper
import numpy as np
from collections import defaultdict

ACTION_TYPES = ['Monitor', 'Analyse', 'Remove', 'Restore']
HOSTS = ['Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0', 'User0', 'User1', 'User2', 'User3', 'User4']

model = DQN.load(r'D:\Vasanth\RL_CYB\ppo_out\dqn_hierarchical_v1.zip')

cyborg = CybORG(r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml', 'sim', agents={'Red': B_lineAgent})
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

action_counts = defaultdict(int)
rewards = []

for ep in range(10):
    cyborg.reset()
    obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
    ep_reward = 0
    ep_actions = []
    
    for step in range(100):
        action, _ = model.predict(obs, deterministic=True)
        action = int(action)
        type_idx = action // len(HOSTS)
        host_idx = action % len(HOSTS)
        action_obj = action_map.get((type_idx, host_idx))
        
        result = cyborg.step(agent='Blue', action=action_obj)
        ep_reward += result.reward
        obs = np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()
        
        action_name = str(action_obj)
        action_counts[action_name] += 1
        ep_actions.append(action_name)
    
    rewards.append(ep_reward)
    print(f'Episode {ep+1}: Reward={ep_reward:.1f}, Unique={len(set(ep_actions))}')

print(f'\nMean Reward: {np.mean(rewards):.1f} (+/- {np.std(rewards):.1f})')
print(f'Total unique actions: {len(action_counts)}')
print('\nTop 10 actions:')
total = sum(action_counts.values())
for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:10]:
    print(f'  {action}: {count} ({count/total*100:.1f}%)')