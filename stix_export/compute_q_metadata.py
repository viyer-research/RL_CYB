import json
import sys
import torch
import numpy as np
import os
sys.path.insert(0, r'D:\Vasanth\RL_CYB')
from train_dqn_3x import CybORG3xEnv, ALL_HOSTS, ACTION_TYPES, N_HOSTS, N_ACTION_TYPES
from stable_baselines3 import DQN

MODEL_PATH = r'D:\Vasanth\RL_CYB\ppo_out\dqn_3x_v1.zip'

def decode_action(idx):
    type_idx = idx // N_HOSTS
    host_idx = idx % N_HOSTS
    return ACTION_TYPES[type_idx], ALL_HOSTS[host_idx]

model = DQN.load(MODEL_PATH)
results = {}
for red in ['meander','bline']:
    env = CybORG3xEnv(max_steps=100, red_agent=red, obs_delay=1)
    obs, _ = env.reset()
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy().flatten()
    # per action type stats
    type_stats = {}
    per_host_q = {}
    for type_idx, action_type in enumerate(ACTION_TYPES):
        vals = [float(q_values[type_idx * N_HOSTS + h]) for h in range(N_HOSTS)]
        type_stats[action_type] = {'mean': float(np.mean(vals)), 'max': float(np.max(vals)), 'min': float(np.min(vals))}
        per_host_q[action_type] = vals
    top_idxs = np.argsort(q_values)[-15:][::-1]
    top_actions = []
    for idx in top_idxs:
        a_type, a_host = decode_action(idx)
        top_actions.append({'action': f"{a_type} {a_host}", 'Q': float(q_values[idx])})
    results[red] = {'type_stats': type_stats, 'per_host_q': per_host_q, 'top_actions': top_actions}

with open('q_metadata.json','w') as f:
    json.dump(results,f,indent=2)
print('Wrote q_metadata.json')
