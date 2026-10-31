"""
Table 3 re-run under one protocol (3x scenario).
Protocol = eval_dqn_3x.py::evaluate_vs_red: CybORG3xEnv(Scenario3x.yaml, max_steps=100, red_agent=<profile>,
obs_delay=1), 10 episodes x 100 steps, reward = wrapper step reward (CybORG native reward + Isolate/Patch shaping).
Policies: DQN dqn_3x_v1.zip with model.predict(deterministic=True); random = env.action_space.sample().
Profiles: bline, meander, sleep. Seeds 0, 1, 2 (random, numpy, torch, action_space) -> 30 episodes per cell.
Usage: python table3_3x.py <dir with train_dqn_3x.py> <Scenario3x.yaml> <dqn_3x_v1.zip> <out.json>
"""
import sys, json, random, time
import numpy as np, torch
from collections import Counter
sys.path.insert(0, sys.argv[1])
from train_dqn_3x import CybORG3xEnv, ACTION_TYPES
from stable_baselines3 import DQN

SCEN, MODEL, OUT = sys.argv[2], sys.argv[3], sys.argv[4]
N_EP, T, SEEDS = 10, 100, [0, 1, 2]
PROFILES = ['bline', 'meander', 'sleep']
model = DQN.load(MODEL, device='cpu')
print(f'Scenario: {SCEN}\nModel: {MODEL}\nProtocol: 10 episodes x {T} steps per seed, obs_delay=1, seeds {SEEDS}; '
      f'DQN deterministic=True; random = action_space.sample()\n', flush=True)
results = {}
for policy in ['DQN', 'Random']:
    for prof in PROFILES:
        cell = {'per_seed': {}, 'episodes': [], 'types': Counter(), 'actions': Counter(), 'changes50': []}
        for sd in SEEDS:
            random.seed(sd); np.random.seed(sd); torch.manual_seed(sd)
            env = CybORG3xEnv(scenario_path=SCEN, max_steps=T, red_agent=prof, obs_delay=1)
            env.action_space.seed(sd)
            eps = []
            for ep in range(N_EP):
                obs, _ = env.reset(); R = 0.0; seq = []
                for t in range(T):
                    if policy == 'DQN':
                        a, _ = model.predict(obs, deterministic=True)
                    else:
                        a = env.action_space.sample()
                    obs, r, done, _, info = env.step(a)
                    R += r; seq.append(info['action_name'])
                    cell['types'][info['action_type']] += 1; cell['actions'][info['action_name']] += 1
                    if done:
                        break
                eps.append(R)
                cell['changes50'].append(sum(1 for i in range(1, 50) if seq[i] != seq[i - 1]))
            cell['per_seed'][sd] = (float(np.mean(eps)), float(np.std(eps)))
            cell['episodes'] += eps
            print(f'[{policy:6s} {prof:8s} seed {sd}] mean {np.mean(eps):8.1f} +/- {np.std(eps):6.1f}  '
                  f'episodes {[round(e, 1) for e in eps]}', flush=True)
        e = np.array(cell['episodes']); tot = sum(cell['types'].values())
        summ = dict(mean=float(e.mean()), std=float(e.std()), n=len(e), per_seed=cell['per_seed'],
                    type_pct={k: round(cell['types'][k] / tot * 100, 1) for k in ACTION_TYPES},
                    top_actions=cell['actions'].most_common(8), changes_per_50=float(np.mean(cell['changes50'])))
        results[f'{policy}|{prof}'] = summ
        print(f'==> {policy} {prof}: {summ["mean"]:.1f} +/- {summ["std"]:.1f} (n={summ["n"]}) | types {summ["type_pct"]} '
              f'| changes/50 {summ["changes_per_50"]:.1f}\n    top {summ["top_actions"][:6]}\n', flush=True)
json.dump(results, open(OUT, 'w'), indent=1)
