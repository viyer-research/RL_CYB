"""Blue does nothing (Sleep) on a scenario: native reward, 10 x 100, Red=B_lineAgent. Usage: python sleep_baseline.py <scenario.yaml> [seed]"""
import sys, random, numpy as np
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Shared.Actions import Sleep
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
random.seed(seed); np.random.seed(seed)
env = CybORG(sys.argv[1], 'sim', agents={'Red': B_lineAgent})
R = []
for ep in range(10):
    env.reset(); r = 0.0
    for t in range(100):
        r += env.step(agent='Blue', action=Sleep()).reward
    R.append(r); print(f'Episode {ep+1}: Reward={r:.1f}')
print(f'{sys.argv[1]}\nBlue=Sleep, native reward, 10x100, seed={seed}: mean {np.mean(R):.1f} (+/- {np.std(R):.1f})')
