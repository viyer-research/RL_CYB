"""
3x random baseline on the Sleep (no-adversary) profile, using the DQN evaluation protocol of
eval_dqn_3x.py::evaluate_vs_red: CybORG3xEnv(max_steps=100, red_agent='sleep', obs_delay=1),
n_episodes=10, max_steps=100, reward = sum of the wrapper's step reward (native + Isolate/Patch shaping).
Only the policy differs: env.action_space.sample() (uniform over the 120 actions), as in
eval_dqn_3x.py::evaluate_random_baseline. Also logs the per-action-type reward decomposition.
Usage: python random_3x_sleep.py <dir containing train_dqn_3x.py> <Scenario3x.yaml> [seed]
"""
import sys, random, numpy as np
from collections import defaultdict
sys.path.insert(0, sys.argv[1])
import train_dqn_3x as T3
from train_dqn_3x import CybORG3xEnv, ACTION_TYPES

SCEN = sys.argv[2]
SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 0
N_EP, T = 10, 100
random.seed(SEED); np.random.seed(SEED)

env = CybORG3xEnv(scenario_path=SCEN, max_steps=T, red_agent='sleep', obs_delay=1)
env.action_space.seed(SEED)
print(f'Scenario: {SCEN}\nProfile: sleep | episodes={N_EP} | T={T} | obs_delay=1 | policy=uniform random '
      f'(env.action_space.sample) | seed={SEED}')
cnt = defaultdict(int); rew = defaultdict(float); nat = defaultdict(float)
eps = []
for ep in range(N_EP):
    obs, _ = env.reset()
    R = 0.0
    for t in range(T):
        a = env.action_space.sample()
        obs, r, done, _, info = env.step(a)
        R += r
        ty = info['action_type']; cnt[ty] += 1; rew[ty] += r; nat[ty] += info['reward_native']
        if done:
            break
    eps.append(R)
    print(f'Episode {ep+1}: Reward={R:.1f}  isolated={info["isolated_count"]}  patched={info["patched_count"]}')
print(f'\nRANDOM vs sleep: mean {np.mean(eps):.1f} (+/- {np.std(eps):.1f})  min {min(eps):.1f}  max {max(eps):.1f}')
tot = sum(cnt.values())
print('\nPer action type (all episodes):  count  share  total_reward  native_part  shaping_part  per-episode_reward')
for ty in ACTION_TYPES:
    print(f'  {ty:8s} {cnt[ty]:5d}  {cnt[ty]/tot*100:5.1f}%  {rew[ty]:9.1f}  {nat[ty]:9.1f}  {rew[ty]-nat[ty]:9.1f}  {rew[ty]/N_EP:8.2f}')
print(f'  {"TOTAL":8s} {tot:5d}         {sum(rew.values()):9.1f}  {sum(nat.values()):9.1f}  {sum(rew.values())-sum(nat.values()):9.1f}  {sum(rew.values())/N_EP:8.2f}')
