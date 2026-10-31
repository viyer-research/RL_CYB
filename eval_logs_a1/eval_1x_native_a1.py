"""
(a1) Re-evaluation of 1x agents on Scenario1b-vulnerable.yaml with CybORG's NATIVE
Blue reward, deterministic (argmax) execution, 10 episodes x 100 steps.
Protocol mirrors eval_dqn.py / eval_hierarchical.py:
  CybORG(<Scenario1b-vulnerable.yaml>, 'sim', agents={'Red': B_lineAgent}),
  FixedFlatWrapper observation, cyborg.reset() per episode, reward = sum(result.reward).
Models:
  ppo_flat  : ppo_cyborg_fixedflat_shaped_vulnerable.zip (PPO, native 54-action EnumActionWrapper space)
  lstm_ppo  : lstm_ppo_hierarchical_v1.zip (RecurrentPPO, 4 types x 13 hosts = 52 actions)
  ppo_hier  : ppo_hierarchical_v1.zip   (reference re-run, same session)
  dqn       : dqn_hierarchical_v1.zip   (reference re-run, same session)
Usage: python eval_1x_native_a1.py <scenario.yaml> <model_dir> <out_dir> [seed]
"""
import sys, os, json, random, time
import numpy as np
import torch
from collections import Counter
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper
from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO

SCEN, MDIR, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 0
N_EP, T = 10, 100
ACTION_TYPES = ['Monitor', 'Analyse', 'Remove', 'Restore']
HOSTS = ['Defender', 'Enterprise0', 'Enterprise1', 'Enterprise2', 'Op_Host0', 'Op_Host1',
         'Op_Host2', 'Op_Server0', 'User0', 'User1', 'User2', 'User3', 'User4']
os.makedirs(OUT, exist_ok=True)


def make_env():
    cyborg = CybORG(SCEN, 'sim', agents={'Red': B_lineAgent})
    table = BlueTableWrapper(env=cyborg, output_mode='vector')
    enum = EnumActionWrapper(env=table)
    flat = FixedFlatWrapper(env=cyborg, agent='Blue')
    enum.action_space_change(cyborg.get_action_space('Blue'))
    acts = enum.possible_actions
    amap = {}
    for a in acts:
        s = str(a)
        for ti, at in enumerate(ACTION_TYPES):
            if s.startswith(at):
                if at == 'Monitor':
                    for hi in range(len(HOSTS)):
                        amap[(ti, hi)] = a
                else:
                    for hi, h in enumerate(HOSTS):
                        if h in s:
                            amap[(ti, hi)] = a
                            break
                break
    sleep = next(a for a in acts if 'Sleep' in str(a))
    for ti in range(len(ACTION_TYPES)):
        for hi in range(len(HOSTS)):
            amap.setdefault((ti, hi), sleep)
    return cyborg, flat, acts, amap


def obs_of(flat):
    return np.array(flat.get_observation('Blue'), dtype=np.float32).flatten()


def run(name, model, kind, log):
    cyborg, flat, acts, amap = make_env()
    rewards, uniq, changes50, counts, per_ep_actions = [], [], [], Counter(), []
    for ep in range(N_EP):
        cyborg.reset()
        obs = obs_of(flat)
        lstm_state, start = None, np.ones((1,), dtype=bool)
        R, seq = 0.0, []
        for t in range(T):
            if kind == 'lstm':
                a, lstm_state = model.predict(obs.reshape(1, -1), state=lstm_state,
                                              episode_start=start, deterministic=True)
                start = np.zeros((1,), dtype=bool)
            else:
                a, _ = model.predict(obs, deterministic=True)
            a = int(np.asarray(a).flatten()[0])
            act = acts[a] if kind == 'flat' else amap[(a // len(HOSTS), a % len(HOSTS))]
            res = cyborg.step(agent='Blue', action=act)
            R += res.reward
            obs = obs_of(flat)
            seq.append(str(act))
        rewards.append(R)
        uniq.append(len(set(seq)))
        changes50.append(sum(1 for i in range(1, 50) if seq[i] != seq[i - 1]))
        counts.update(seq)
        per_ep_actions.append(Counter(seq).most_common(3))
        log(f'[{name}] Episode {ep+1}: Reward={R:.1f}, Unique={len(set(seq))}, '
            f'ChangesFirst50={changes50[-1]}, Top={Counter(seq).most_common(2)}')
    tot = sum(counts.values())
    types = {at: sum(c for k, c in counts.items() if k.startswith(at)) / tot * 100 for at in ACTION_TYPES}
    summ = dict(model=name, n_episodes=N_EP, steps=T, mean=float(np.mean(rewards)),
                std=float(np.std(rewards)), per_episode=[round(r, 2) for r in rewards],
                unique_per_ep=float(np.mean(uniq)), total_unique=len(counts),
                changes_per_50=float(np.mean(changes50)),
                type_pct={k: round(v, 1) for k, v in types.items()},
                top_actions=counts.most_common(8))
    log(f'[{name}] Mean Reward: {summ["mean"]:.1f} (+/- {summ["std"]:.1f}) | unique/ep '
        f'{summ["unique_per_ep"]:.1f} | total unique {summ["total_unique"]} | changes/50 '
        f'{summ["changes_per_50"]:.1f} | types {summ["type_pct"]}')
    log(f'[{name}] Top actions: {summ["top_actions"]}')
    return summ


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    lf = open(os.path.join(OUT, f'eval_1x_native_a1_seed{SEED}_{stamp}.log'), 'w')

    def log(s):
        print(s); lf.write(s + '\n'); lf.flush()
    log(f'Scenario: {SCEN}\nReward: CybORG native Blue reward (sum of result.reward)\n'
        f'Execution: deterministic, {N_EP} episodes x {T} steps, Red=B_lineAgent, seed={SEED}')
    specs = [('PPO (Flat)', 'ppo_cyborg_fixedflat_shaped_vulnerable.zip', PPO, 'flat'),
             ('LSTM-PPO', 'lstm_ppo_hierarchical_v1.zip', RecurrentPPO, 'lstm'),
             ('PPO (Hierarchical) [reference re-run]', 'ppo_hierarchical_v1.zip', PPO, 'hier'),
             ('DQN [reference re-run]', 'dqn_hierarchical_v1.zip', DQN, 'hier')]
    results = []
    for name, f, cls, kind in specs:
        m = cls.load(os.path.join(MDIR, f), device='cpu')
        log(f'\n=== {name}: {f} | obs {m.observation_space.shape} | actions {m.action_space} ===')
        results.append(run(name, m, kind, log))
    json.dump(results, open(os.path.join(OUT, f'eval_1x_native_a1_seed{SEED}_{stamp}.json'), 'w'), indent=1)
    lf.close()


if __name__ == '__main__':
    main()
