"""Per-state policy entropy and top-1/top-2 logit margin of PPO (Hierarchical) on the states it visits
under the Table 1 protocol (modified Scenario 1b, Red=B_lineAgent, argmax, 10 x 100, seed 0).
Usage: python ppo_hier_entropy_margin.py <Scenario1b-vulnerable.yaml> <ppo_hierarchical_v1.zip> [seed]"""
import sys, random, numpy as np, torch
from CybORG import CybORG
from CybORG.Agents import B_lineAgent
from CybORG.Agents.Wrappers import EnumActionWrapper, FixedFlatWrapper, BlueTableWrapper
from stable_baselines3 import PPO
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
TY = ['Monitor', 'Analyse', 'Remove', 'Restore']
H = ['Defender','Enterprise0','Enterprise1','Enterprise2','Op_Host0','Op_Host1','Op_Host2','Op_Server0','User0','User1','User2','User3','User4']
m = PPO.load(sys.argv[2], device='cpu')
c = CybORG(sys.argv[1], 'sim', agents={'Red': B_lineAgent})
t = BlueTableWrapper(env=c, output_mode='vector'); e = EnumActionWrapper(env=t); f = FixedFlatWrapper(env=c, agent='Blue')
e.action_space_change(c.get_action_space('Blue')); A = e.possible_actions
amap = {}
for a in A:
    s = str(a)
    for ti, at in enumerate(TY):
        if s.startswith(at):
            if at == 'Monitor':
                for hi in range(len(H)): amap[(ti, hi)] = a
            else:
                for hi, h in enumerate(H):
                    if h in s: amap[(ti, hi)] = a; break
            break
sl = next(a for a in A if 'Sleep' in str(a))
ent, marg, pmax = [], [], []
for ep in range(10):
    c.reset()
    for k in range(100):
        o = np.array(f.get_observation('Blue'), dtype=np.float32).flatten()
        with torch.no_grad():
            d = m.policy.get_distribution(torch.as_tensor(o).unsqueeze(0))
            logits = d.distribution.logits[0]
            ent.append(float(d.entropy()[0])); p = torch.softmax(logits, -1)
            top = torch.topk(logits, 2).values; marg.append(float(top[0] - top[1])); pmax.append(float(p.max()))
            a = int(torch.argmax(logits))
        c.step(agent='Blue', action=amap.get((a // 13, a % 13), sl))
ent, marg, pmax = map(np.array, (ent, marg, pmax))
print(f'states={len(ent)}  ln(52)={np.log(52):.3f}')
print(f'entropy nats: mean {ent.mean():.3f}  median {np.median(ent):.3f}  min {ent.min():.3f}  max {ent.max():.3f}')
print(f'top1-top2 logit margin: mean {marg.mean():.3f}  median {np.median(marg):.3f}  min {marg.min():.3f}  max {marg.max():.3f}')
print(f'argmax-action probability: mean {pmax.mean():.3f}  median {np.median(pmax):.3f}')
