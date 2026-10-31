Table 3 (3x) re-run under one protocol — 2026-10-30, Linux cloud container (CPU), CybORG cage-challenge-1 @ 6d5d9fd.
Protocol = eval_dqn_3x.py::evaluate_vs_red: CybORG3xEnv(Scenario3x.yaml, max_steps=100, red_agent=<profile>, obs_delay=1),
10 episodes x 100 steps per seed, seeds 0,1,2 (random, numpy, torch, action_space) -> 30 episodes per cell.
Reward = wrapper step reward = CybORG native reward + Isolate/Patch shaping (Isolate -0.5 first / -0.1 repeat; Patch +0.5 first;
+0.2 for other actions on an isolated host). DQN: dqn_3x_v1.zip, deterministic=True. Random: env.action_space.sample().
usage: python table3_3x.py <dir with train_dqn_3x.py> <Scenario3x.yaml> <dqn_3x_v1.zip> <out.json>

Pooled mean +/- std (n=30) and per-seed means (seed 0 / 1 / 2):
  DQN    B_line   -279.8 +/- 3.1     (-280.3 / -280.6 / -278.5)
  DQN    Meander  -308.5 +/- 41.2    (-326.2 / -300.5 / -298.9)
  DQN    Sleep     -87.5 +/- 0.0     (-87.5 / -87.5 / -87.5)
  Random B_line   -784.5 +/- 348.9   (-754.0 / -875.8 / -723.8)
  Random Meander  -415.2 +/- 236.5   (-388.6 / -463.8 / -393.2)
  Random Sleep     -13.2 +/- 4.4     (-14.9 / -12.1 / -12.6)
Derived: active-adversary mean DQN -294.2 vs random -599.9 (51.0% lower penalty);
overall DQN -225.3 vs random -404.3 (44.3% lower); Sleep: DQN/random = 6.6x.
DQN executes the same actions under every profile: Restore Op_Server0 90%, Patch 10% (Op_Host1, Research0, Enterprise0, User1,
DMZ_Server0; 2 each per episode), 5 action changes per 50 steps.
The earlier random_3x_sleep/ runs (same seeds) give the identical Sleep values.
