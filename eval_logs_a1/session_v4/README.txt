Camera-ready (v4) evaluation session, 2026-10-30 — the numbers in Tables 1, 2, 4, 8, 10 and Figure 2 come from here.
Run in a Linux cloud container (CPU), CybORG cage-challenge-1 @ 6d5d9fd (same commit as the RL_CYB submodule).
Scenario: cage-challenge-1/CybORG/CybORG/Shared/Scenarios/Scenario1b-vulnerable.yaml; Red = B_lineAgent;
CybORG native Blue reward; 10 episodes x 100 steps; seed 0 (random/numpy/torch; the CybORG sim is otherwise unseeded).

eval_1x_native_a1.py <scenario.yaml> <ppo_out dir> <out dir> [seed]
  -> eval_1x_native_a1_seed0_20261030_232059.log / .json  (ONE session, in this order:
     PPO (Flat), LSTM-PPO, PPO (Hier.), DQN [argmax]; PPO (Hier.) sampled; No defense (Sleep))
ppo_hier_entropy_margin.py <scenario.yaml> <ppo_hierarchical_v1.zip> [seed]
  -> ppo_hier_entropy_margin_seed0.log  (entropy 3.535 nats / logit margin 0.065 in every visited state)
decompose_ppo_flat_reward.py <scenario.yaml> <ppo_cyborg_fixedflat_shaped_vulnerable.zip>
  -> decompose_ppo_flat_reward_seed0.log  (3 episodes: Restore cost -100, breach ~ -194, no outage step)
