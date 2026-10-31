(a1) re-evaluation, 2026-10-30 (run in a Linux cloud container, CPU)
- CybORG: cage-challenge-1 @ 6d5d9fd (same commit as the RL_CYB submodule)
- Scenario: cage-challenge-1/CybORG/CybORG/Shared/Scenarios/Scenario1b-vulnerable.yaml (copy staged from the 5090)
- Protocol: Red = B_lineAgent, CybORG native Blue reward, deterministic, 10 episodes x 100 steps, seed 0
  (random/numpy/torch seeded; CybORG sim itself is otherwise unseeded, as in the original scripts)
- eval_1x_native_a1.py           -> eval_1x_native_a1_seed0_*.log / .json
    usage: python eval_1x_native_a1.py <Scenario1b-vulnerable.yaml> <ppo_out dir> <out dir> [seed]
- decompose_ppo_flat_reward.py <scenario.yaml> <ppo_flat.zip> -> decompose_ppo_flat_reward_seed0.log  (3 episodes; restore cost vs breach/outage, Red action mix)
Paths in the scripts are command-line arguments; on the 5090 pass the D:\Vasanth\RL_CYB paths.
