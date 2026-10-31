3x random baseline, Sleep (no adversary) profile — re-run 2026-10-30 (Linux cloud container, CPU).
Protocol = eval_dqn_3x.py::evaluate_vs_red for DQN: CybORG3xEnv(Scenario3x.yaml, max_steps=100, red_agent='sleep',
obs_delay=1), 10 episodes x 100 steps, reward = wrapper step reward (native + Isolate/Patch shaping).
Policy = env.action_space.sample() (uniform over 120 actions), as in evaluate_random_baseline. Seeds: random, numpy, action_space.
CybORG: cage-challenge-1 @ 6d5d9fd. Environment code: train_dqn_3x.py from this repo (unchanged; scenario path passed in).
usage: python random_3x_sleep.py <dir with train_dqn_3x.py> <Scenario3x.yaml> [seed]
seed 0: -14.9 +/- 5.6 | seed 1: -12.1 +/- 3.1 | seed 2: -12.6 +/- 3.3   (paper v4 had -8.7 +/- 0.7, N=3, no log)
Each log ends with a per-action-type decomposition (native vs shaping reward).
