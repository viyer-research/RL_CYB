"""
3x Scaled DQN Evaluation — Detailed Policy Analysis
====================================================
Evaluates the trained DQN agent on the 3x scenario with:
  - Per-Red-agent breakdown (meander, bline, sleep)
  - Q-value analysis and action distribution
  - State-dependency check
  - Comparison with random baseline
  - Per-subnet action targeting analysis
"""

import sys
import os
import numpy as np
import torch
from collections import defaultdict, Counter

sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from stable_baselines3 import DQN
from train_dqn_3x import (
    CybORG3xEnv, ALL_HOSTS, ACTION_TYPES, RED_AGENT_NAMES,
    N_HOSTS, N_ACTION_TYPES
)

MODEL_PATH = r'D:\Vasanth\RL_CYB\ppo_out\dqn_3x_v1.zip'

# Subnet membership for analysis
SUBNET_MAP = {
    'User':        ['User0', 'User1', 'User2', 'User3', 'User4', 'User5', 'User6'],
    'Enterprise':  ['Enterprise0', 'Enterprise1', 'Enterprise2', 'Defender'],
    'Operational': ['Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0'],
    'DMZ':         ['DMZ_Server0', 'DMZ_Server1'],
    'Research':    ['Research0', 'Research1', 'Research2'],
}

def get_host_subnet(host):
    for subnet, hosts in SUBNET_MAP.items():
        if host in hosts:
            return subnet
    return 'Unknown'


def decode_action(action):
    type_idx = action // N_HOSTS
    host_idx = action % N_HOSTS
    return ACTION_TYPES[type_idx], ALL_HOSTS[host_idx]


def evaluate_vs_red(model, red_name, n_episodes=10, max_steps=100,
                    deterministic=True, verbose=True):
    """Evaluate model against a specific Red agent."""
    env = CybORG3xEnv(max_steps=max_steps, red_agent=red_name, obs_delay=1)

    episode_rewards = []
    action_counts = defaultdict(int)
    type_counts = defaultdict(int)
    subnet_counts = defaultdict(int)
    unique_per_ep = []
    isolation_counts = []
    patch_counts = []

    for ep in range(n_episodes):
        obs, info = env.reset()
        ep_reward = 0
        ep_actions = []

        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _, info = env.step(action)
            ep_reward += reward

            a_name = info['action_name']
            a_type = info['action_type']
            a_host = info['host']
            action_counts[a_name] += 1
            type_counts[a_type] += 1
            subnet_counts[get_host_subnet(a_host)] += 1
            ep_actions.append(a_name)

            if done:
                break

        episode_rewards.append(ep_reward)
        unique_per_ep.append(len(set(ep_actions)))
        isolation_counts.append(info.get('isolated_count', 0))
        patch_counts.append(info.get('patched_count', 0))

    mean_r = np.mean(episode_rewards)
    std_r = np.std(episode_rewards)
    mean_unique = np.mean(unique_per_ep)

    if verbose:
        total_a = sum(type_counts.values()) or 1
        total_s = sum(subnet_counts.values()) or 1

        print(f"\n  {'='*60}")
        print(f"  vs {red_name.upper()} ({'deterministic' if deterministic else 'stochastic'})")
        print(f"  {'='*60}")
        print(f"  Reward:    {mean_r:.1f} ± {std_r:.1f}  "
              f"(min={min(episode_rewards):.1f}, max={max(episode_rewards):.1f})")
        print(f"  Unique actions/ep: {mean_unique:.1f}")
        print(f"  Avg isolated: {np.mean(isolation_counts):.1f}, "
              f"Avg patched: {np.mean(patch_counts):.1f}")

        print(f"\n  Action Type Distribution:")
        for t in ACTION_TYPES:
            bar = '█' * int(type_counts[t] / total_a * 40)
            print(f"    {t:10s}: {type_counts[t]:5d} ({type_counts[t]/total_a*100:5.1f}%) {bar}")

        print(f"\n  Subnet Targeting:")
        for s in SUBNET_MAP:
            bar = '█' * int(subnet_counts[s] / total_s * 40)
            print(f"    {s:12s}: {subnet_counts[s]:5d} ({subnet_counts[s]/total_s*100:5.1f}%) {bar}")

        print(f"\n  Top 10 Actions:")
        for action, count in sorted(action_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    {action:35s}: {count:4d} ({count/total_a*100:.1f}%)")

    return mean_r, std_r, type_counts, action_counts


def evaluate_random_baseline(n_episodes=3, max_steps=100):
    """Random action baseline for comparison."""
    print(f"\n{'='*60}")
    print(f"  RANDOM BASELINE ({n_episodes} episodes per Red)")
    print(f"{'='*60}")

    results = {}
    for red_name in RED_AGENT_NAMES:
        env = CybORG3xEnv(max_steps=max_steps, red_agent=red_name, obs_delay=1)
        episode_rewards = []

        for ep in range(n_episodes):
            obs, _ = env.reset()
            ep_reward = 0
            for step in range(max_steps):
                action = env.action_space.sample()
                obs, reward, done, _, info = env.step(action)
                ep_reward += reward
                if done:
                    break
            episode_rewards.append(ep_reward)
            print(f"    [{red_name}] ep {ep+1}/{n_episodes}: {ep_reward:.1f}")

        mean_r = np.mean(episode_rewards)
        std_r = np.std(episode_rewards)
        results[red_name] = (mean_r, std_r)
        print(f"  vs {red_name:10s}: {mean_r:.1f} ± {std_r:.1f}")

    return results


def q_value_analysis(model, red_name='meander'):
    """Analyse Q-values at initial state."""
    print(f"\n{'='*60}")
    print(f"  Q-VALUE ANALYSIS (initial state, vs {red_name})")
    print(f"{'='*60}")

    env = CybORG3xEnv(max_steps=100, red_agent=red_name, obs_delay=1)
    obs, _ = env.reset()

    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(model.device)
    with torch.no_grad():
        q_values = model.q_net(obs_tensor).cpu().numpy().flatten()

    print(f"\n  Q-value statistics:")
    print(f"    Mean: {q_values.mean():.2f}")
    print(f"    Std:  {q_values.std():.2f}")
    print(f"    Min:  {q_values.min():.2f}")
    print(f"    Max:  {q_values.max():.2f}")
    gap = np.sort(q_values)[-1] - np.sort(q_values)[-2]
    print(f"    Gap (max - 2nd): {gap:.4f}")

    # Per action type
    print(f"\n  Q-values by action type:")
    for type_idx, action_type in enumerate(ACTION_TYPES):
        type_q = [q_values[type_idx * N_HOSTS + h] for h in range(N_HOSTS)]
        print(f"    {action_type:10s}: mean={np.mean(type_q):7.2f}, "
              f"max={np.max(type_q):7.2f}, min={np.min(type_q):7.2f}")

    # Per subnet
    print(f"\n  Q-values by target subnet:")
    for subnet, hosts in SUBNET_MAP.items():
        subnet_q = []
        for host in hosts:
            host_idx = ALL_HOSTS.index(host)
            for type_idx in range(N_ACTION_TYPES):
                subnet_q.append(q_values[type_idx * N_HOSTS + host_idx])
        print(f"    {subnet:12s}: mean={np.mean(subnet_q):7.2f}, "
              f"max={np.max(subnet_q):7.2f}")

    # Top 15 actions
    print(f"\n  Top 15 actions by Q-value:")
    top_15 = np.argsort(q_values)[-15:][::-1]
    for rank, idx in enumerate(top_15):
        a_type, a_host = decode_action(idx)
        subnet = get_host_subnet(a_host)
        print(f"    {rank+1:2d}. {a_type:10s} {a_host:15s} [{subnet:10s}]: "
              f"Q={q_values[idx]:7.2f}")


def state_dependency_check(model, red_name='meander'):
    """Check if the policy changes actions based on state."""
    print(f"\n{'='*60}")
    print(f"  STATE-DEPENDENCY CHECK (vs {red_name})")
    print(f"{'='*60}")

    env = CybORG3xEnv(max_steps=100, red_agent=red_name, obs_delay=1)

    for ep in range(3):
        obs, _ = env.reset()
        prev_action = None
        action_changes = 0
        actions_taken = []

        print(f"\n  --- Episode {ep+1} ---")
        for step in range(50):
            action, _ = model.predict(obs, deterministic=True)
            action = int(action)
            a_type, a_host = decode_action(action)

            if prev_action is not None and action != prev_action:
                action_changes += 1
                if action_changes <= 8:
                    prev_type, prev_host = decode_action(prev_action)
                    print(f"    Step {step:2d}: {prev_type} {prev_host} "
                          f"→ {a_type} {a_host}")

            actions_taken.append(f"{a_type} {a_host}")
            prev_action = action
            obs, _, done, _, _ = env.step(action)
            if done:
                break

        unique = len(set(actions_taken))
        print(f"    Changes: {action_changes}/50, Unique: {unique}, "
              f"{'STATE-DEPENDENT ✓' if action_changes > 5 else 'MOSTLY STATIC ✗'}")


def main():
    print("=" * 70)
    print("  3x DQN Policy Evaluation")
    print("=" * 70)

    # Load model
    if not os.path.exists(MODEL_PATH):
        print(f"\n  ❌ Model not found: {MODEL_PATH}")
        print(f"  Run train_dqn_3x.py first!")
        return

    print(f"\n  Loading model: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    # 1. Evaluate vs each Red agent
    dqn_results = {}
    for red_name in RED_AGENT_NAMES:
        m, s, _, _ = evaluate_vs_red(model, red_name,
                                      n_episodes=10, deterministic=True)
        dqn_results[red_name] = (m, s)

    # 2. Random baseline
    random_results = evaluate_random_baseline(n_episodes=3)

    # 3. Q-value analysis
    q_value_analysis(model, red_name='meander')

    # 4. State dependency
    state_dependency_check(model, red_name='meander')

    # 5. Summary comparison
    print(f"\n{'='*70}")
    print(f"  FINAL COMPARISON: DQN vs RANDOM")
    print(f"{'='*70}")
    print(f"  {'Red Agent':12s} {'DQN':>15s} {'Random':>15s} {'Improvement':>15s}")
    print(f"  {'-'*57}")

    improvements = []
    for red_name in RED_AGENT_NAMES:
        dm, ds = dqn_results[red_name]
        rm, rs = random_results[red_name]
        imp = dm - rm
        improvements.append(imp)
        print(f"  {red_name:12s} {dm:7.1f}±{ds:4.1f} {rm:7.1f}±{rs:4.1f} "
              f"  {'+' if imp > 0 else ''}{imp:.1f}")

    dqn_overall = np.mean([m for m, _ in dqn_results.values()])
    rnd_overall = np.mean([m for m, _ in random_results.values()])
    print(f"\n  {'Overall':12s} {dqn_overall:7.1f}       {rnd_overall:7.1f}       "
          f"  {'+' if dqn_overall > rnd_overall else ''}{dqn_overall-rnd_overall:.1f}")

    # Scaling comparison
    print(f"\n{'='*70}")
    print(f"  3x SCALING SUMMARY")
    print(f"{'='*70}")
    print(f"  Network:     20 hosts, 5 subnets (vs 13 hosts, 3 subnets)")
    print(f"  Actions:     {N_ACTION_TYPES} types x {N_HOSTS} hosts = "
          f"{N_ACTION_TYPES * N_HOSTS} (vs 52)")
    print(f"  Red agents:  {len(RED_AGENT_NAMES)} profiles (vs 1)")
    print(f"  Obs delay:   1 step (vs 0)")
    print(f"  New actions: Isolate, Patch")
    print(f"  DQN overall: {dqn_overall:.1f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
