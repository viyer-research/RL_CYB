"""
Generate training plots for the LaTeX paper.
Produces:
  1. training_1x_comparison.png  — 1× PPO vs LSTM-PPO vs DQN learning curves
  2. training_3x_curves.png      — 3× DQN detailed training panels
  3. training_1x_vs_3x.png       — Side-by-side 1× vs 3× DQN comparison
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

# Try tensorboard, fall back to manual
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    HAS_TB = True
except ImportError:
    HAS_TB = False
    print("WARNING: tensorboard not available, using fallback")

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'legend.fontsize': 9,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

OUTPUT_DIR = r'D:\Vasanth\RL_CYB'

# ─── TensorBoard log paths ───
LOGS_1X = {
    'PPO (Hierarchical)': r'D:\Vasanth\RL_CYB\ppo_out\PPO_hierarchical_v1_1',
    'LSTM-PPO':           r'D:\Vasanth\RL_CYB\ppo_out\lstm_ppo_logs\RecurrentPPO_1',
    'DQN':                r'D:\Vasanth\RL_CYB\ppo_out\dqn_logs\DQN_1',
}

LOGS_3X = {
    'DQN 3x': r'D:\Vasanth\RL_CYB\ppo_out\dqn_3x_logs\DQN_1',
}

# Also include some early PPO runs for comparison
LOGS_PPO_EARLY = {
    'PPO Run 1': r'D:\Vasanth\RL_CYB\cage-challenge-1\ppo_cyborg_tensorboard\PPO_1',
    'PPO Run 4': r'D:\Vasanth\RL_CYB\cage-challenge-1\ppo_cyborg_tensorboard\PPO_4',
    'PPO Run 7': r'D:\Vasanth\RL_CYB\cage-challenge-1\ppo_cyborg_tensorboard\PPO_7',
}


def load_tb_scalars(logdir, tags=None):
    """Load scalar data from TensorBoard event files."""
    if not HAS_TB or not os.path.exists(logdir):
        return {}
    
    ea = EventAccumulator(logdir)
    ea.Reload()
    
    available = ea.Tags().get('scalars', [])
    if tags is None:
        tags = available
    
    data = {}
    for tag in tags:
        if tag in available:
            events = ea.Scalars(tag)
            steps = [e.step for e in events]
            values = [e.value for e in events]
            data[tag] = (np.array(steps), np.array(values))
    
    return data


def smooth(values, weight=0.9):
    """Exponential moving average."""
    smoothed = []
    last = values[0] if len(values) > 0 else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return np.array(smoothed)


def plot_1x_comparison():
    """Figure 1: 1× algorithm comparison — loss, entropy, exploration."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    colors = {
        'PPO (Hierarchical)': '#e74c3c',
        'LSTM-PPO': '#f39c12',
        'DQN': '#2ecc71',
    }
    
    # Panel 1: Training Loss (all algorithms have this)
    ax = axes[0]
    ax.set_title('(a) Training Loss')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    
    for name, logdir in LOGS_1X.items():
        data = load_tb_scalars(logdir)
        if 'train/loss' in data:
            steps, vals = data['train/loss']
            ax.plot(steps, smooth(vals, 0.95), label=name, color=colors[name],
                    linewidth=2)
            ax.plot(steps, vals, color=colors[name], alpha=0.08)
    
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Panel 2: PPO entropy vs DQN exploration
    ax = axes[1]
    ax.set_title('(b) Exploration Behaviour')
    ax.set_xlabel('Training Steps')
    
    # PPO entropy loss (negated so higher = more exploration)
    for name in ['PPO (Hierarchical)', 'LSTM-PPO']:
        data = load_tb_scalars(LOGS_1X[name])
        if 'train/entropy_loss' in data:
            steps, vals = data['train/entropy_loss']
            # Entropy loss is negative; plot absolute value
            ax.plot(steps, smooth(-vals, 0.9), label=f'{name} (entropy)',
                    color=colors[name], linewidth=2)
    
    ax.set_ylabel('Entropy (PPO)', color='#e74c3c')
    ax.tick_params(axis='y', labelcolor='#e74c3c')
    
    # DQN epsilon on secondary axis
    ax2 = ax.twinx()
    data_dqn = load_tb_scalars(LOGS_1X['DQN'])
    if 'rollout/exploration_rate' in data_dqn:
        steps, vals = data_dqn['rollout/exploration_rate']
        ax2.plot(steps, vals, label='DQN (ε)', color=colors['DQN'],
                 linewidth=2, linestyle='--')
        ax2.set_ylabel('ε-greedy (DQN)', color=colors['DQN'])
        ax2.tick_params(axis='y', labelcolor=colors['DQN'])
    
    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Panel 3: PPO action type distribution (shows near-uniform)
    ax = axes[2]
    ax.set_title('(c) PPO Action Type Distribution')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Action Type %')
    
    data_ppo = load_tb_scalars(LOGS_1X['PPO (Hierarchical)'])
    act_colors_bar = {'Analyse': '#e74c3c', 'Monitor': '#9b59b6',
                      'Remove': '#f39c12', 'Restore': '#2ecc71'}
    for atype, color in act_colors_bar.items():
        tag = f'action_type/{atype}'
        if tag in data_ppo:
            steps, vals = data_ppo[tag]
            ax.plot(steps, vals, label=atype, color=color, linewidth=2)
    
    ax.axhline(y=25, color='gray', linestyle=':', alpha=0.5, label='Uniform (25%)')
    ax.set_ylim(15, 35)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('1× Scenario: Algorithm Comparison (PPO vs LSTM-PPO vs DQN)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_1x_comparison.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_3x_curves():
    """Figure 2: 3× DQN detailed training panels."""
    logdir = list(LOGS_3X.values())[0]
    data = load_tb_scalars(logdir)
    
    if not data:
        print("No 3x data found, skipping")
        return None
    
    print(f"  3x tags available: {list(data.keys())}")
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    # (a) Overall eval reward
    ax = axes[0, 0]
    ax.set_title('(a) Eval Reward (Overall)')
    for tag in ['eval/overall_mean_reward']:
        if tag in data:
            steps, vals = data[tag]
            ax.plot(steps, smooth(vals, 0.7), color='#2ecc71', linewidth=2,
                    label='Mean')
            ax.fill_between(steps, vals, smooth(vals, 0.7), alpha=0.15,
                            color='#2ecc71')
            # Add std if available
            if 'eval/overall_std_reward' in data:
                _, std_vals = data['eval/overall_std_reward']
                sv = smooth(vals, 0.7)
                ax.fill_between(steps, sv - std_vals, sv + std_vals,
                                alpha=0.1, color='#2ecc71', label='±1 Std')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Per-Red reward
    ax = axes[0, 1]
    ax.set_title('(b) Reward by Red Agent')
    red_colors = {'meander': '#e74c3c', 'bline': '#3498db', 'sleep': '#95a5a6'}
    for red_name, color in red_colors.items():
        tag = f'eval/{red_name}_mean_reward'
        if tag in data:
            steps, vals = data[tag]
            ax.plot(steps, smooth(vals, 0.7), color=color, linewidth=2,
                    label=red_name.capitalize())
    ax.set_xlabel('Steps')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Training loss
    ax = axes[0, 2]
    ax.set_title('(c) Training Loss')
    if 'train/loss' in data:
        steps, vals = data['train/loss']
        ax.plot(steps, smooth(vals, 0.95), color='#e67e22', linewidth=2)
        ax.plot(steps, vals, color='#e67e22', alpha=0.1)
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.grid(True, alpha=0.3)
    
    # (d) Action distribution (stacked area)
    ax = axes[1, 0]
    ax.set_title('(d) Action Distribution')
    action_types = ['Restore', 'Patch', 'Monitor', 'Analyse', 'Remove', 'Isolate']
    act_colors = ['#2ecc71', '#3498db', '#9b59b6', '#e74c3c', '#f39c12', '#1abc9c']
    act_data = {}
    common_steps = None
    for atype in action_types:
        tag = f'eval/action_pct_{atype}'
        if tag in data:
            steps, vals = data[tag]
            act_data[atype] = vals
            if common_steps is None:
                common_steps = steps
    
    if act_data and common_steps is not None:
        bottom = np.zeros(len(common_steps))
        for atype, color in zip(action_types, act_colors):
            if atype in act_data:
                vals = act_data[atype]
                # Ensure same length
                min_len = min(len(common_steps), len(vals))
                ax.fill_between(common_steps[:min_len], bottom[:min_len],
                                bottom[:min_len] + vals[:min_len],
                                alpha=0.7, color=color, label=atype)
                bottom[:min_len] += vals[:min_len]
    ax.set_xlabel('Steps')
    ax.set_ylabel('Action %')
    ax.legend(loc='center right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # (e) Exploration rate
    ax = axes[1, 1]
    ax.set_title('(e) Exploration Rate (ε)')
    if 'rollout/exploration_rate' in data:
        steps, vals = data['rollout/exploration_rate']
        ax.plot(steps, vals, color='#8e44ad', linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel('ε')
    ax.grid(True, alpha=0.3)
    
    # (f) Unique actions & patching
    ax = axes[1, 2]
    ax.set_title('(f) Policy Diversity & Patching')
    ax2 = ax.twinx()
    if 'eval/unique_actions_used' in data:
        steps, vals = data['eval/unique_actions_used']
        ax.plot(steps, vals, color='#2c3e50', linewidth=2, label='Unique Actions')
    if 'eval/avg_patched_per_ep' in data:
        steps, vals = data['eval/avg_patched_per_ep']
        ax2.plot(steps, vals, color='#3498db', linewidth=2, linestyle='--',
                 label='Hosts Patched')
    ax.set_xlabel('Steps')
    ax.set_ylabel('Unique Actions', color='#2c3e50')
    ax2.set_ylabel('Hosts Patched', color='#3498db')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('3× Scenario: DQN Training Progress (750K Steps, RTX 5090)',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_3x_curves.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


def plot_1x_vs_3x():
    """Figure 3: Side-by-side 1× DQN vs 3× DQN comparison."""
    data_1x = load_tb_scalars(LOGS_1X['DQN'])
    data_3x = load_tb_scalars(list(LOGS_3X.values())[0])
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    
    # (a) Reward curves
    ax = axes[0]
    ax.set_title('(a) Eval Reward Comparison')
    
    # 1x DQN
    for tag in ['rollout/ep_rew_mean', 'eval/mean_reward',
                'eval/overall_mean_reward']:
        if tag in data_1x:
            steps, vals = data_1x[tag]
            ax.plot(steps, smooth(vals, 0.8), color='#3498db', linewidth=2,
                    label='1× DQN (vs B_line)')
            ax.plot(steps, vals, color='#3498db', alpha=0.1)
            break
    
    # 3x DQN
    if 'eval/overall_mean_reward' in data_3x:
        steps, vals = data_3x['eval/overall_mean_reward']
        ax.plot(steps, smooth(vals, 0.8), color='#e74c3c', linewidth=2,
                label='3× DQN (overall)')
        ax.plot(steps, vals, color='#e74c3c', alpha=0.1)
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (b) Loss comparison
    ax = axes[1]
    ax.set_title('(b) Training Loss')
    
    if 'train/loss' in data_1x:
        steps, vals = data_1x['train/loss']
        ax.plot(steps, smooth(vals, 0.95), color='#3498db', linewidth=2,
                label='1× DQN')
    if 'train/loss' in data_3x:
        steps, vals = data_3x['train/loss']
        ax.plot(steps, smooth(vals, 0.95), color='#e74c3c', linewidth=2,
                label='3× DQN')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # (c) Exploration rate
    ax = axes[2]
    ax.set_title('(c) Exploration Rate (ε)')
    
    if 'rollout/exploration_rate' in data_1x:
        steps, vals = data_1x['rollout/exploration_rate']
        ax.plot(steps, vals, color='#3498db', linewidth=2, label='1× DQN')
    if 'rollout/exploration_rate' in data_3x:
        steps, vals = data_3x['rollout/exploration_rate']
        ax.plot(steps, vals, color='#e74c3c', linewidth=2, label='3× DQN')
    
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('ε')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle('1× vs 3× DQN Training Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'training_1x_vs_3x.png')
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved: {path}")
    return path


if __name__ == '__main__':
    print("="*60)
    print("  Generating Training Plots for Paper")
    print("="*60)
    
    # First, scan what tags are available
    print("\n--- Available TensorBoard tags ---")
    for name, logdir in {**LOGS_1X, **LOGS_3X}.items():
        data = load_tb_scalars(logdir)
        tags = list(data.keys())
        print(f"  {name}: {len(tags)} tags")
        for t in tags[:15]:
            steps, vals = data[t]
            print(f"    {t}: {len(steps)} points, range [{vals.min():.2f}, {vals.max():.2f}]")
    
    print("\n--- Generating plots ---")
    p1 = plot_1x_comparison()
    p2 = plot_3x_curves()
    p3 = plot_1x_vs_3x()
    
    print("\n--- Done ---")
    print(f"  {p1}")
    print(f"  {p2}")
    print(f"  {p3}")
