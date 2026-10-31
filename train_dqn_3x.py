"""
3x Scaled DQN Training for CybORG Blue Agent
=============================================
Scaling dimensions vs Scenario1b baseline:
  - 20 hosts (vs 13)  across 5 subnets (vs 3)
  - 3 Red agent profiles (Meander, B_line, Sleep) with random selection
  - 2 new Blue meta-actions: IsolateHost, PatchVulnerability
  - 1-step observation delay (partial observability)
  - BlueTableWrapper vector obs (compact 4-features/host, scales cleanly)
  - Hierarchical action space: action_type x host
"""

import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import random as pyrandom

# Add CybORG to path
sys.path.insert(0, r'D:\Vasanth\RL_CYB\cage-challenge-1\CybORG')

from CybORG import CybORG
from CybORG.Agents import B_lineAgent, RedMeanderAgent, SleepAgent
from CybORG.Agents.Wrappers import BlueTableWrapper, EnumActionWrapper

# ============================================================
#  Configuration
# ============================================================
SCENARIO_PATH = r'D:\Vasanth\RL_CYB\Scenario3x.yaml'
OUTPUT_DIR = r'D:\Vasanth\RL_CYB\ppo_out'

# 3x topology: 5 subnets, 20 hosts
ALL_HOSTS = [
    'Defender',
    'Enterprise0', 'Enterprise1', 'Enterprise2',
    'Op_Host0', 'Op_Host1', 'Op_Host2', 'Op_Server0',
    'User0', 'User1', 'User2', 'User3', 'User4', 'User5', 'User6',
    'DMZ_Server0', 'DMZ_Server1',
    'Research0', 'Research1', 'Research2',
]
N_HOSTS = len(ALL_HOSTS)  # 20

# Extended action types: original 4 + IsolateHost + PatchVulnerability
ACTION_TYPES = ['Monitor', 'Analyse', 'Remove', 'Restore', 'Isolate', 'Patch']
N_ACTION_TYPES = len(ACTION_TYPES)

# Red agent pool — randomly selected per episode
RED_AGENTS = {
    'meander': RedMeanderAgent,
    'bline':   B_lineAgent,
    'sleep':   SleepAgent,
}
RED_AGENT_NAMES = list(RED_AGENTS.keys())

# Observation delay (1-step lag for partial observability)
OBS_DELAY_STEPS = 1


# ============================================================
#  Custom 3x Environment
# ============================================================
class CybORG3xEnv(gym.Env):
    """
    3x scaled CybORG environment with:
      - BlueTableWrapper vector observation (compact, scales with hosts)
      - Hierarchical action space: action_type x host
      - Red agent randomisation per episode
      - 1-step observation delay
      - IsolateHost / PatchVulnerability meta-actions
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_path=SCENARIO_PATH, max_steps=100,
                 obs_delay=OBS_DELAY_STEPS, red_agent=None):
        super().__init__()

        self.scenario_path = scenario_path
        self.max_steps = max_steps
        self.obs_delay = obs_delay
        self.fixed_red_agent = red_agent  # None = random each episode

        # Hierarchical action space
        self.n_action_types = N_ACTION_TYPES
        self.n_hosts = N_HOSTS
        self.action_space = spaces.Discrete(self.n_action_types * self.n_hosts)

        # Create initial environment to determine obs shape
        self._create_env('meander')

        # Observation space: BlueTableWrapper vector = 4 features per host
        # [activity_scan, activity_exploit, compromised_unknown, compromised_level]
        # + 2 extra features per host for our meta-actions (isolated, patched)
        # + 1 for observation delay indicator
        self.features_per_host = 6  # 4 from BlueTable + 1 isolated + 1 patched
        obs_size = self.features_per_host * N_HOSTS + 1  # +1 for delay flag
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32
        )

        # Internal state tracking
        self._t = 0
        self._obs_buffer = []
        self._isolated_hosts = set()
        self._patched_hosts = set()
        self._red_agent_name = 'meander'

        # Build CybORG action mapping
        self._build_action_map()

        print(f"[3x ENV] {N_HOSTS} hosts, {self.action_space.n} actions "
              f"({N_ACTION_TYPES} types x {N_HOSTS} hosts), "
              f"obs_dim={obs_size}, delay={self.obs_delay}")

    def _create_env(self, red_name):
        """Create CybORG with a specific Red agent."""
        red_class = RED_AGENTS[red_name]
        self._cyborg = CybORG(self.scenario_path, "sim",
                              agents={'Red': red_class})
        self._table = BlueTableWrapper(env=self._cyborg, output_mode='vector')
        self._enum = EnumActionWrapper(env=self._table)
        self._enum.action_space_change(self._cyborg.get_action_space('Blue'))
        self._all_actions = self._enum.possible_actions
        self._red_agent_name = red_name

    def _build_action_map(self):
        """Map (action_type_idx, host_idx) -> CybORG action object."""
        self._action_map = {}
        cyborg_action_types = ['Monitor', 'Analyse', 'Remove', 'Restore']

        for action in self._all_actions:
            action_str = str(action)
            for type_idx, action_type in enumerate(cyborg_action_types):
                if action_str.startswith(action_type):
                    if action_type == 'Monitor':
                        # Monitor has no host parameter — map to all hosts
                        for host_idx in range(self.n_hosts):
                            self._action_map[(type_idx, host_idx)] = action
                    else:
                        for host_idx, host in enumerate(ALL_HOSTS):
                            if host in action_str:
                                self._action_map[(type_idx, host_idx)] = action
                                break
                    break

        # Sleep fallback
        self._sleep_action = None
        for action in self._all_actions:
            if 'Sleep' in str(action):
                self._sleep_action = action
                break

        # Fill gaps with Sleep
        for type_idx in range(self.n_action_types):
            for host_idx in range(self.n_hosts):
                if (type_idx, host_idx) not in self._action_map:
                    self._action_map[(type_idx, host_idx)] = self._sleep_action

    def _decode_action(self, action):
        """Decode flat action idx -> (type_idx, host_idx)."""
        type_idx = action // self.n_hosts
        host_idx = action % self.n_hosts
        return type_idx, host_idx

    def _get_cyborg_obs_vector(self):
        """Get compact observation from BlueTableWrapper."""
        try:
            vec = self._table.get_observation('Blue')
            if vec is None:
                vec = np.zeros(4 * N_HOSTS, dtype=np.float32)
            elif not isinstance(vec, np.ndarray):
                vec = np.array(vec, dtype=np.float32)
        except Exception:
            vec = np.zeros(4 * N_HOSTS, dtype=np.float32)

        # Pad or truncate to expected size
        expected = 4 * N_HOSTS
        if len(vec) < expected:
            vec = np.pad(vec, (0, expected - len(vec)), constant_values=0.0)
        elif len(vec) > expected:
            vec = vec[:expected]

        return vec.astype(np.float32)

    def _build_full_obs(self, cyborg_vec):
        """Augment CybORG obs with isolation/patch state + delay flag."""
        # cyborg_vec: [4 * N_HOSTS] from BlueTableWrapper
        # Reshape to per-host: [N_HOSTS, 4]
        host_features = cyborg_vec.reshape(N_HOSTS, 4) if len(cyborg_vec) == 4 * N_HOSTS \
            else np.zeros((N_HOSTS, 4), dtype=np.float32)

        # Add isolation and patch flags: [N_HOSTS, 6]
        full = np.zeros((N_HOSTS, self.features_per_host), dtype=np.float32)
        full[:, :4] = host_features
        for host_idx, host in enumerate(ALL_HOSTS):
            full[host_idx, 4] = 1.0 if host in self._isolated_hosts else 0.0
            full[host_idx, 5] = 1.0 if host in self._patched_hosts else 0.0

        # Flatten + delay indicator
        obs = np.concatenate([full.flatten(), [float(self._t < self.obs_delay)]])
        return obs.astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomise Red agent each episode (unless fixed)
        if self.fixed_red_agent is None:
            red_name = pyrandom.choice(RED_AGENT_NAMES)
        else:
            red_name = self.fixed_red_agent

        self._create_env(red_name)
        self._build_action_map()

        self._cyborg.reset()
        self._t = 0
        self._isolated_hosts = set()
        self._patched_hosts = set()
        self._obs_buffer = []

        cyborg_vec = self._get_cyborg_obs_vector()
        obs = self._build_full_obs(cyborg_vec)

        # Pre-fill obs buffer with initial obs for delay
        self._obs_buffer = [obs.copy()] * (self.obs_delay + 1)

        return obs, {'red_agent': red_name}

    def step(self, action):
        action = int(action)
        type_idx, host_idx = self._decode_action(action)
        host_name = ALL_HOSTS[host_idx]
        action_type = ACTION_TYPES[type_idx]

        reward = 0.0
        info = {'action_type': action_type, 'host': host_name,
                'red_agent': self._red_agent_name}

        # --- Handle meta-actions (Isolate / Patch) ---
        if action_type == 'Isolate':
            # Isolating a host blocks Red from using it for lateral movement
            # but causes availability penalty (like a milder Restore)
            if host_name not in self._isolated_hosts:
                self._isolated_hosts.add(host_name)
                # Execute a Misinform or Sleep in CybORG as proxy
                misinform_action = None
                for a in self._all_actions:
                    if 'Misinform' in str(a) and host_name in str(a):
                        misinform_action = a
                        break
                if misinform_action:
                    result = self._cyborg.step(agent='Blue', action=misinform_action)
                    reward = result.reward - 0.5  # isolation cost
                else:
                    result = self._cyborg.step(agent='Blue', action=self._sleep_action)
                    reward = result.reward - 0.5
            else:
                # Already isolated — wasted action
                result = self._cyborg.step(agent='Blue', action=self._sleep_action)
                reward = result.reward - 0.1

        elif action_type == 'Patch':
            # Patching removes vulnerability but takes effort
            if host_name not in self._patched_hosts:
                self._patched_hosts.add(host_name)
                # Execute Analyse + Remove as proxy for patching
                remove_action = self._action_map.get((2, host_idx))  # Remove
                if remove_action and 'Sleep' not in str(remove_action):
                    result = self._cyborg.step(agent='Blue', action=remove_action)
                    reward = result.reward + 0.5  # bonus for patching
                else:
                    result = self._cyborg.step(agent='Blue', action=self._sleep_action)
                    reward = result.reward + 0.3
            else:
                # Already patched
                result = self._cyborg.step(agent='Blue', action=self._sleep_action)
                reward = result.reward

        else:
            # --- Standard CybORG actions ---
            cyborg_action = self._action_map.get((type_idx, host_idx), self._sleep_action)
            result = self._cyborg.step(agent='Blue', action=cyborg_action)
            reward = result.reward

            # Bonus: if host is isolated and Red tried to exploit it, extra reward
            if host_name in self._isolated_hosts:
                reward += 0.2

        self._t += 1
        done = self._t >= self.max_steps

        # Get current observation
        cyborg_vec = self._get_cyborg_obs_vector()
        current_obs = self._build_full_obs(cyborg_vec)

        # Apply observation delay
        self._obs_buffer.append(current_obs.copy())
        if self.obs_delay > 0 and len(self._obs_buffer) > self.obs_delay:
            delayed_obs = self._obs_buffer[-1 - self.obs_delay]
        else:
            delayed_obs = current_obs

        info['reward_native'] = float(result.reward)
        info['isolated_count'] = len(self._isolated_hosts)
        info['patched_count'] = len(self._patched_hosts)
        info['action_name'] = f"{action_type} {host_name}"

        return delayed_obs, float(reward), done, False, info


# ============================================================
#  Logging Callback
# ============================================================
class DQN3xCallback(BaseCallback):
    """Periodic evaluation callback for 3x training."""

    def __init__(self, eval_env, log_freq=20000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.log_freq = log_freq
        self.eval_results = []

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self):
        action_counts = defaultdict(int)
        type_counts = defaultdict(int)
        total_rewards = []
        total_isolated = 0
        total_patched = 0
        total_steps = 0

        for red_name in RED_AGENT_NAMES:
            self.eval_env.fixed_red_agent = red_name
            ep_rewards = []

            for _ in range(3):
                obs, _ = self.eval_env.reset()
                ep_reward = 0
                for step in range(100):
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, _, info = self.eval_env.step(action)
                    ep_reward += reward
                    action_counts[info['action_name']] += 1
                    type_counts[info['action_type']] += 1
                    total_steps += 1
                    if done:
                        break
                ep_rewards.append(ep_reward)
                total_isolated += info.get('isolated_count', 0)
                total_patched += info.get('patched_count', 0)

            mean_r = np.mean(ep_rewards)
            total_rewards.extend(ep_rewards)
            if self.verbose:
                print(f"  {red_name:10s}: {mean_r:.1f}")

            # -- TensorBoard: per-Red reward --
            self.logger.record(f"eval/{red_name}_mean_reward", mean_r)

        overall_mean = np.mean(total_rewards)
        overall_std = np.std(total_rewards)
        self.eval_results.append((self.n_calls, overall_mean))

        # -- TensorBoard: overall eval metrics --
        self.logger.record("eval/overall_mean_reward", overall_mean)
        self.logger.record("eval/overall_std_reward", overall_std)
        self.logger.record("eval/unique_actions_used", len(action_counts))
        self.logger.record("eval/avg_isolated_per_ep",
                           total_isolated / (len(RED_AGENT_NAMES) * 3))
        self.logger.record("eval/avg_patched_per_ep",
                           total_patched / (len(RED_AGENT_NAMES) * 3))

        # -- TensorBoard: action-type distribution --
        total_type = sum(type_counts.values()) or 1
        for t in ACTION_TYPES:
            pct = type_counts[t] / total_type * 100
            self.logger.record(f"eval/action_pct_{t}", pct)

        type_str = ", ".join(f"{t}:{type_counts[t]/total_type*100:.0f}%"
                            for t in ACTION_TYPES)

        print(f"\n[Step {self.n_calls}] Overall: {overall_mean:.1f} ± "
              f"{overall_std:.1f} | Unique: {len(action_counts)} | "
              f"Types: {type_str}")

        # Reset to random for training
        self.eval_env.fixed_red_agent = None


# ============================================================
#  Training
# ============================================================
def make_env(red_agent=None):
    def _init():
        return CybORG3xEnv(max_steps=100, red_agent=red_agent)
    return _init


def main():
    print("=" * 70)
    print("  3x Scaled DQN Training for CybORG Blue Agent")
    print("=" * 70)
    print(f"  Hosts:        {N_HOSTS}")
    print(f"  Subnets:      5 (User, Enterprise, Operational, DMZ, Research)")
    print(f"  Action types: {N_ACTION_TYPES} ({', '.join(ACTION_TYPES)})")
    print(f"  Action space: Discrete({N_ACTION_TYPES * N_HOSTS})")
    print(f"  Red agents:   {', '.join(RED_AGENT_NAMES)}")
    print(f"  Obs delay:    {OBS_DELAY_STEPS} step(s)")
    print("=" * 70)

    # Create training environment (random Red)
    print("\nCreating training environment...")
    env = DummyVecEnv([make_env(red_agent=None)])

    # Create eval environment
    eval_env = CybORG3xEnv(max_steps=100, red_agent=None)

    # DQN with larger network for bigger state/action space
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-4,
        buffer_size=200000,             # Larger buffer for 3x
        learning_starts=15000,
        batch_size=128,                 # Larger batches
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=2000,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(
            net_arch=[512, 256, 128]     # Wider network for bigger problem
        ),
        verbose=1,
        tensorboard_log=os.path.join(OUTPUT_DIR, "dqn_3x_logs"),
        device="auto",
    )

    print(f"\nModel: MlpPolicy [512, 256, 128]")
    print(f"Replay buffer: 200K, batch_size: 128")
    print(f"Exploration: eps 1.0 -> 0.05 over 30%")

    callback = DQN3xCallback(eval_env, log_freq=25000)

    # Train
    total_timesteps = 750_000
    print(f"\nStarting training for {total_timesteps} steps...")
    print("=" * 70)

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=10,
        progress_bar=True,
    )

    # Save
    model_path = os.path.join(OUTPUT_DIR, "dqn_3x_v1.zip")
    model.save(model_path)
    print(f"\n✅ Model saved to: {model_path}")
    tb_dir = os.path.join(OUTPUT_DIR, "dqn_3x_logs")
    print(f"📊 TensorBoard logs: {tb_dir}")
    print(f"   Launch: tensorboard --logdir {tb_dir}")

    # --------------------------------------------------------
    #  Final evaluation
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Final Evaluation (10 episodes per Red agent)")
    print("=" * 70)

    all_results = {}
    for red_name in RED_AGENT_NAMES:
        eval_env.fixed_red_agent = red_name
        episode_rewards = []
        action_counts = defaultdict(int)
        type_counts = defaultdict(int)

        for ep in range(10):
            obs, _ = eval_env.reset()
            ep_reward = 0
            ep_actions = []

            for step in range(100):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, _, info = eval_env.step(action)
                ep_reward += reward
                action_counts[info['action_name']] += 1
                type_counts[info['action_type']] += 1
                ep_actions.append(info['action_name'])
                if done:
                    break

            episode_rewards.append(ep_reward)
            unique = len(set(ep_actions))
            print(f"  [{red_name}] Ep {ep+1}: Reward={ep_reward:.1f}, "
                  f"Unique={unique}, "
                  f"Isolated={info['isolated_count']}, "
                  f"Patched={info['patched_count']}")

        mean_r = np.mean(episode_rewards)
        std_r = np.std(episode_rewards)
        all_results[red_name] = (mean_r, std_r)

        total_a = sum(type_counts.values()) or 1
        print(f"\n  [{red_name}] Mean: {mean_r:.1f} ± {std_r:.1f}")
        print(f"  Action distribution: " + ", ".join(
            f"{t}:{type_counts[t]/total_a*100:.0f}%" for t in ACTION_TYPES
        ))
        print(f"  Total unique actions: {len(action_counts)}")
        print()

    # Summary
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for red_name, (m, s) in all_results.items():
        print(f"  vs {red_name:10s}: {m:8.1f} ± {s:.1f}")

    overall = np.mean([m for m, s in all_results.values()])
    print(f"\n  Overall mean:  {overall:.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
