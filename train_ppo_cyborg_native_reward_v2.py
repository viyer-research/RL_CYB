import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor


# Scenarios for curriculum learning (easier -> harder)
SCENARIO_EASY = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b.yaml"  # SleepAgent (no attacks)
SCENARIO_HARD = r"D:\Vasanth\RL_CYB\cage-challenge-1\CybORG\CybORG\Shared\Scenarios\Scenario1b-vulnerable.yaml"  # B_lineAgent (active attacks)


class CybORGNativeRewardEnv(gym.Env):
    """
    PPO environment for CybORG Scenario1b:
    - Uses CybORG's NATIVE reward (captures game state properly)
    - Small action bonuses to encourage active defense
    - Diversity penalty to prevent single-action collapse
    """

    metadata = {"render_modes": []}

    def __init__(self, scenario_path, agent="Blue", max_steps=200):
        super().__init__()

        from CybORG import CybORG
        from CybORG.Agents.Wrappers.FixedFlatWrapper import FixedFlatWrapper
        from CybORG.Agents.Wrappers.EnumActionWrapper import EnumActionWrapper
        from CybORG.Agents.Wrappers.BlueTableWrapper import BlueTableWrapper

        self.agent = agent
        self.max_steps = max_steps
        self.scenario_path = scenario_path

        self._cyborg = CybORG(scenario_path, "sim")
        
        # Chain wrappers properly: BlueTableWrapper -> EnumActionWrapper for proper action space
        self._table = BlueTableWrapper(env=self._cyborg, output_mode='vector')
        self._enum = EnumActionWrapper(env=self._table)
        self._flat = FixedFlatWrapper(env=self._cyborg, agent=self.agent)

        # Get the properly enumerated actions from EnumActionWrapper
        action_space_size = self._enum.action_space_change(
            self._cyborg.get_action_space(agent=self.agent)
        )
        self.possible_actions = self._enum.possible_actions
        
        # Build readable action names
        self.action_names = [str(act) for act in self.possible_actions]
        print(f"[INFO] Blue agent has {len(self.possible_actions)} discrete actions")

        self.action_space = spaces.Discrete(len(self.possible_actions))

        # determine observation shape
        _ = self._cyborg.reset()
        flat_obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

        self.observation_space = spaces.Box(
            low=-1e9, high=1e9, shape=flat_obs.shape, dtype=np.float32
        )

        self._t = 0
        self._last_action = ""
        self._action_counts = {}  # Track action usage for diversity

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        _ = self._cyborg.reset()
        self._t = 0
        self._last_action = ""
        self._action_counts = {}

        obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()
        info = {}
        return obs, info
    
    def step(self, action_idx):
        self._t += 1

        # Use the properly enumerated action object
        action_obj = self.possible_actions[int(action_idx)]
        current_action_name = self.action_names[int(action_idx)]

        # take CybORG step
        results = self._cyborg.step(agent=self.agent, action=action_obj)

        # ===========================================
        # USE CYBORG'S NATIVE REWARD AS BASE
        # This captures: host availability, confidentiality, Red progress
        # ===========================================
        reward = float(results.reward)
        
        # Small action-based bonuses to encourage active defense
        action_lower = current_action_name.lower()
        if "monitor" in action_lower:
            reward += 0.05
        elif "analyse" in action_lower:
            reward += 0.1
        elif "remove" in action_lower:
            reward += 0.15
        elif "restore" in action_lower:
            reward += 0.1
        elif "misinform" in action_lower:
            reward += 0.05
        # Sleep gets no bonus
        
        # Diversity penalty - discourage repeating same action
        if current_action_name == self._last_action:
            reward -= 0.2
        
        # Track action counts for logging
        self._action_counts[current_action_name] = self._action_counts.get(current_action_name, 0) + 1
        self._last_action = current_action_name

        terminated = bool(results.done)
        truncated = self._t >= self.max_steps

        obs = np.array(self._flat.get_observation(self.agent), dtype=np.float32).flatten()

        info = {
            "action_name": current_action_name,
            "reward_native": float(results.reward),
            "reward_shaped": float(reward),
        }

        return obs, float(reward), terminated, truncated, info


def make_env(scenario_path):
    def _init():
        env = CybORGNativeRewardEnv(scenario_path=scenario_path)
        env = Monitor(env)
        return env
    return _init


def main():
    from stable_baselines3.common.vec_env import SubprocVecEnv
    
    out_dir = r"D:\Vasanth\RL_CYB\ppo_out"
    os.makedirs(out_dir, exist_ok=True)
    
    # ===========================================
    # CURRICULUM LEARNING: Train in phases
    # ===========================================
    
    # ----- PHASE 1: Easy scenario (no attacks) - 500K steps -----
    print("\n" + "="*60)
    print("PHASE 1: Training on EASY scenario (SleepAgent - no attacks)")
    print("="*60)
    
    venv_easy = SubprocVecEnv([make_env(SCENARIO_EASY) for _ in range(4)])
    
    model = PPO(
        "MlpPolicy",
        venv_easy,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.1,  # HIGH entropy for exploration
        verbose=1,
        tensorboard_log=r"D:\Vasanth\RL_CYB\cage-challenge-1\ppo_cyborg_tensorboard"
    )
    
    model.learn(total_timesteps=500_000, tb_log_name="PPO_curriculum_phase1")
    
    # Save phase 1 checkpoint
    model.save(os.path.join(out_dir, "ppo_native_reward_phase1.zip"))
    print("\n✅ Phase 1 complete - saved checkpoint")
    
    venv_easy.close()
    
    # ----- PHASE 2: Hard scenario (active attacks) - 1M steps -----
    print("\n" + "="*60)
    print("PHASE 2: Training on HARD scenario (B_lineAgent - active attacks)")
    print("="*60)
    
    venv_hard = SubprocVecEnv([make_env(SCENARIO_HARD) for _ in range(4)])
    
    # Continue training with reduced entropy (exploit learned behavior)
    model.set_env(venv_hard)
    model.ent_coef = 0.05  # Reduce entropy for phase 2
    
    model.learn(total_timesteps=1_000_000, tb_log_name="PPO_curriculum_phase2", reset_num_timesteps=False)
    
    # Save phase 2 checkpoint
    model.save(os.path.join(out_dir, "ppo_native_reward_phase2.zip"))
    print("\n✅ Phase 2 complete - saved checkpoint")
    
    # ----- PHASE 3: Fine-tuning with lower entropy - 500K steps -----
    print("\n" + "="*60)
    print("PHASE 3: Fine-tuning with low entropy (exploitation)")
    print("="*60)
    
    model.ent_coef = 0.02  # Low entropy for fine-tuning
    
    model.learn(total_timesteps=500_000, tb_log_name="PPO_curriculum_phase3", reset_num_timesteps=False)
    
    # Save final model
    final_path = os.path.join(out_dir, "ppo_native_reward_curriculum_final.zip")
    model.save(final_path)
    
    venv_hard.close()
    
    print("\n" + "="*60)
    print("✅ CURRICULUM TRAINING COMPLETE")
    print(f"Final model saved: {final_path}")
    print("="*60)


if __name__ == "__main__":
    main()
