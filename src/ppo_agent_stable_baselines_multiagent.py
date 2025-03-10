# ppo_agent_stable_baselines_multiagent.py

# Standard libraries
import time
import multiprocessing
import numpy as np
import argparse

# PyTorch libraries
import torch
import torch.nn as nn

# Third-party libraries
from tqdm import tqdm, trange
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Local imports
from config import Config

# Overcooked AI imports
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair

# Helper function imports
from ppo_helper import get_observation

# Constants
MAX_WIDTH = 5  # 9
MAX_HEIGHT = 4  # 5
NUM_AGENTS = 2
INPUT_CHANNELS = 26
ACTION_SPACE_SIZE = 6


class OvercookedStableBaselinesEnv(OvercookedEnv):
    """
    Wrapper for the Overcooked environment to be compatible with Stable Baselines3.
    This version supports multi-agent (AgentPair) interactions.
    """

    def __init__(self, mdp, **kwargs):
        super(OvercookedStableBaselinesEnv, self).__init__(mdp, **kwargs)
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

    def _get_observation_space(self):
        """
        Define the observation space for joint observations (both agents).
        """
        return spaces.Box(
            low=0,
            high=1,
            shape=(NUM_AGENTS * INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT),
            dtype=np.float32,
        )

    def _get_action_space(self):
        """
        Define the action space for joint actions (both agents).
        """
        return spaces.MultiDiscrete([ACTION_SPACE_SIZE] * NUM_AGENTS)

    def reset(self):
        """
        Reset the environment and return the initial joint observation.
        """
        state = super(OvercookedStableBaselinesEnv, self).reset()
        obs = get_observation(self, state)
        joint_obs = np.concatenate([obs[0], obs[1]], axis=0)  # Concatenate observations
        return joint_obs

    def step(self, action):
        """
        Execute one time step within the environment.
        Returns joint observations, rewards, and done flags.
        """
        # Convert joint action to individual actions
        action_p0 = Action.INDEX_TO_ACTION[action[0]]
        action_p1 = Action.INDEX_TO_ACTION[action[1]]
        joint_action = (action_p0, action_p1)

        # Step the environment
        state, reward, done, info = super(OvercookedStableBaselinesEnv, self).step(joint_action)

        # Get joint observations
        obs = get_observation(self, state)
        joint_obs = np.concatenate([obs[0], obs[1]], axis=0)

        # Return joint observations, rewards, and done flags
        return joint_obs, reward, done, info


class CustomNetwork(BaseFeaturesExtractor):
    """
    Custom neural network for processing joint observations and outputting actions for both agents.
    """

    def __init__(self, observation_space, features_dim=128):
        super(CustomNetwork, self).__init__(observation_space, features_dim)
        self.network = nn.Sequential(
            nn.Conv2d(in_channels=NUM_AGENTS * INPUT_CHANNELS, out_channels=32, kernel_size=5, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="valid"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(64 * MAX_WIDTH * MAX_HEIGHT, features_dim),
            nn.LeakyReLU(),
        )

    def forward(self, observations):
        return self.network(observations)


class PPOTrainer:
    """
    Handles the PPO training process for multi-agent Overcooked using Stable Baselines3.
    """

    def __init__(self, config):
        # Initialize config and device
        self.config = config
        self.device = self.config.device

        # Initialize WandB
        # wandb.init(project="overcooked-ppo", config=self.config.__dict__)

        # Initialize environment
        mdp = OvercookedGridworld.from_layout_name(layout_name=self.config.layout)
        self.env = OvercookedStableBaselinesEnv(mdp, horizon=self.config.horizon)

        # Create a vectorized environment
        self.vec_env = make_vec_env(lambda: self.env, n_envs=multiprocessing.cpu_count(), vec_env_cls=SubprocVecEnv)

        # Initialize PPO agent with a custom network
        policy_kwargs = dict(
            features_extractor_class=CustomNetwork,
            features_extractor_kwargs=dict(features_dim=128),
        )
        self.agent = PPO(
            "MultiInputPolicy",
            self.vec_env,
            verbose=1,
            device=self.device,
            policy_kwargs=policy_kwargs,
            learning_rate=self.config.learning_rate,
            n_steps=self.config.horizon,
            batch_size=self.config.horizon * self.config.num_episodes,
            n_epochs=self.config.num_epochs,
            gamma=self.config.gae_gamma,
            gae_lambda=self.config.gae_lambda,
            clip_range=self.config.clip_param,
            ent_coef=self.config.entropy_coeff_start,
            max_grad_norm=0.5,
        )

    def train(self, debug=False):
        """
        Performs training.
        """
        # Training loop, up to 420 iterations
        for iter in range(self.config.num_iters):
            # Debugging print statement
            if debug and iter % 10 == 0:
                print(f"\n=========== Training for Iteration {iter} ===========\n")

            # Train the agent
            self.agent.learn(total_timesteps=self.config.horizon * self.config.num_episodes, reset_num_timesteps=False)

            # Evaluate the agent
            average_reward_per_episode = self.evaluate()
            print(f"\n=========== Average Reward per Episode: {average_reward_per_episode} ===========\n")
            # wandb.log({"Average Reward": average_reward_per_episode, "Iteration": iter})

        # wandb.finish()
        print("Training complete.")

    def evaluate(self):
        """
        Evaluate the agent's performance.
        """
        total_rewards = []
        for _ in range(self.config.num_episodes):
            obs = self.env.reset()
            done = False
            total_reward = 0
            while not done:
                action, _ = self.agent.predict(obs)
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
            total_rewards.append(total_reward)
        return np.mean(total_rewards)


if __name__ == "__main__":
    layout_mapping = {
        0: "cramped_room",
        1: "padded_cramped_room",
        2: "padded_asymmetric_advantages_tomato",
        3: "padded_coordination_ring",
        4: "padded_forced_coordination",
        5: "padded_counter_circuit",
    }

    parser = argparse.ArgumentParser(description="Train a PPO agent in the Overcooked environment.")
    parser.add_argument(
        "--layout",
        type=int,
        default=0,
        choices=[0, 1, 2, 3, 4, 5],
        help="The layout to use for training.",
    )

    args = parser.parse_args()
    config = Config(layout=layout_mapping[args.layout])

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainer = PPOTrainer(config)
    trainer.train()
    