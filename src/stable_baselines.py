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
import gymnasium as gym
from gymnasium import spaces
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


class OvercookedSBGym(gym.Env):
    """
    Wrapper for the Overcooked environment to be compatible with Stable Baselines3.
    This version supports multi-agent (AgentPair) interactions.
    """

    def __init__(self, base_env):
        super(OvercookedSBGym, self).__init__()

        self.base_env = base_env
        self.featurize_fn = get_observation

        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()

        self.reset()

    def _get_observation_space(self):
        """
        Define the observation space for joint observations (both agents).
        """

        dummy_mdp = self.base_env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        obs_shape = self.featurize_fn(dummy_state)[0].shape
        high = np.ones(obs_shape, dtype=np.float32) * float("inf")
        low = np.zeros(obs_shape, dtype=np.float32)
        return spaces.Box(low, high, dtype=np.float32)

        # return spaces.Box(
        #     low=0,
        #     high=1,
        #     shape=(NUM_AGENTS * INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT),
        #     dtype=np.float32,
        # )

    def _get_action_space(self):
        """
        Define the action space for joint actions (both agents).
        """
        return spaces.Discrete(len(ACTION_SPACE_SIZE))

    def step(self, action):
        """
        Execute one time step within the environment.
        Returns joint observations, rewards, and done flags.
        """
        # Convert joint action to individual actions
        agent0_action, agent1_action = [
            Action.INDEX_TO_ACTION[a] for a in action
        ]

        if self.agent_idx == 0:
            joint_action = (agent0_action, agent1_action)
        else:
            joint_action = (agent1_action, agent0_action)

            

        # Step the environment
        next_state, reward, done, env_info = self.base_env.step(joint_action)
        obs_agent0, obs_agent1 = self.featurize_fn(next_state)

        if self.agent_idx == 0:
            both_agents_ob = (obs_agent0, obs_agent1)
        else:
            both_agents_ob = (obs_agent1, obs_agent0)

        env_info["policy_agent_idx"] = self.agent_idx

        if "episode" in env_info.keys():
            env_info["episode"]["policy_agent_idx"] = self.agent_idx

        obs = {
            "both_agent_obs": both_agents_ob,
            "overcooked_state": next_state,
            "other_agent_env_idx": 1 - self.agent_idx,
        }
        return obs, reward, done, env_info
    
    def reset(self):
        """
        When training on individual maps, we want to randomize which agent is assigned to which
        starting location, in order to make sure that the agents are trained to be able to
        complete the task starting at either of the hardcoded positions.

        NOTE: a nicer way to do this would be to just randomize starting positions, and not
        have to deal with randomizing indices.
        """
        self.base_env.reset()
        self.mdp = self.base_env.mdp
        self.agent_idx = np.random.choice([0, 1])
        ob_p0, ob_p1 = self.featurize_fn(self.base_env.state)

        if self.agent_idx == 0:
            both_agents_ob = (ob_p0, ob_p1)
        else:
            both_agents_ob = (ob_p1, ob_p0)
        return {
            "both_agent_obs": both_agents_ob,
            "overcooked_state": self.base_env.state,
            "other_agent_env_idx": 1 - self.agent_idx,
        }


class PPONetwork(nn.Module):
    """
    Neural network for PPO agent.
    """
    def __init__(self, input_channels=26, action_space_size=6):
        super(PPONetwork, self).__init__()
        
        self.model_trunk0 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=25, kernel_size=5, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="valid"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(150, 32), 
            # nn.Linear(525, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
        )

        self.model_trunk1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=25, kernel_size=5, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="valid"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(150, 32), 
            # nn.Linear(525, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
        )

        self.latent_dim_pi = 32
        self.latent_dim_vf = 32

    def forward(self, obs):
        """
        Forward pass of the network.
        """
        x0 = self.model_trunk0(obs)
        x1 = self.model_trunk1(obs)
        return x0, x1
    

    

class CustomSBNetwork(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )

        self.values_net0 = nn.Linear(32, 1)
        self.values_net1 = nn.Linear(32, 1)

        self.mlp_extractor0 = PPONetwork(input_channels=26, action_space_size=6)
        self.mlp_extractor1 = PPONetwork(input_channels=26, action_space_size=6)

        self.action_net0 = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor0.latent_dim_pi)
        self.action_net1 = self.action_dist.proba_distribution_net(latent_dim=self.mlp_extractor0.latent_dim_pi)

        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def forward(self, obs, deterministic=False):
        
        latent_0 = self.mlp_extractor0(obs)
        latent_1 = self.mlp_extractor1(obs)

        mean_actions0 = self.action_net(latent_0)
        distribution0 = self.action_dist.proba_distribution(action_logits=mean_actions0)
        actions0 = distribution0.get_actions(deterministic=deterministic)
        log_prob0 = distribution0.log_prob(actions0)
        actions0 = actions0.reshape((-1, *self.action_space.shape))  # type: ignore[misc]

        mean_actions1 = self.action_net(latent_1)
        distribution1 = self._get_action_dist_from_latent(latent_1)
        actions1 = distribution1.get_actions(deterministic=deterministic)
        log_prob1 = distribution1.log_prob(actions1)
        actions1 = actions1.reshape((-1, *self.action_space.shape))
    
        values0 = self.value_net0(latent_0)
        values1 = self.value_net1(latent_1)

        return (actions0, actions1), (values0, values1), (log_prob0, log_prob1)









# class CustomNetwork(BaseFeaturesExtractor):
#     """
#     Custom neural network for processing joint observations and outputting actions for both agents.
#     """

#     def __init__(self, observation_space, features_dim=128):
#         super(CustomNetwork, self).__init__(observation_space, features_dim)
#         self.network = nn.Sequential(
#             nn.Conv2d(in_channels=NUM_AGENTS * INPUT_CHANNELS, out_channels=32, kernel_size=5, padding="same"),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="valid"),
#             nn.LeakyReLU(),
#             nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding="same"),
#             nn.LeakyReLU(),
#             nn.Flatten(),
#             nn.Linear(64 * MAX_WIDTH * MAX_HEIGHT, features_dim),
#             nn.LeakyReLU(),
#         )

#     def forward(self, observations):
#         return self.network(observations)


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
        self.env = OvercookedSBGym(mdp)

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
    