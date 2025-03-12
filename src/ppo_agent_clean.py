# ppo_agent_clean.py

# Standard libraries
import time
import multiprocessing
import numpy as np
import argparse

# PyTorch libraries
import torch
import torch.nn as nn
from torch.distributions import Categorical

# Third-party libraries
from tqdm import tqdm, trange
import wandb

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
MAX_WIDTH = 5 #9
MAX_HEIGHT = 4 #5
NUM_AGENTS = 2
INPUT_CHANNELS = 26
ACTION_SPACE_SIZE = 6


class PPONetwork(nn.Module):
    """
    Neural network for PPO agent. We separate the actor and critic (no shared network) for more stable learning. 
    For more information, see: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/.
    We use a network defined in the paper: 
    """
    def __init__(self, device, input_channels=INPUT_CHANNELS, action_space_size=ACTION_SPACE_SIZE):
        super(PPONetwork, self).__init__()

        # Device: Cuda, CPU, or MPS 
        self.device = device

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=INPUT_CHANNELS, out_channels=25, kernel_size=5, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="valid"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(150, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
        )

        # Actor: responsible for selecting actions based on the current policy
        self.actor = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=25, kernel_size=5, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="valid"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(150, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, action_space_size)
        ).to(self.device)

        # Critic: estimates the value function of the current state
        self.critic = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=25, kernel_size=5, padding="same"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="valid"),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=25, out_channels=25, kernel_size=3, padding="same"),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(150, 32), 
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        ).to(self.device)

    def get_state_value(self, observation):
        """
        Returns the computed state value (critic) on observation. 
        """
        return self.critic(observation)

    def forward(self, observation):
        """
        Forward pass of the network given observations. 
        Returns action_logits and value.
        """
        action_logits = self.actor(observation)
        value = self.critic(observation)
        return action_logits, value


class PPOAgent(Agent):
    """
    PPO Agent implementation for the Overcooked AI environment. 
    Agents can choose from the following 6 actions: [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), 'interact'].
    """
    def __init__(self, env, idx, config=None, debug_name=None):
        super(PPOAgent, self).__init__()

        self.debug_name = debug_name
        self.agent_index = idx
        
        # Initialize configuration
        self.config = config if config else Config()
        self.device = self.config.device if config else 'cpu'

        # Storing environment information
        self.env = env
        self.action_space = Action.ALL_ACTIONS 
        
        # Initialize neural network
        self.network = PPONetwork(
            device=self.device,
            input_channels=INPUT_CHANNELS, 
            action_space_size=ACTION_SPACE_SIZE
        ).to(self.device)

        # Initialize loss functions
        self.value_loss_fn = nn.MSELoss()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), 
            lr=self.config.learning_rate
        )

    def _get_distribution(self, observation):
        """
        Get action distribution from the current policy.
        """
        action_logits, _ = self.network.forward(observation)
        distribution = Categorical(logits=action_logits)
        return distribution
    
    def _get_value_estimate(self, observation):
        """
        Get value estimate from the current policy.
        """
        _, value = self.network.forward(observation)
        return value

    def action(self, state):
        """
        Get action from the current policy.
        """
        observation = torch.tensor(get_observation(self.env, state)[self.agent_index], dtype=torch.float32, device=self.network.device).unsqueeze(0) 
        assert observation.shape == (1, INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT)
        distribution = self._get_distribution(observation)
        action_idx = distribution.sample()
        og_log_probs = distribution.log_prob(action_idx)
        action = Action.INDEX_TO_ACTION[action_idx.item()]
        info = {'og_distribution': distribution.probs, 'og_log_probs': og_log_probs}
        return action, info  
    
    # def compute_GAE(self, rewards, values, dones):
    #     """
    #     Computes the generalized advantage estimator (GAE), a weighted sum of future TD errors. 
    #     GAE reduces the variance of policy gradient estimates to improve efficiency and stability of training. 
        
    #     Advantage Function: A(s_t, a_t) = Q(s_t, a_t) - V(s_t), measures how much better or worse action a is compared to the expected value of the state.

    #     Returns advantages, which contains the GAE for each timestep in the trajectory.
    #     """
    #     # Number of timesteps
    #     total_time_steps = len(dones)
    
    #     # Advantages, initialized to 0
    #     advantages = torch.zeros_like(rewards, device = self.device)

    #     # The next GAE value to propagate back 
    #     next_gae = torch.zeros_like(rewards[-1], device=self.device)
        
    #     for ts in reversed(range(total_time_steps)):
    #         if ts == total_time_steps - 1:
    #             # For the last timestep, the bootstrap value should be zero only if the episode terminates
    #             bootstrap_value = values[ts] * (1 - dones[ts])
    #             delta = rewards[ts] + self.config.gae_gamma * bootstrap_value - values[ts]
    #         else:
    #             # Calculate TD error for each timestep 
    #             delta = rewards[ts] + self.config.gae_gamma * values[ts + 1] * (1 - dones[ts]) - values[ts]
            
    #         # Update the advantage for the current timestep using the recursive formula
    #         advantages[ts] = delta + self.config.gae_gamma * self.config.gae_lambda * next_gae * (1 - dones[ts])

    #         # Update next_gae for the next iteration
    #         next_gae = advantages[ts]
        
    #     return advantages

    def compute_GAE(self, rewards, values, dones):
        last_values = values[:, -1]
        last_values = last_values.clone().reshape((30, 1))  # type: ignore[assignment]

        advantages = torch.zeros((30, 400), dtype=torch.float32)

        last_gae_lam = 0
        for step in reversed(range(400)):
            if step == 400 - 1:
                next_non_terminal = 1.0 - dones[:,-1].reshape((30,1))
                next_values = last_values
            else:
                next_non_terminal = 1.0
                next_values = values[:, step + 1]

            delta = rewards[:, step].reshape((30, 1)) + self.config.gae_gamma * next_values.reshape((30, 1)) * next_non_terminal - values[:, step].reshape((30, 1))
            last_gae_lam = delta.reshape((30, 1)) + self.config.gae_gamma * self.config.gae_lambda * next_non_terminal * last_gae_lam

            advantages[:, step] = last_gae_lam.flatten()
        return advantages


    
    def update_policy(self, observations, actions, advantages, returns, old_action_log_prob, entropy_coeff_current=None, debug=False):
        """
        Updates the policy using Proximal Point Optimization (PPO). 
        Uses the PPO-Clip objective. 
        """

        # Move to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        old_action_log_prob = old_action_log_prob.to(self.device)
        
        # Get the current policy's log probabilities and entropy 
        distribution = self._get_distribution(observations)
        curr_action_log_prob = distribution.log_prob(actions)
    
        # Compute the ratio of new and old log probabilities 
        log_prob_ratio = torch.exp(curr_action_log_prob - old_action_log_prob)

        values = self._get_value_estimate(observations)


        # DOING ADVANTAGE CALCULATION OUTSIDE
        # # Compute advantages (reshape value and returns to have dim self.config.horizon), (2000, 1) --> (5, 400)
        # returns = returns.view((-1, self.config.horizon))
        # dones = dones.reshape((-1, self.config.horizon))
        # advantages = self.compute_GAE(returns, values, dones).view(-1)

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO clip objective: This ensures we don't move too far from the old policy, negated for gradient ascent
        surrogate_loss = -1 * torch.min(
            log_prob_ratio * advantages, 
            torch.clamp(log_prob_ratio, 1 - self.config.clip_param, 1 + self.config.clip_param) * advantages
        ).mean()

        """
        # TODO: For now, ignore other losses (value, entropy)
        value_loss = self.value_loss_fn(curr_value.squeeze(), rewards)
        entropy_loss = -curr_entropy_value.mean()
        # Total loss = surrogate loss + value loss - entropy loss (the negative entropy encourages exploration)
        total_loss = surrogate_loss + self.config.vf_loss_coeff * value_loss + entropy_coeff_current * entropy_loss
        """
        value_loss = self.value_loss_fn(values.squeeze(), returns.squeeze())
        entropy_loss = -distribution.entropy().mean()
        total_loss = surrogate_loss + self.config.vf_loss_coeff * value_loss + entropy_coeff_current * entropy_loss
        # total_loss = surrogate_loss

        # SGD Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    
class PPOTrainer:
    """
    Handles the PPO training process for multi-agent Overcooked.
    """
    def __init__(self, config):
        # Initialize config and device 
        self.config = config
        self.device = self.config.device

        # Initialize WandB
        wandb.init(project="overcooked-ppo", config=self.config.__dict__)

        # Initialize environment
        mdp = OvercookedGridworld.from_layout_name(layout_name=self.config.layout)   
        self.env = OvercookedEnv.from_mdp(mdp, horizon=self.config.horizon)
        
        # Initialize agents
        self.agent0 = PPOAgent(self.env, 0, self.config)
        self.agent1 = PPOAgent(self.env, 1, self.config)
        self.agent_pair = AgentPair(self.agent0, self.agent1)

    def compute_linear_decay_coefficient(self, epoch_iter_step):
        """
        Compute the linearly decaying entropy coefficient
        """
        # Ensure current_step doesn't exceed horizon
        step = min(epoch_iter_step, self.config.entropy_coeff_horizon)
        
        # Linear interpolation, coefficient decreases linearly from start_value to end_value
        decay_fraction = step / self.config.entropy_coeff_horizon
        current_coeff = self.config.entropy_coeff_start - (self.config.entropy_coeff_start - self.config.entropy_coeff_end) * decay_fraction
        return current_coeff

    def collect_trajectories(self):
        """
        Collect trajectories from the environment. 
        Returns NN-compatible tensors for state, action, reward, infos, and dones. 
        """
        # Call `get_rollouts` function to obtain trajectories for number of episodes and extract information
        
        temp_agent0 = PPOAgent(self.env, self.config, debug_name="TEMP0")
        temp_agent1 = PPOAgent(self.env, self.config, debug_name="TEMP1")
        temp_agent0.agent_index = 0
        temp_agent1.agent_index = 1
        temp_agent0.network = temp_agent0.network.cpu()
        temp_agent1.network = temp_agent1.network.to("cpu")
        temp_agent0.device = "cpu" # custom variable
        temp_agent1.device = "cpu" # custom variable
        temp_agent0.network.device = "cpu" # custom variable
        temp_agent1.network.device = "cpu" # custom variable
        # copy weights into temp agent
        state_dict0 = {k: v.cpu() for k, v in self.agent0.network.state_dict().items()}
        state_dict1 = {k: v.cpu() for k, v in self.agent1.network.state_dict().items()}
        temp_agent0.network.load_state_dict(state_dict0)
        temp_agent1.network.load_state_dict(state_dict1)
        temp_agent_pair = AgentPair(temp_agent0, temp_agent1)
        trajectories = self.env.get_rollouts(temp_agent_pair, self.config.num_episodes, info=True, display_phi=True)
        
        # trajectories = self.env.get_rollouts(self.agent_pair, self.config.num_episodes, info=True, display_phi=True)
        states = trajectories["ep_states"] # Representation of the OvercookedGridWorld, with players and objects mapping
        actions = trajectories["ep_actions"] # Tuples, representing action (p0_action, p1_action)
        rewards = trajectories["ep_rewards"].astype(np.float32) # Total rewards obtain for each action
        dones = trajectories["ep_dones"] # Whether or not finished 
        infos = trajectories["ep_infos"] # Containing action probabilities per action and per agent
        assert states.shape == actions.shape == rewards.shape == dones.shape == infos.shape == (self.config.num_episodes, self.config.horizon)

        # Get encoding for each player 
        state_p0_tensor = torch.tensor(np.array([[get_observation(self.env, state)[0] for state in episode] for episode in states]), dtype=torch.float32)
        state_p1_tensor = torch.tensor(np.array([[get_observation(self.env, state)[1] for state in episode] for episode in states]), dtype=torch.float32)

        # Get action tensors 
        action_tensor = torch.tensor([[[Action.ACTION_TO_INDEX[single_action] for single_action in agent_action] for agent_action in episode] for episode in actions], dtype=torch.long) 

        # Get reward tensor 
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        # Check shapes 
        assert state_p0_tensor.shape == state_p1_tensor.shape == (self.config.num_episodes, self.config.horizon, INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT)
        assert reward_tensor.shape == (self.config.num_episodes, self.config.horizon)
        assert action_tensor.shape == (self.config.num_episodes, self.config.horizon, NUM_AGENTS)
        return state_p0_tensor, state_p1_tensor, action_tensor, reward_tensor, infos, dones
        
    def train(self, debug=False): 
        """
        Performs training. 
        """
        # Training loop, up to 420 iterations
        for iter in range(self.config.num_iters):
            # Debugging print statement 
            if debug and iter % 10 == 0: print("\n===========Training for Iteration {iter} ===========\n") 

            # Obtain a set of trajectories by running current policy in the environment 
            # self.agent0.network.to('cpu') # Move to CPU 
            # self.agent1.network.to('cpu')
            state_p0_tensor, state_p1_tensor, action_tensor, sparse_rewards_tensor, infos, dones = self.collect_trajectories()
            # self.agent0.network.to(self.device)
            # self.agent1.network.to(self.device)
            
            # Extract old log probabilities and rewards
            dense_rewards0, dense_rewards1, old_log_probs_p0, old_log_probs_p1 = [], [], [], []
            for game in range(self.config.num_episodes):
                # curr_reward = [info['phi_s_prime'] - info['phi_s'] for info in infos[game]]
                curr_reward0 = [info["shaped_r_by_agent"][0] for info in infos[game]]
                curr_reward1 = [info["shaped_r_by_agent"][1] for info in infos[game]]
                og_log_p0 = [info['agent_infos'][0]['og_log_probs'] for info in infos[game]]
                og_log_p1 = [info['agent_infos'][1]['og_log_probs'] for info in infos[game]]
                dense_rewards0.append(curr_reward0)
                dense_rewards1.append(curr_reward1)
                old_log_probs_p0.append(og_log_p0)
                old_log_probs_p1.append(og_log_p1)

            # Convert to tensors 
            dense_rewards0 = torch.tensor(dense_rewards0, dtype=torch.float32)
            dense_rewards1 = torch.tensor(dense_rewards1, dtype=torch.float32)
            # dense_rewards = torch.tensor(dense_rewards, dtype=torch.float32)
                   
            # TODO: Ignore dense_rewards and shaped_rewards for now
            shaped_rewards_tensor0 = sparse_rewards_tensor + dense_rewards0 * self.config.reward_shaping_factor
            shaped_rewards_tensor1 = sparse_rewards_tensor + dense_rewards1 * self.config.reward_shaping_factor


            # print(shaped_rewards_tensor.type)
            # shaped_rewards_tensor = sparse_rewards_tensor

            #### DUMMY CHECK #### - STAY IN PLACE REWARD
            # step_num = 0
            # episode_rewards = [] 
            # for episode in action_tensor:
            #     if step_num == 0:
            #         print(episode[:10])
            #         step_num += 1
            #     horizon_rewards = [] 
            #     for time_step in episode:
            #         curr_reward = 0 
            #         for action_pair in time_step:
            #             if action_pair.item() == 4: 
            #                 curr_reward += 1
            #         horizon_rewards.append(curr_reward)
            #     episode_rewards.append(np.array(horizon_rewards))

            # shaped_rewards_tensor = torch.tensor(np.array(episode_rewards), dtype=torch.float32)
            # #### DUMMY CHECK #### - STAY IN PLACE REWARD

            average_reward_per_episode = shaped_rewards_tensor0.sum(axis=1).mean()
            print("\n=========== Average Reward per Episode: ===========\n", average_reward_per_episode.item())
            wandb.log({"Average Reward": average_reward_per_episode, "Iteration": iter})
            
            # Move to device 
            state_p0_tensor = state_p0_tensor.to(self.device)
            state_p1_tensor = state_p1_tensor.to(self.device)

            # Compute value estimates (already on GPU)
            values_p0 = self.agent0._get_value_estimate(state_p0_tensor.view(-1, INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT))
            values_p1 = self.agent1._get_value_estimate(state_p1_tensor.view(-1, INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT))

            # Convert dones to tensor and move to device
            dones = torch.tensor(dones.astype(np.int64)).to(self.device)
            # move rewards and values to device
            shaped_rewards_tensor0 = shaped_rewards_tensor0.to(self.device)
            shaped_rewards_tensor1 = shaped_rewards_tensor1.to(self.device)
            # shaped_rewards_tensor = shaped_rewards_tensor.to(self.device)

            # shape values into epsiode form
            values_p0 = values_p0.view(self.config.num_episodes, self.config.horizon)
            values_p1 = values_p1.view(self.config.num_episodes, self.config.horizon)

            # Compute advantages 
            advantages_p0 = self.agent0.compute_GAE(shaped_rewards_tensor0, values_p0, dones) # TODO for cleanup, can move the "get_value_estimate" to the compute_GAE function
            advantages_p1 = self.agent1.compute_GAE(shaped_rewards_tensor1, values_p1, dones)

            returns_p0 = values_p0 + advantages_p0.view((-1, 1))
            returns_p1 = values_p1 + advantages_p1.view((-1, 1))

            # Flatten for batch processing - State 
            state_p0_batch = state_p0_tensor.view(-1, *state_p0_tensor.shape[-3:])
            state_p1_batch = state_p1_tensor.view(-1, *state_p1_tensor.shape[-3:])
            assert state_p0_batch.shape == state_p1_batch.shape == (self.config.horizon * self.config.num_episodes, INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT)
            
            # Flatten for batch processing - Action 
            action_tensor_joint = action_tensor.view(-1, action_tensor.shape[-1])       
            action_p0_batch = action_tensor_joint[:, 0] 
            action_p1_batch = action_tensor_joint[:, 1] 

            # Flatten for batch processing 
            returns_batch0 = returns_p0.view((-1, 1))
            returns_batch1 = returns_p1.view((-1, 1))
            assert returns_batch0.shape == returns_batch1.shape == (self.config.horizon * self.config.num_episodes, 1)

            # Flatten for batch processing - Dones 
            dones_batch = dones.view(-1, 1)
            assert dones_batch.shape == (self.config.horizon * self.config.num_episodes, 1)

            # Flatten for batch processing - Advantages 
            advantages_p0_batch = advantages_p0.view((-1, 1))
            advantages_p1_batch = advantages_p1.view((-1, 1))
            assert advantages_p0_batch.shape == advantages_p1_batch.shape == (self.config.horizon * self.config.num_episodes, 1)

            # Reshape old log probabilities 
            old_log_probs_p0_batch = torch.tensor(old_log_probs_p0).view(-1, 1)
            old_log_probs_p1_batch = torch.tensor(old_log_probs_p1).view(-1, 1)
            assert old_log_probs_p0_batch.shape == old_log_probs_p1_batch.shape == (self.config.horizon * self.config.num_episodes, 1)

            # detach and remove from gradient tracking
            action_p0_batch = action_p0_batch.detach().requires_grad_(False)
            action_p1_batch = action_p1_batch.detach().requires_grad_(False)
            returns_p0_batch = returns_batch0.detach().requires_grad_(False)
            returns_p1_batch = returns_batch1.detach().requires_grad_(False)
            advantages_p0_batch = advantages_p0_batch.detach().requires_grad_(False)
            advantages_p1_batch = advantages_p1_batch.detach().requires_grad_(False)
            old_log_probs_p0_batch = old_log_probs_p0_batch.detach().requires_grad_(False)
            old_log_probs_p1_batch = old_log_probs_p1_batch.detach().requires_grad_(False)
            dones_batch = dones_batch.detach().requires_grad_(False)
            
            # Move to device 
            action_p0_batch = action_p0_batch.to(self.device)
            action_p1_batch = action_p1_batch.to(self.device)
            returns_p0_batch = returns_p0_batch.to(self.device)
            returns_p1_batch = returns_p1_batch.to(self.device)
            advantages_p0_batch = advantages_p0_batch.to(self.device)
            advantages_p1_batch = advantages_p1_batch.to(self.device)
            old_log_probs_p0_batch = old_log_probs_p0_batch.to(self.device)
            old_log_probs_p1_batch = old_log_probs_p1_batch.to(self.device)
            dones_batch = dones_batch.to(self.device)

            # Run minibatch updates 
            total_batches = self.config.num_epochs * self.config.num_mini_batches # (8 epochs * 6 batches = 48)
            with tqdm(total=total_batches, desc="Training", leave=True) as progress_bar:
                for epoch in range(self.config.num_epochs): # 8 epochs
                    # Calculate current iteration of training
                    curr_epoch_iter = (self.config.num_epochs * iter) + epoch
                    
                    # TODO: Ignore this for now, Compute decaying entropy coefficient
                    entropy_coeff_current = self.compute_linear_decay_coefficient(curr_epoch_iter)

                    # Shuffle the data for random sampling
                    shuffled_indices = torch.randperm(self.config.horizon * self.config.num_episodes)
                    curr_state_tensor_p0 = state_p0_batch[shuffled_indices]
                    curr_state_tensor_p1 = state_p1_batch[shuffled_indices]
                    curr_action_tensor_p0 = action_p0_batch[shuffled_indices]
                    curr_action_tensor_p1 = action_p1_batch[shuffled_indices]
                    curr_returns_p0 = returns_p0_batch[shuffled_indices]
                    curr_returns_p1 = returns_p1_batch[shuffled_indices]
                    curr_advantages_p0 = advantages_p0_batch[shuffled_indices]
                    curr_advantages_p1 = advantages_p1_batch[shuffled_indices]
                    curr_old_log_probs_p0 = old_log_probs_p0_batch[shuffled_indices]
                    curr_old_log_probs_p1 = old_log_probs_p1_batch[shuffled_indices]
                    curr_dones = dones_batch[shuffled_indices]

                    for k in range(self.config.num_mini_batches):
                        # Compute minibatches of data, 2000 points
                        start = ((self.config.horizon * self.config.num_episodes) // self.config.num_mini_batches) * k
                        end = ((self.config.horizon * self.config.num_episodes) // self.config.num_mini_batches) * (k + 1)
                        state_tensor_p0_minibatch = curr_state_tensor_p0[start:end]
                        state_tensor_p1_minibatch = curr_state_tensor_p1[start:end]
                        action_tensor_p0_minibatch = curr_action_tensor_p0[start:end]
                        action_tensor_p1_minibatch = curr_action_tensor_p1[start:end]
                        returns_p0_minibatch = curr_returns_p0[start:end]
                        returns_p1_minibatch = curr_returns_p1[start:end]
                        advantages_p0_minibatch = curr_advantages_p0[start:end]
                        advantages_p1_minibatch = curr_advantages_p1[start:end]
                        old_log_probs_p0_minibatch = curr_old_log_probs_p0[start:end]
                        old_log_probs_p1_minibatch = curr_old_log_probs_p1[start:end]
                        dones_minibatch = curr_dones[start:end]
                        
                        # Update policy 
                        self.agent0.update_policy(state_tensor_p0_minibatch, action_tensor_p0_minibatch, advantages_p0_minibatch, returns_p0_minibatch, old_log_probs_p0_minibatch, entropy_coeff_current=entropy_coeff_current, debug=True)
                        self.agent1.update_policy(state_tensor_p1_minibatch, action_tensor_p1_minibatch, advantages_p1_minibatch, returns_p1_minibatch, old_log_probs_p1_minibatch, entropy_coeff_current=entropy_coeff_current, debug=False)
                        
                        # Update progress bar
                        progress_bar.update(1)

        wandb.finish()
        print("Reward:", rewards)
        return rewards
                    

if __name__ == "__main__":  
    layout_mapping = {
            0: 'cramped_room',
            1: 'padded_cramped_room', 
            2: 'padded_asymmetric_advantages_tomato',
            3: 'padded_coordination_ring', 
            4: 'padded_forced_coordination', 
            5: 'padded_counter_circuit'
    }

    parser = argparse.ArgumentParser(description="Train a PPO agent in the Overcooked environment.")
    parser.add_argument(
        "--layout", 
        type=int, 
        default=0, 
        choices=["padded_cramped_room", "padded_asymmetric_advantages_tomato", "padded_coordination_ring", "padded_forced_coordination", "padded_counter_circuit"],
        help="The layout to use for training."
    )

    args = parser.parse_args()
    config = Config(layout=layout_mapping[args.layout])

    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    trainer = PPOTrainer(config)
    rewards = trainer.train()
