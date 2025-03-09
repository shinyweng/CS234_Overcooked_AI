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
CHANNELS = 26
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
    def __init__(self, env, config=None):
        super(PPOAgent, self).__init__()
        
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
        observation = torch.tensor(get_observation(self.env, state)[self.agent_index], dtype=torch.float32).unsqueeze(0) 
        assert observation.shape == (1, INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT)
        distribution = self._get_distribution(observation)
        action_idx = distribution.sample()
        og_log_probs = distribution.log_prob(action_idx)
        action = Action.INDEX_TO_ACTION[action_idx.item()]
        info = {'og_distribution': distribution.probs, 'og_log_probs': og_log_probs}
        return action, info  
    
    def compute_GAE(self, returns, values, dones):
        """
        Computes the generalized advantage estimator (GAE), a weighted sum of future TD errors. 
        GAE reduces the variance of policy gradient estimates to improve efficiency and stability of training. 
        
        Advantage Function: A(s_t, a_t) = Q(s_t, a_t) - V(s_t), measures how much better or worse action a is compared to the expected value of the state.

        Returns advantages, which contains the GAE for each timestep in the trajectory.
        """
        # Number of timesteps
        total_time_steps = len(dones)
    
        # Advantages, initialized to 0
        advantages = torch.zeros_like(returns, device = self.device)

        # The next GAE value to propagate back 
        next_gae = torch.zeros_like(returns[-1], device=self.device)
        
        for ts in reversed(range(total_time_steps)):
            if ts == total_time_steps - 1:
                # For the last timestep, use the next_value, which is zero if done
                delta = returns[ts] + self.config.gae_gamma * 0 * (1 - dones[ts]) - values[ts]
            else:
                # Calculate TD error for each timestep 
                delta = returns[ts] + self.config.gae_gamma * values[ts + 1] * (1 - dones[ts]) - values[ts]
            
            # Update the advantage for the current timestep using the recursive formula
            advantages[ts] = delta + self.config.gae_gamma * self.config.gae_lambda * next_gae * (1 - dones[ts])

            # Update next_gae for the next iteration
            next_gae = advantages[ts]
        
        return advantages
    
    def update_policy(self, observations, actions, returns, old_action_log_prob, dones, rewards=None, entropy_coeff_current=None, debug=False):
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

        # Compute advantages (reshape value and returns to have dim self.config.horizon), (2000, 1) --> (5, 400)
        values = self._get_value_estimate(observations).view((-1, self.config.horizon))
        returns = returns.view((-1, self.config.horizon))
        dones = dones.reshape((-1, self.config.horizon))
        advantages = self.compute_GAE(returns, values, dones).view(-1)

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
        self.agent0 = PPOAgent(self.env, self.config)
        self.agent1 = PPOAgent(self.env, self.config)
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

    def get_returns(self, rewards, discount_factor_gamma=1):
        """
        Calculate the returns G_t for each timestep: G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ... + γ^{T-t} r_T.
        Note: Due to finite horizon, we let discount_factor_gamma = 1.
        """
        all_returns = []
        for episode_rewards in rewards:
            returns = np.zeros_like(episode_rewards, dtype=np.float32)
            cumulative_return = 0.0
            
            # Traverse the rewards in reverse order to accumulate the returns
            for t in reversed(range(len(episode_rewards))):
                cumulative_return = episode_rewards[t] + discount_factor_gamma * cumulative_return
                returns[t] = cumulative_return
            all_returns.append(returns)
        
        return torch.Tensor(np.stack(all_returns))

    def collect_trajectories(self):
        """
        Collect trajectories from the environment. 
        Returns NN-compatible tensors for state, action, reward, infos, and dones. 
        """
        # Call `get_rollouts` function to obtain trajectories for number of episodes and extract information
        trajectories = self.env.get_rollouts(self.agent_pair, self.config.num_episodes, info=True, display_phi=True)
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
            self.agent0.network.to('cpu') # Move to CPU 
            self.agent1.network.to('cpu')
            state_p0_tensor, state_p1_tensor, action_tensor, sparse_rewards_tensor, infos, dones = self.collect_trajectories()
            self.agent0.network.to(self.device)
            self.agent1.network.to(self.device)
            
            # Extract old log probabilities and rewards
            dense_rewards, old_log_probs_p0, old_log_probs_p1 = [], [], []
            for game in range(self.config.num_episodes):
                curr_reward = [info['phi_s_prime'] - info['phi_s'] for info in infos[game]]
                og_log_p0 = [info['agent_infos'][0]['og_log_probs'] for info in infos[game]]
                og_log_p1 = [info['agent_infos'][1]['og_log_probs'] for info in infos[game]]
                dense_rewards.append(curr_reward)
                old_log_probs_p0.append(og_log_p0)
                old_log_probs_p1.append(og_log_p1)

            # Convert to tensors 
            dense_rewards = torch.tensor(dense_rewards, dtype=torch.float32)
                   
            # TODO: Ignore dense_rewards and shaped_rewards for now
            shaped_rewards_tensor = sparse_rewards_tensor + dense_rewards * self.config.reward_shaping_factor
            # shaped_rewards_tensor = sparse_rewards_tensor

            average_reward_per_episode = shaped_rewards_tensor.sum(axis=1).mean()
            print("\n=========== Average Reward per Episode: ===========\n", average_reward_per_episode)
            wandb.log({"Average Reward": average_reward_per_episode, "Iteration": iter})

            # Flatten for batch processing - State 
            state_p0_batch = state_p0_tensor.view(-1, *state_p0_tensor.shape[-3:])
            state_p1_batch = state_p1_tensor.view(-1, *state_p1_tensor.shape[-3:])
            assert state_p0_batch.shape == state_p1_batch.shape == (self.config.horizon * self.config.num_episodes, INPUT_CHANNELS, MAX_WIDTH, MAX_HEIGHT)
            
            # Flatten for batch processing - Action 
            action_tensor_joint = action_tensor.view(-1, action_tensor.shape[-1])       
            action_p0_batch = action_tensor_joint[:, 0] 
            action_p1_batch = action_tensor_joint[:, 1] 

            # Flatten for batch processing - Dones 
            dones_batch = torch.Tensor(dones.astype(np.bool_)).view(-1, 1)
            assert dones_batch.shape == (self.config.horizon * self.config.num_episodes, 1)

            # Compute and flatten for batch processing - Returns 
            returns = self.get_returns(sparse_rewards_tensor)
            returns_batch = returns.view((-1, 1)) 
            assert returns_batch.shape == (self.config.horizon * self.config.num_episodes, 1)

            # Reshape old log probabilities 
            old_log_probs_p0_batch = torch.Tensor(old_log_probs_p0).view(-1, 1)
            old_log_probs_p1_batch = torch.Tensor(old_log_probs_p1).view(-1, 1)
            assert old_log_probs_p0_batch.shape == old_log_probs_p1_batch.shape == (self.config.horizon * self.config.num_episodes, 1)

            # Move to device 
            state_p0_batch = state_p0_batch.to(self.device)
            state_p1_batch = state_p1_batch.to(self.device)
            action_p0_batch = action_p0_batch.to(self.device)
            action_p1_batch = action_p1_batch.to(self.device)
            returns_batch = returns_batch.to(self.device)
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
                    curr_returns = returns_batch[shuffled_indices]
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
                        returns_minibatch = curr_returns[start:end]
                        old_log_probs_p0_minibatch = curr_old_log_probs_p0[start:end]
                        old_log_probs_p1_minibatch = curr_old_log_probs_p1[start:end]
                        dones_minibatch = curr_dones[start:end]
                        
                        # Update policy 
                        self.agent0.update_policy(state_tensor_p0_minibatch, action_tensor_p0_minibatch, returns_minibatch, old_log_probs_p0_minibatch, dones_minibatch, entropy_coeff_current=entropy_coeff_current, debug=True)
                        self.agent1.update_policy(state_tensor_p1_minibatch, action_tensor_p1_minibatch, returns_minibatch, old_log_probs_p1_minibatch, dones_minibatch, entropy_coeff_current=entropy_coeff_current, debug=True)
                        
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
