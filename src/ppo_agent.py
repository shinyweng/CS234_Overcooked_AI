# ppo_agent.py
import os 
import argparse
import torch
import torch.nn as nn
import numpy as np
from config import Config
from tqdm import tqdm, trange
# import wandb


from overcooked_ai_py.mdp.actions import Action
from torch.distributions import Categorical
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair

# padding constants
MAX_WIDTH = 9
MAX_HEIGHT = 5

# very jank ass solution from claude 
from overcooked_ai_py.mdp.overcooked_mdp import Recipe

Recipe.configure({})  # Or with appropriate configuration


def get_obs(env, state):
    """
    Get observation from the environment.
    """
    encoding = np.array(env.lossless_state_encoding_mdp(state))
    obs0, obs1 = encoding[0], encoding[1]
    obs0 = np.transpose(obs0, (2, 0, 1)) # permute from (w, h, c) to (c, w, h)
    obs1 = np.transpose(obs1, (2, 0, 1)) # permute from (w, h, c) to (c, w, h)
    return obs0, obs1 
    # return pad_obs(obs0), pad_obs(obs1)

# def pad_obs(obs, target_width=MAX_WIDTH, target_height=MAX_HEIGHT): 
#     """"Pad obs from (c,w,h) to (c, max_w, max_h)"""
#     c, w, h = obs.shape
#     width_diff = (target_width - w ) // 2
#     height_diff = (target_height - h) // 2
#     padded = np.pad(obs + 1, (
#         (0, 0), # don't change c
#         (width_diff, target_width - w - width_diff), # middle pad 
#         (height_diff, target_height - h - height_diff) # middle pad 
#     ), mode='constant', constant_values=0)
#     return padded
    
class PPONetwork(nn.Module):
    """
    Neural network for PPO agent.
    """
    def __init__(self, device, input_channels=26, action_space_size=6):
        super(PPONetwork, self).__init__()
        self.device = device
        
        self.model_trunk = nn.Sequential(
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
        ).to(self.device)

        self.action_head = nn.Linear(32, action_space_size).to(self.device)
        self.value_head = nn.Linear(32, 1).to(self.device)

    def forward(self, obs):
        """
        Forward pass of the network.
        """
        x = self.model_trunk(obs)
        action_logits = self.action_head(x)
        value = self.value_head(x)
        return action_logits, value
    

class PPOAgent(Agent):
    """
    PPO Agent implementation for Overcooked AI environment.
    """
    def __init__(self, env, config=None):
        super(PPOAgent, self).__init__()
        
        # Initialize configuration
        self.config = config if config else Config()
        self.device = self.config.device if config else 'cpu'
        self.action_space = Action.ALL_ACTIONS

        # Storing the env that the agent is in
        self.env = env
        
        # Initialize neural network
        self.network = PPONetwork(
            device=self.device,
            input_channels=26, 
            action_space_size=len(self.action_space)
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
        # print("Observation", observation.shape)
        logits, _ = self.network.forward(observation)
        # print("Logits", logits.shape)
        distribution = Categorical(logits=logits)
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
        observation = torch.tensor(get_obs(self.env, state)[self.agent_index], dtype=torch.float32).unsqueeze(0).to(self.device)
        distribution = self._get_distribution(observation)
        action_idx = distribution.sample()
        og_log_probs = distribution.log_prob(action_idx)
        action = Action.INDEX_TO_ACTION[action_idx.item()]
        info = {'og_distribution': distribution.probs, 'og_log_probs': og_log_probs}
        return action, info 
    

    def compute_gae(self, values, rewards):
        """
        observations: 
        """
        values = torch.cat((values, torch.zeros(values.shape[0], 1, device=self.device)), dim=1)  # Append a zero to the end of values
        # print("value shape in compute_gae", values.shape)
        deltas = rewards + self.config.gae_gamma * values[:, 1:] - values[:, :-1] # next value minus current value 
        advantages = torch.zeros_like(deltas, device=self.device)
        gae = 0
        for t in reversed(range(deltas.shape[1])):
            gae = deltas[:, t] + self.config.gae_gamma * self.config.gae_lambda * gae
            advantages[:, t] = gae
        return advantages

    def update_policy(self, observations, actions, returns, old_logprobs, entropy_coeff_current, debug=False):
        """
        Update the policy using PPO.

        observations: torch.tensor of shape (num_episodes * num_steps, num_channels, w, h)
        actions: torch.tensor of shape (num_episodes * num_steps)
        advantages: torch.tensor of shape (num_episodes * num_steps)
        returns: torch.tensor of shape (num_episodes * num_steps)
        old_logprobs: torch.tensor of shape (num_episodes * num_steps)
        entropy_coeff_current: float
        """
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        
        distribution = self._get_distribution(observations)
        # print(observations.shape)
        value = self._get_value_estimate(observations)
        # print(value.shape)

        # Compute advantages
        # turn value and returns back to episode shape
        value_reshaped = value.view((-1, self.config.horizon))
        returns_reshaped = returns.view((-1, self.config.horizon))

        advantages = self.compute_gae(value_reshaped, returns_reshaped)
        advantages = advantages.view(-1)

        # policy loss
        logprobs = distribution.log_prob(actions)
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
        policy_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()



        # value loss
        assert value.shape == returns.shape # (2000, 1)
        # value_loss = -self.value_loss_fn(value, returns)
        value_loss = self.value_loss_fn(value, returns)
    
        # Compute entropy loss
        # Entropy encourages exploration by penalizing deterministic policies
        entropy_loss = distribution.entropy().mean()
        
        total_loss = -policy_loss + (self.config.vf_loss_coeff * value_loss) - (entropy_coeff_current * entropy_loss)
    
        if debug: print(f"value loss: {value_loss}, policy loss: {policy_loss}, entropy loss: {entropy_loss}")
        
        # Zero out previous gradients
        self.optimizer.zero_grad()
        
        # Compute and clip gradients
        total_loss.backward()
        if debug: print("Total Loss", total_loss.item())
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.config.max_grad_norm)
        
        # Update network parameters
        self.optimizer.step()


class PPOTrainer:
    """Handles the PPO training process."""
    def __init__(self, config):
        self.config = config
        
        
        # Initialize WandB
        # wandb.init(project="overcooked-ppo", config=self.config.__dict__)
        
        # Initialize environment
        mdp = OvercookedGridworld.from_layout_name(layout_name=self.config.layout)   
        self.env = OvercookedEnv.from_mdp(mdp, horizon=self.config.horizon)
        
        # Initialize agents
        self.agent0 = PPOAgent(self.env, self.config)
        self.agent1 = PPOAgent(self.env, self.config)
        self.agent_pair = AgentPair(self.agent0, self.agent1)
        self.device = self.config.device

    def collect_trajectories(self):
        """
        Collect trajectories from the environment.
        """
        trajectories = self.env.get_rollouts(self.agent_pair, self.config.num_episodes, info=True, display_phi=True)
        # print("Trajectories", trajectories)

        states = trajectories["ep_states"] # shape (num_episodes, num_steps, num_states)
        actions = trajectories["ep_actions"] # shape (num_episodes, num_steps, num_agents, num_actions)
        rewards = trajectories["ep_rewards"].astype(np.float32) # shape (num_episodes, num_steps)
        dones = trajectories["ep_dones"] # shape (num_episodes, num_steps)
        infos = trajectories["ep_infos"] # shape (num_episodes, num_steps, num_agents)

        # Featurize and convert to tensors
        # print("TEST")
        # print(states[0][0])

        state_p0 = np.array([[get_obs(self.env, state)[0] for state in episode] for episode in states])
        state_p1 = np.array([[get_obs(self.env, state)[1] for state in episode] for episode in states])
        state_tensor_p0 = torch.tensor(state_p0, dtype=torch.float32) # shape (num_episodes, num_steps, num_agents, num_channels, h/w, h/w)
        state_tensor_p1 = torch.tensor(state_p1, dtype=torch.float32)

        action_tensor = torch.tensor([[[Action.ACTION_TO_INDEX[act] for act in agent_acts] for agent_acts in episode] for episode in actions], dtype=torch.long) # shape (num_episodes, num_steps, num_agents)
        # print("rewards", rewards)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        # Move tensors to device
        state_tensor_p0 = state_tensor_p0.to(self.device)
        state_tensor_p1 = state_tensor_p1.to(self.device)
        action_tensor = action_tensor.to(self.device)
        reward_tensor = reward_tensor.to(self.device)

        return state_tensor_p0, state_tensor_p1, action_tensor, reward_tensor, infos
    
    def get_returns(self, shaped_reward):
        """"
        shaped_reward: torch.tensor of shape (num_episodes, num_steps)

        Returns: returns of shape (num_episodes, num_steps, with gamma=1)
        """

        reversed = shaped_reward.flip(1)
        returns = torch.cumsum(reversed, dim=1).flip(1)
        return returns

    def compute_advantages(self, returns):
        """
        Compute advantages for each trajectory in the batch.
        Args:
            trajectories (list): List of trajectories, each trajectory is a dictionary containing 'reward'.
            returns (np.ndarray): Array of discounted returns for each trajectory.
        """
        
        if self.config.normalize_advantage:        
            advantages = (returns - returns.mean()) / (returns.std() + 1e-8)
        return advantages
    
    def compute_linear_decay_coefficient(self, epoch_iter_step):
        """
        Compute the linearly decaying entropy coefficient
        """
        # Ensure current_step doesn't exceed horizon
        step = min(epoch_iter_step, self.config.entropy_coeff_horizon)
        
        # Linear interpolation
        # Coefficient decreases linearly from start_value to end_value
        decay_fraction = step / self.config.entropy_coeff_horizon
        current_coeff = self.config.entropy_coeff_start - (self.config.entropy_coeff_start - self.config.entropy_coeff_end) * decay_fraction
        return current_coeff
    


    def train(self):
        """
        Train the PPO agent.
        """
        # Initialize environment

        # Training loop
        for iter in range(self.config.num_iters):
            if iter % 10 == 0:
                print(f"iter {iter}")
                
            #### Collect all data ####
            # self.agent0.network.to('cpu')
            # self.agent1.network.to('cpu')
            state_tensor_p0, state_tensor_p1, action_tensor, sparse_rewards, infos = self.collect_trajectories()
            # self.agent0.network.to(self.device)
            # self.agent1.network.to(self.device)

            # Compute dense rewards
            # print("Collecting Rewards")
            dense_rewards = []
            old_log_probs0 = []
            old_log_probs1 = []
            # print(infos[0][0])
            for game in range(len(infos)):
                rewards = [info['phi_s_prime'] - info['phi_s'] for info in infos[game]]
                og_log0 = [info['agent_infos'][0]['og_log_probs'] for info in infos[game]]
                og_log1 = [info['agent_infos'][1]['og_log_probs'] for info in infos[game]]
                dense_rewards.append(rewards)
                old_log_probs0.append(og_log0)
                old_log_probs1.append(og_log1)
            dense_rewards = torch.tensor(dense_rewards, dtype=torch.float32)
            old_log_probs0 = torch.tensor(old_log_probs0, dtype=torch.float32)
            old_log_probs1 = torch.tensor(old_log_probs1, dtype=torch.float32)

            # Move to device
            dense_rewards = dense_rewards.to(self.device)
            old_log_probs0 = old_log_probs0.to(self.device)
            old_log_probs1 = old_log_probs1.to(self.device)

            # Compute shaped rewards
            shaped_rewards = sparse_rewards + dense_rewards * self.config.reward_shaping_factor

            #### DUMMY CHECK #### - STAY IN PLACE REWARD
            episode_rewards = [] 
            for episode in action_tensor:
                horizon_rewards = [] 
                for time_step in episode:
                    curr_reward = 0 
                    for action_pair in time_step:
                        if action_pair == (0, 0):
                            curr_reward += 1
                    horizon_rewards.append(curr_reward)
                episode_rewards.append(np.array(horizon_rewards))

            shaped_rewards = torch.tensor(episode_rewards, device=self.device)
            #### DUMMY CHECK #### - STAY IN PLACE REWARD

            avg_reward = shaped_rewards.sum(axis=1).mean()
            print("Average reward per episode", avg_reward)

            # wandb.log({"Average Reward": avg_reward, "Iteration": iter})

            # # get old log probs
            # # infos['og_log_probs'] is shape # (num_episodes, num_steps, num_agents)
            # print(infos.shape)
            # print(infos)
            # old_logprobs_p0, old_log_probs_p1 = torch.tensor(infos['og_log_probs'], dtype=torch.float32).split(1, dim=2)
            # old_logprobs_p0 = torch.squeeze(old_logprobs_p0, dim=2)
            # old_log_probs_p1 = torch.squeeze(old_log_probs_p1, dim=2)

            # Compute discounted returns and advantages
            returns = self.get_returns(shaped_rewards)
            
            # unroll everything into batch dimension
            state_tensor_p0 = state_tensor_p0.view(-1, *state_tensor_p0.shape[-3:]) # shape (num_episodes * num_steps, num_channels, h/w, h/w)
            state_tensor_p1 = state_tensor_p1.view(-1, *state_tensor_p1.shape[-3:])
            action_tensor_joint = action_tensor.view(-1, action_tensor.shape[-1]) # shape (num_episodes * num_steps, num_agents)
            action_tensor0 = action_tensor_joint[:, 0] # shape (num_episodes * num_steps)
            action_tensor1 = action_tensor_joint[:, 1] # shape (num_episodes * num_steps)
            returns = returns.view((-1, 1)) # shape (num_episodes * num_steps)
            old_log_probs0 = old_log_probs0.view(-1)
            old_log_probs1 = old_log_probs1.view(-1)
            
            # move to device
            state_tensor_p0 = state_tensor_p0.to(self.device)
            state_tensor_p1 = state_tensor_p1.to(self.device)
            action_tensor0 = action_tensor0.to(self.device)
            action_tensor1 = action_tensor1.to(self.device)
            returns = returns.to(self.device)
            old_log_probs0 = old_log_probs0.to(self.device)
            old_log_probs1 = old_log_probs1.to(self.device)

            # Update policy
            # 8 epochs 
            total_batches = self.config.num_epochs * self.config.num_mini_batches
            with tqdm(total=total_batches, desc="Training", leave=True) as pbar:
                for epoch in range(self.config.num_epochs):
                    # Compute epoch_iter_step 
                    epoch_iter_step = (self.config.num_epochs * iter) + epoch
                    entropy_coeff_current = self.compute_linear_decay_coefficient(epoch_iter_step)
                    
                    # shuffle data
                    indices = torch.randperm(state_tensor_p0.shape[0])
                    state_tensor_p0 = state_tensor_p0[indices]
                    state_tensor_p1 = state_tensor_p1[indices]
                    action_tensor0 = action_tensor0[indices]
                    action_tensor1 = action_tensor1[indices]
                    returns = returns[indices]
                    old_log_probs0 = old_log_probs0[indices]
                    old_log_probs1 = old_log_probs1[indices]
                    
                    # 6 minibatches
                    for i in range(self.config.num_mini_batches):
                        # select minibatch
                        start = ((self.config.horizon * self.config.num_episodes) // self.config.num_mini_batches) * i
                        end = ((self.config.horizon * self.config.num_episodes) // self.config.num_mini_batches) * (i + 1)
                        state_tensor_p0_batch = state_tensor_p0[start:end]
                        state_tensor_p1_batch = state_tensor_p1[start:end]
                        action_tensor0_batch = action_tensor0[start:end]
                        action_tensor1_batch = action_tensor1[start:end]
                        returns_batch = returns[start:end]
                        old_log_probs0_batch = old_log_probs0[start:end]
                        old_log_probs1_batch = old_log_probs1[start:end]
                        
                        # run update 
                        self.agent0.update_policy(state_tensor_p0_batch, action_tensor0_batch, returns_batch, old_logprobs=old_log_probs0_batch, entropy_coeff_current=entropy_coeff_current, debug=True)
                        self.agent1.update_policy(state_tensor_p1_batch, action_tensor1_batch, returns_batch, old_logprobs=old_log_probs1_batch, entropy_coeff_current=entropy_coeff_current)
                        
                        pbar.update(1)

        # wandb.finish()
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
    trainer = PPOTrainer(config)
    rewards = trainer.train()