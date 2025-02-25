# ppo_agent.py

import torch
import torch.nn as nn
import numpy as np
from config import Config
from tqdm import tqdm, trange


from overcooked_ai_py.mdp.actions import Action
from torch.distributions import Categorical
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair

def get_obs(env, state):
    """
    Get observation from the environment.
    """
    encoding = np.array(env.lossless_state_encoding_mdp(state))
    obs0, obs1 = encoding[0], encoding[1]
    obs0 = np.transpose(obs0, (2, 0, 1)) # permute from (h/w, h/w, c) to (c, h/w, h/w)
    obs1 = np.transpose(obs1, (2, 0, 1)) # permute from (h/w, h/w, c) to (c, h/w, h/w)
    return obs0, obs1


class PPONetwork(nn.Module):
    """
    Neural network for PPO agent.
    """
    def __init__(self, input_channels=26, action_space_size=6):
        super(PPONetwork, self).__init__()
        
        self.model_trunk = nn.Sequential(
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
        )

        self.action_head = nn.Linear(32, action_space_size)
        self.value_head = nn.Linear(32, 1)

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
        self.action_space = Action.ALL_ACTIONS

        # Storing the env that the agent is in
        self.env = env
        
        # Initialize neural network
        self.network = PPONetwork(
            input_channels=26, 
            action_space_size=len(self.action_space)
        )
        
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

    def action(self, state):
        """
        Get action from the current policy.
        """
        observation = torch.tensor(get_obs(self.env, state)[self.agent_index], dtype=torch.float32).unsqueeze(0)
        distribution = self._get_distribution(observation)
        action_idx = distribution.sample()
        og_log_probs = distribution.log_prob(action_idx)
        action = Action.INDEX_TO_ACTION[action_idx.item()]
        info = {'og_distribution': distribution.probs, 'og_log_probs': og_log_probs}
        return action, info 

    def update_policy(self, observations, actions, advantages, old_logprobs):
        """
        Update the policy using PPO.
        """

        distribution = self._get_distribution(observations)
        logprobs = distribution.log_prob(actions)
        # print(old_logprobs.shape)
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1 - self.config.eps_clip, 1 + self.config.eps_clip)
        loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class PPOTrainer:
    """Handles the PPO training process."""
    def __init__(self, config):
        self.config = config
        
        # Initialize environment
        mdp = OvercookedGridworld.from_layout_name(layout_name=self.config.layout)   
        self.env = OvercookedEnv.from_mdp(mdp, horizon=self.config.horizon)
        
        # Initialize agents
        self.agent0 = PPOAgent(self.env, self.config)
        self.agent1 = PPOAgent(self.env, self.config)
        self.agent_pair = AgentPair(self.agent0, self.agent1)

    def collect_trajectories(self):
        """
        Collect trajectories from the environment.
        """
        trajectories = self.env.get_rollouts(self.agent_pair, self.config.num_episodes, info=False, display_phi=True)
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
    
    def train(self):
        """
        Train the PPO agent.
        """
        # Initialize environment

        # Training loop
        for epoch in range(self.config.num_epochs):

            state_tensor_p0, state_tensor_p1, action_tensor, sparse_rewards, infos = self.collect_trajectories()

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


            # Compute shaped rewards
            shaped_rewards = sparse_rewards + dense_rewards * self.config.reward_shaping_factor

            # # get old log probs
            # # infos['og_log_probs'] is shape # (num_episodes, num_steps, num_agents)
            # print(infos.shape)
            # print(infos)
            # old_logprobs_p0, old_log_probs_p1 = torch.tensor(infos['og_log_probs'], dtype=torch.float32).split(1, dim=2)
            # old_logprobs_p0 = torch.squeeze(old_logprobs_p0, dim=2)
            # old_log_probs_p1 = torch.squeeze(old_log_probs_p1, dim=2)

            # Compute discounted returns and advantages
            returns = self.get_returns(shaped_rewards)
            advantages = self.compute_advantages(returns)

            # unroll everything into batch dimension
            state_tensor_p0 = state_tensor_p0.view(-1, *state_tensor_p0.shape[-3:]) # shape (num_episodes * num_steps, num_channels, h/w, h/w)
            state_tensor_p1 = state_tensor_p1.view(-1, *state_tensor_p1.shape[-3:])
            action_tensor_joint = action_tensor.view(-1, action_tensor.shape[-1]) # shape (num_episodes * num_steps, num_agents)
            action_tensor0 = action_tensor_joint[:, 0] # shape (num_episodes * num_steps)
            action_tensor1 = action_tensor_joint[:, 1] # shape (num_episodes * num_steps)
            advantages = advantages.view(-1) # shape (num_episodes * num_steps)
            old_log_probs0 = old_log_probs0.view(-1)
            old_log_probs1 = old_log_probs1.view(-1)
            

            # Update policy
            for _ in range(self.config.update_freq):
                self.agent0.update_policy(state_tensor_p0, action_tensor0, advantages, old_logprobs=old_log_probs0)
                self.agent1.update_policy(state_tensor_p1, action_tensor1, advantages, old_logprobs=old_log_probs1)

        return rewards
    
if __name__ == "__main__":  
    config = Config()
    trainer = PPOTrainer(config)
    rewards = trainer.train()