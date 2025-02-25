import torch
import torch.nn as nn
import argparse
import numpy as np
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.envs.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.agents.random_agent import RandomAgent
from overcooked_ai_py.agents.agent import AgentPair
from src.agent import PPOAgent



def collect_trajectories(env, agent_pair, num_episodes=1):

    # collect trajectories in MDP space
    trajectories = env.get_rollouts(agent_pair, num_episodes)

    states = trajectories["ep_states"] # shape (num_episodes, num_steps, 1)
    actions = trajectories["ep_actions"] # shape (num_episodes, num_steps, num_agents, num_actions)
    rewards = trajectories["ep_rewards"] # shape (num_episodes, num_steps)
    dones = trajectories["ep_dones"] # shape (num_episodes, num_steps)
    infos = trajectories["ep_infos"] # shape (num_episodes, num_steps, num_agents)

    # featurize and convert to tensors
    print()

    state_p0 = [[env.lossless_state_encoding(state)[0] for state in episode] for episode in states]
    state_p1 = [[env.lossless_state_encoding(state)[1] for state in episode] for episode in states]
    state_tensor_p0 = torch.tensor(state_p0)
    state_tensor_1 = torch.tensor(state_p1)
    
    action_tensor = torch.tensor([[[Action.ACTION_TO_INDEX[act] for act in agent_acts] for agent_acts in episode] for episode in actions])
    reward_tensor = torch.tensor(rewards)

    
    

    return state_tensor_p0, state_tensor_1, action_tensor, reward_tensor, infos

    
def get_discounted_returns(trajectories, gamma):
    """"
    Compute discounted returns for each trajectory in the batch.
    Args:
        trajectories (list): List of trajectories, each trajectory is a dictionary containing 'reward'.
        gamma (float): Discount factor.
    Returns:
        returns (np.ndarray): Array of discounted returns for each trajectory.
    """
    # Compute discounted returns
    all_returns = []
    for path in trajectories:
        rewards = path["reward"]
        returns = np.zeros(len(rewards))
        G_t = 0
        for i in range(len(rewards)-1, -1, -1):
            G_t = rewards[i] + gamma * G_t
            returns[i] = G_t

        all_returns.append(returns)
    returns = np.concatenate(all_returns)
    return returns

def compute_advantages(returns):
    """
    Compute advantages for each trajectory in the batch.
    Args:
        trajectories (list): List of trajectories, each trajectory is a dictionary containing 'reward'.
        returns (np.ndarray): Array of discounted returns for each trajectory.
    """
    if self.config.normalize_advantage:        
        advantages = self.normalize_advantage(returns)
    return advantages


def train_PPO():
    # Extract Parameters 
    horizon = args.horizon
    seed = args.seed
    layout = args.layout
    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_episodes = args.num_episodes
    num_batches = args.num_batches
    gamma = args.gamma
    reward_shaping_factor = args.reward_shaping_factor

    # Initialize environment
    mdp = OvercookedGridworld.from_layout_name(layout=layout)
    env = OvercookedEnv(mdp, horizon=horizon)
    agent1, agent2 = PPOAgent(), PPOAgent()
    agent_pair = AgentPair(agent1, agent2)
    all_rewards = [] # Store rewards for each episode


    optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

    # Training Loop
    for t in range(num_epochs):
        state_tensor_p0, state_tensor_p1, action_tensor, sparse_rewards, infos = collect_trajectories(env, agent_pair, num_episodes=num_episodes)
        # shapes are
        # state_tensor0: (num_episodes, num_steps, h/w, h/w, layers)
        # state_tensor1: (num_episodes, num_steps, h/w, h/w, layers)
        # action_tensor: (num_episodes, num_steps, num_agents, num_actions)
        # sparse_rewards: (num_episodes, num_steps)
        
        # TODO: vectorize this
        dense_rewards = []
        # num episodes, num steps, num agents = 2
        for game in range(len(infos)):
            rewards = []
            for d in infos[game]:
                potential = d['phi_s_prime'] - d['phi_s']
                rewards.append(potential)
            dense_rewards.append(rewards)
        dense_rewards = torch.tensor(dense_rewards)
        shaped_reward =  sparse_rewards + dense_rewards * reward_shaping_factor

        shaped_reward_p0 = shaped_reward.copy()
        shaped_reward_p1  = shaped_reward.copy()

        all_rewards.append(shaped_reward)
        # Compute discounted returns
        returns = get_discounted_returns(shaped_reward, gamma)
        # Compute advantages and normalize if flag is set 
        advantages = compute_advantages(returns)
        
        
        
        # Update policy
        for k in range(self.config.update_freq):
            self.baseline_network.update_baseline(returns, observations)
            self.update_policy(observations, actions, advantages, 
                                old_logprobs)
        
        
        
        
    return


def parse_arguments():
    # Argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--num_hidden', type=int, default=64)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--num_actions', type=int, default=5)
    parser.add_argument('--num_states', type=int, default=10)
    parser.add_argument('--num_episodes', type=int, default=10)
    parser.add_argument('--layout', type=str, default='cramped_room')
    parser.add_argument('--update_freq', type=int, default=10)

    return parser.parse_args()

def parse_args_from_config():
    # Load configuration from a file
    # This is a placeholder for loading a config file
    config = {
        'num_epochs': 1000,
        'num_hidden': 64,
        'learning_rate': 0.01,
        'horizon': 200,
        'layout': 'cramped_room',
        'num_agents': 2,
        'num_actions': 5,
        'num_states': 10,
        'num_episodes': 10,
        'num_batches': 10
    }
    return config

if __name__ == '__main__':
    # Parse from ArgumentParser
    # args = parse_arguments()

    # Parse from Config File 
    args = parse_args_from_config()
    
    # Train PPO Agent
    train_PPO(args)