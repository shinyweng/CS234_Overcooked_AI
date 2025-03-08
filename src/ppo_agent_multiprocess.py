# ppo_agent.py

import torch
import torch.nn as nn
from torch.distributions import Categorical
import numpy as np
import multiprocessing
from config import Config
from tqdm import tqdm, trange
import wandb
import time

from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair

# Constants
MAX_WIDTH = 9
MAX_HEIGHT = 5
CHANNELS = 26
NUM_AGENTS = 2

def get_obs(env, state):
    """
    Get observation from the environment.
    """

    # Featurizes a OvercookedState object into a stack of boolean masks that are easily readable by a CNN
    encoding = np.array(env.lossless_state_encoding_mdp(state))

    # Obtain lossless encoding of the state
    obs0, obs1 = encoding[0], encoding[1]

    # Permute from (w, h, c) to (c, w, h)
    obs0 = np.transpose(obs0, (2, 0, 1)) 
    obs1 = np.transpose(obs1, (2, 0, 1))

    ######## TESTING SHAPES ########
    assert obs0.shape == (CHANNELS, MAX_WIDTH, MAX_HEIGHT) == obs1.shape 
    ######## TESTING SHAPES ########

    return obs0, obs1 

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
    def __init__(self, mdp, config=None):
        super(PPOAgent, self).__init__()
        
        # Initialize configuration
        self.config = config if config else Config()
        self.device = self.config.device if config else 'cpu'
        self.action_space = Action.ALL_ACTIONS

        # Storing the mdp that the agent is in
        self.mdp = mdp
        
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
        observation = torch.tensor(get_obs(self.mdp, self.config.horizon, state)[self.agent_index], dtype=torch.float32).unsqueeze(0)#.to(self.device)
        distribution = self._get_distribution(observation)
        action_idx = distribution.sample()
        og_log_probs = distribution.log_prob(action_idx)
        action = Action.INDEX_TO_ACTION[action_idx.item()]
        info = {'og_distribution': distribution.probs.detach(), 'og_log_probs': og_log_probs.detach()}
        return action, info 
    
    def load_network_state_dict(self, state_dict):
        self.network.load_state_dict(state_dict)
    

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

        returns = advantages + values[:, :-1] # TD(lambda) return estimation
        return advantages, returns

    def update_policy(self, observations, actions, rewards, old_logprobs, entropy_coeff_current, debug=False):
        """
        Update the policy using PPO.

        observations: torch.tensor of shape (num_episodes * num_steps, num_channels, w, h)
        actions: torch.tensor of shape (num_episodes * num_steps)
        advantages: torch.tensor of shape (num_episodes * num_steps)
        returns: torch.tensor of shape (num_episodes * num_steps)
        old_logprobs: torch.tensor of shape (num_episodes * num_steps)
        entropy_coeff_current: float
        """
        # observations = observations.to(self.device)
        # actions = actions.to(self.device)
        # returns = returns.to(self.device)
        # old_logprobs = old_logprobs.to(self.device)
        
        distribution = self._get_distribution(observations)
        # print(observations.shape)
        value = self._get_value_estimate(observations)
        # print(value.shape)

        # Compute advantages
        # turn value and returns back to episode shape
        value_reshaped = value.view((-1, self.config.horizon))
        # returns_reshaped = returns.view((-1, self.config.horizon))
        rewards_reshaped = rewards.view((-1, self.config.horizon))

        advantages, returns = self.compute_gae(value_reshaped, rewards_reshaped)
        advantages = advantages.view(-1)
        returns = returns.view((-1, 1))

        # returns = advantage + values 
        # print(returns[0])
        # print(advantages[0])
        # print(value[0])
        # print('check advantage and value', returns == advantages + value)
        # print(returns_reshaped == advantages + value_reshaped)


        # normalize advantages
        advantages = ( advantages - advantages.mean()) / (advantages.std() + 1e-8)
        

        # policy loss
        logprobs = distribution.log_prob(actions)
        ratio = torch.exp(logprobs - old_logprobs)
        clipped_ratio = ratio
        # clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_param, 1 + self.config.clip_param)
        policy_loss = torch.min(ratio * advantages, clipped_ratio * advantages).mean()

        # print("Returns range", returns.min().item(), returns.max().item())
        # print("ADVANTAGE AT MAX RETURNS", advantages[torch.argmax(returns)])
        # print("clipped_ration at max returns", ratio[torch.argmax(returns)])
        # print("Value range", value.min().item(), value.max().item())

        # value loss
        assert value.shape == returns.shape # (2000, 1)

        value_loss = self.value_loss_fn(value, returns)
    
        # Compute entropy loss
        # Entropy encourages exploration by penalizing deterministic policies
        entropy_loss = distribution.entropy().mean()
        
        total_loss = -policy_loss + (self.config.vf_loss_coeff * value_loss) - (entropy_coeff_current * entropy_loss)
    
        # if debug: print(f"value loss: {value_loss}, policy loss: {policy_loss}, entropy loss: {entropy_loss}")
        
        # Zero out previous gradients
        self.optimizer.zero_grad()
        
        # Compute and clip gradients
        total_loss.backward()
        # if debug: print("Total Loss", total_loss.item())
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=self.config.max_grad_norm)
        
        # Update network parameters
        self.optimizer.step()

        
        # print(self.network.action_head.weight)
        # print(self.network.value_head.weight)
            
        return total_loss.item()


class PPOTrainer:
    """Handles the PPO training process."""
    def __init__(self, config, pool):
        self.config = config
        self.pool = pool
        
        # Initialize WandB
        # wandb.init(project="overcooked-ppo", config=self.config.__dict__)
        # Initialize environmentƒself.
        mdp = OvercookedGridworld.from_layout_name(layout_name=self.config.layout)   
        self.env = OvercookedEnv.from_mdp(mdp, horizon=self.config.horizon)
        
        # Initialize agents
        self.agent0 = PPOAgent(self.env.mdp, self.config)
        self.agent1 = PPOAgent(self.env.mdp, self.config)
        self.agent_pair = AgentPair(self.agent0, self.agent1)
        self.device = self.config.device

    @staticmethod
    def perform_rollout_wrapper(agent0_statedict, agent1_statedict):
        global worker_env
        global worker_agent0
        global worker_agent1
        global worker_agent_pair

        worker_agent0.load_network_state_dict(agent0_statedict)
        worker_agent1.load_network_state_dict(agent1_statedict)

        worker_agent_pair = AgentPair(worker_agent0, worker_agent1)
        return worker_env.get_rollouts(worker_agent_pair, 6, info=False, display_phi=True)

    def collect_trajectories_parallel(self):
        # start timer
        start = time.time()   
        agent0_state_dict = {k: v.cpu().detach().clone() for k, v in self.agent0.network.state_dict().items()}
        agent1_state_dict = {k: v.cpu().detach().clone() for k, v in self.agent1.network.state_dict().items()}
        parallel_trajectories = self.pool.starmap(self.perform_rollout_wrapper, [(agent0_state_dict, agent1_state_dict)] * 5)
        # end timer
        end = time.time()
        print(f"Time taken to collect parallel trajectories: {end - start}")

        states = np.concatenate([trajectory["ep_states"] for trajectory in parallel_trajectories], axis=0) # shape (num_episodes, num_steps)
        actions = np.concatenate([trajectory["ep_actions"] for trajectory in parallel_trajectories], axis=0) # shape (num_episodes, num_steps, num_agents, num_actions)
        # if action is to the right, give reward + 1
        
        rewards = np.concatenate([trajectory["ep_rewards"] for trajectory in parallel_trajectories], axis=0) # shape (num_episodes, num_steps)
        dones = np.concatenate([trajectory["ep_dones"] for trajectory in parallel_trajectories], axis=0) # shape (num_episodes, num_steps)
        infos = np.concatenate([trajectory["ep_infos"] for trajectory in parallel_trajectories], axis=0) # shape (num_episodes, num_steps, num_agents)


        # Featurize and convert to tensors
        state_p0 = np.array([[get_obs(self.env.mdp, self.env.horizon, state)[0] for state in episode] for episode in states])
        state_p1 = np.array([[get_obs(self.env.mdp, self.env.horizon, state)[1] for state in episode] for episode in states])
        state_tensor_p0 = torch.tensor(state_p0, dtype=torch.float32) # shape (num_episodes, num_steps, num_agents, num_channels, h/w, h/w)
        state_tensor_p1 = torch.tensor(state_p1, dtype=torch.float32)

        action_tensor = torch.tensor([[[Action.ACTION_TO_INDEX[act] for act in agent_acts] for agent_acts in episode] for episode in actions], dtype=torch.long) # shape (num_episodes, num_steps, num_agents)
        reward_tensor = torch.tensor(rewards.astype(np.float32), dtype=torch.float32)

        return state_tensor_p0, state_tensor_p1, action_tensor, reward_tensor, infos


    def collect_trajectories(self):
        """
        Collect trajectories from the environment.
        """

        """
        initialize our worker pool and give it the environemnt and a copy of the agents each time
        - either we 


        wrapper: get_rollout(i):
        - returns: env.get_rollouts(length 1)

        pool.map(get_rollouts, range(num_episodes))

        accumulate the individual rollouts

        states = concat[rollout['states'] for rollout in rollouts]
        ...*

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

        state_p0 = np.array([[get_obs(self.env.mdp, self.env.horizon, state)[0] for state in episode] for episode in states])
        state_p1 = np.array([[get_obs(self.env.mdp, self.env.horizon, state)[1] for state in episode] for episode in states])
        state_tensor_p0 = torch.tensor(state_p0, dtype=torch.float32) # shape (num_episodes, num_steps, num_agents, num_channels, h/w, h/w)
        state_tensor_p1 = torch.tensor(state_p1, dtype=torch.float32)

        action_tensor = torch.tensor([[[Action.ACTION_TO_INDEX[act] for act in agent_acts] for agent_acts in episode] for episode in actions], dtype=torch.long) # shape (num_episodes, num_steps, num_agents)
        # print("rewards", rewards)
        reward_tensor = torch.tensor(rewards, dtype=torch.float32)

        return state_tensor_p0, state_tensor_p1, action_tensor, reward_tensor, infos
    
    # def get_returns(self, shaped_reward):
    #     """"
    #     shaped_reward: torch.tensor of shape (num_episodes, num_steps)

    #     Returns: returns of shape (num_episodes, num_steps, with gamma=1)
    #     """
    #     returns = torch.zeros_like(shaped_reward)
    #     for t in reversed(range(len(shaped_reward[0]) -1)): 
    #         returns[:, t] = shaped_reward[:, t] + self.config.gae_gamma * returns[:, t + 1]
    #     return returns

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
            with torch.no_grad():
                # self.agent0.network.to('cpu')
                # self.agent1.network.to('cpu')
                state_tensor_p0, state_tensor_p1, action_tensor, sparse_rewards, infos = self.collect_trajectories_parallel()
                # self.agent0.network.to(self.device)
                # self.agent1.network.to(self.device)

            # Compute dense rewards
            # print("Collecting Rewards")
            # dense_rewards = []
            dense_rewards0 = []
            dense_rewards1 = []
            old_log_probs0 = []
            old_log_probs1 = []
            print(infos[0][0])
            for game in range(len(infos)):
                # potential = info['phi_s_prime'] - info['phi_s']
                # rewards = [info['phi_s_prime'] - info['phi_s'] for info in infos[game]]
                rewards0 = [info["shaped_r_by_agent"][0] for info in infos[game]]
                rewards1 = [info["shaped_r_by_agent"][1] for info in infos[game]]

                og_log0 = [info['agent_infos'][0]['og_log_probs'] for info in infos[game]]
                og_log1 = [info['agent_infos'][1]['og_log_probs'] for info in infos[game]]
                # dense_rewards.append(rewards)
                dense_rewards0.append(rewards0)
                dense_rewards1.append(rewards1)
                old_log_probs0.append(og_log0)
                old_log_probs1.append(og_log1)
            # dense_rewards = torch.tensor(dense_rewards, dtype=torch.float32)
            dense_rewards0 = torch.tensor(dense_rewards0, dtype=torch.float32)
            dense_rewards1 = torch.tensor(dense_rewards1, dtype=torch.float32)
            old_log_probs0 = torch.tensor(old_log_probs0, dtype=torch.float32)
            old_log_probs1 = torch.tensor(old_log_probs1, dtype=torch.float32)

            # Compute shaped rewards
            # shaped_rewards = sparse_rewards + dense_rewards * self.config.reward_shaping_factor
            shaped_rewards0 = sparse_rewards + dense_rewards0 * self.config.reward_shaping_factor
            shaped_rewards1 = sparse_rewards + dense_rewards1 * self.config.reward_shaping_factor

            avg_reward0 = shaped_rewards0.sum(axis=1).mean()
            print("Average reward per episode for agent0", avg_reward0)
            avg_reward1 = shaped_rewards1.sum(axis=1).mean()
            print("Average reward per episode for agent0", avg_reward1)


            # wandb.log({"Average Reward": avg_reward, "Iteration": iter})

            # # get old log probs
            # # infos['og_log_probs'] is shape # (num_episodes, num_steps, num_agents)
            # print(infos.shape)
            # print(infos)
            # old_logprobs_p0, old_log_probs_p1 = torch.tensor(infos['og_log_probs'], dtype=torch.float32).split(1, dim=2)
            # old_logprobs_p0 = torch.squeeze(old_logprobs_p0, dim=2)
            # old_log_probs_p1 = torch.squeeze(old_log_probs_p1, dim=2)

            
            # unroll everything into batch dimension
            state_tensor_p0 = state_tensor_p0.view(-1, *state_tensor_p0.shape[-3:]) # shape (num_episodes * num_steps, num_channels, h/w, h/w)
            state_tensor_p1 = state_tensor_p1.view(-1, *state_tensor_p1.shape[-3:])
            action_tensor_joint = action_tensor.view(-1, action_tensor.shape[-1]) # shape (num_episodes * num_steps, num_agents)
            action_tensor0 = action_tensor_joint[:, 0] # shape (num_episodes * num_steps)
            action_tensor1 = action_tensor_joint[:, 1] # shape (num_episodes * num_steps)
            # returns = returns.view((-1, 1)) # shape (num_episodes * num_steps)
            old_log_probs0 = old_log_probs0.view(-1)
            old_log_probs1 = old_log_probs1.view(-1)
            # rewards = shaped_rewards.veiew(-1)
            rewards0 = shaped_rewards0.view(-1)
            rewards1 = shaped_rewards1.view(-1)
            
            # move to device
            state_tensor_p0 = state_tensor_p0.to(self.device)
            state_tensor_p1 = state_tensor_p1.to(self.device)
            action_tensor0 = action_tensor0.to(self.device)
            action_tensor1 = action_tensor1.to(self.device)
            # returns = returns.to(self.device)
            old_log_probs0 = old_log_probs0.to(self.device)
            old_log_probs1 = old_log_probs1.to(self.device)
            # rewards = rewards.to(self.device)
            rewards0 = rewards0.to(self.device)
            rewards1 = rewards1.to(self.device)

            # Update policy
            # 8 epochs 
            for epoch in trange(self.config.num_epochs + 1000):
                # Compute epoch_iter_step 
                epoch_iter_step = (self.config.num_epochs * iter) + epoch
                entropy_coeff_current = self.compute_linear_decay_coefficient(epoch_iter_step)

                minibatch_losses0 = []
                minibatch_losses1 = []
                
                # shuffle data
                indices = torch.randperm(state_tensor_p0.shape[0])
                state_tensor_p0 = state_tensor_p0[indices]
                state_tensor_p1 = state_tensor_p1[indices]
                action_tensor0 = action_tensor0[indices]
                action_tensor1 = action_tensor1[indices]
                # returns = returns[indices]
                old_log_probs0 = old_log_probs0[indices]
                old_log_probs1 = old_log_probs1[indices]
                # rewards = rewards[indices]
                rewards0 = rewards0[indices]
                rewards1 = rewards1[indices]
                
                # 6 minibatches
                for i in range(self.config.num_mini_batches):
                    # select minibatch
                    start = ((self.config.horizon * self.config.num_episodes) // self.config.num_mini_batches) * i
                    end = ((self.config.horizon * self.config.num_episodes) // self.config.num_mini_batches) * (i + 1)
                    state_tensor_p0_batch = state_tensor_p0[start:end]
                    state_tensor_p1_batch = state_tensor_p1[start:end]
                    action_tensor0_batch = action_tensor0[start:end]
                    action_tensor1_batch = action_tensor1[start:end]
                    # returns_batch = returns[start:end]
                    old_log_probs0_batch = old_log_probs0[start:end]
                    old_log_probs1_batch = old_log_probs1[start:end]
                    # rewards_batch = rewards[start:end]
                    rewards0_batch = rewards0[start:end]
                    rewards1_batch = rewards1[start:end]

                    
                    # run update 
                    agent0_loss = self.agent0.update_policy(state_tensor_p0_batch, action_tensor0_batch, rewards0_batch, old_logprobs=old_log_probs0_batch, entropy_coeff_current=entropy_coeff_current, debug=True)
                    agent1_loss = self.agent1.update_policy(state_tensor_p1_batch, action_tensor1_batch, rewards1_batch, old_logprobs=old_log_probs1_batch, entropy_coeff_current=entropy_coeff_current)

                    minibatch_losses0.append(agent0_loss)
                    minibatch_losses1.append(agent1_loss)
                print("Minibatch Losses: agent0", np.mean(minibatch_losses0), "agent1", np.mean(minibatch_losses1))        
                        
        # wandb.finish()
        print("Reward:", rewards)
        return rewards


def worker_init(config):
    global worker_env
    global worker_agent0
    global worker_agent1
    mdp = OvercookedGridworld.from_layout_name(layout_name=config.layout)
    worker_env = OvercookedEnv.from_mdp(mdp, horizon=config.horizon)

    worker_agent0 = PPOAgent(worker_env.mdp, config)
    worker_agent0.network.to('cpu')
    worker_agent1 = PPOAgent(worker_env.mdp, config)
    worker_agent1.network.to('cpu')


if __name__ == "__main__":  
    config = Config()
    
    # Set seed for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    with multiprocessing.Pool(5, worker_init, [config]) as pool:
        trainer = PPOTrainer(config, pool)
        rewards = trainer.train()