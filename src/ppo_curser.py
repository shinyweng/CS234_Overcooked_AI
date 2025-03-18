# ppo_curser.py

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
import wandb
from tqdm import tqdm
from overcooked_ai_py.mdp.actions import Action

class PPOModel(nn.Module):
    """
    PyTorch version of RllibPPOModel. Maps environment states to action probabilities and value estimates.
    """
    def __init__(self, obs_shape, num_actions, model_config):
        super(PPOModel, self).__init__()
        
        # Parse custom network params
        custom_params = model_config["custom_model_config"]
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"]
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        d2rl = custom_params["D2RL"]
        
        # Build network layers
        self.conv_layers = nn.ModuleList()
        
        # Initial conv layer with larger kernel
        if num_convs > 0:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=obs_shape[0],  # Assuming channel-first format
                    out_channels=num_filters,
                    kernel_size=5,
                    padding='same'
                )
            )
        
        # Additional conv layers
        for i in range(num_convs - 1):
            padding = 'same' if i < num_convs - 2 else 'valid'
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=3,
                    padding=padding
                )
            )
            
        # Calculate flattened conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            x = dummy_input
            for conv in self.conv_layers:
                x = F.leaky_relu(conv(x))
            conv_out_size = x.view(1, -1).size(1)
        
        # Dense layers
        self.dense_layers = nn.ModuleList()
        prev_size = conv_out_size
        
        for i in range(num_hidden_layers):
            if i > 0 and d2rl:
                # D2RL: Add skip connections by concatenating conv output
                self.dense_layers.append(
                    nn.Linear(prev_size + conv_out_size, size_hidden_layers)
                )
            else:
                self.dense_layers.append(
                    nn.Linear(prev_size, size_hidden_layers)
                )
            prev_size = size_hidden_layers
            
        # Output layers
        self.policy_head = nn.Linear(prev_size, num_actions)
        self.value_head = nn.Linear(prev_size, 1)
        
        # Save conv output size for D2RL
        self.conv_out_size = conv_out_size
        self.d2rl = d2rl

    def forward(self, obs):
        """
        Forward pass returning both action logits and value estimate
        """
        x = obs
        
        # Conv layers
        conv_out = None
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x))
        conv_out = x
        x = torch.flatten(x, start_dim=1)
        
        # Dense layers with optional D2RL skip connections
        for i, dense in enumerate(self.dense_layers):
            if i > 0 and self.d2rl:
                x = torch.cat([x, conv_out.view(conv_out.size(0), -1)], dim=1)
            x = F.leaky_relu(dense(x))
            
        # Output heads
        action_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return action_logits, value

    def get_action_dist(self, obs):
        """
        Returns action distribution for the given observation
        """
        action_logits, _ = self.forward(obs)
        return Categorical(logits=action_logits)

    def evaluate_actions(self, obs, actions):
        """
        Returns log probs and value estimates for given observations and actions
        """
        action_logits, values = self.forward(obs)
        dist = Categorical(logits=action_logits)
        
        action_log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        
        return action_log_probs, values.squeeze(-1), entropy

class PPOLSTMModel(nn.Module):
    """
    PyTorch version of RllibLSTMPPOModel with recurrent policy
    """
    def __init__(self, obs_shape, num_actions, model_config):
        super(PPOLSTMModel, self).__init__()
        
        # Parse custom params
        custom_params = model_config["custom_model_config"]
        num_hidden_layers = custom_params["NUM_HIDDEN_LAYERS"]
        size_hidden_layers = custom_params["SIZE_HIDDEN_LAYERS"] 
        num_filters = custom_params["NUM_FILTERS"]
        num_convs = custom_params["NUM_CONV_LAYERS"]
        cell_size = custom_params["CELL_SIZE"]
        
        # Conv layers
        self.conv_layers = nn.ModuleList()
        
        if num_convs > 0:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=obs_shape[0],
                    out_channels=num_filters, 
                    kernel_size=5,
                    padding='same'
                )
            )
            
        for i in range(num_convs - 1):
            padding = 'same' if i < num_convs - 2 else 'valid'
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=num_filters,
                    out_channels=num_filters,
                    kernel_size=3, 
                    padding=padding
                )
            )
            
        # Calculate conv output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, *obs_shape)
            x = dummy_input
            for conv in self.conv_layers:
                x = F.leaky_relu(conv(x))
            conv_out_size = x.view(1, -1).size(1)
            
        # Dense layers
        self.dense_layers = nn.ModuleList()
        prev_size = conv_out_size
        
        for _ in range(num_hidden_layers):
            self.dense_layers.append(
                nn.Linear(prev_size, size_hidden_layers)
            )
            prev_size = size_hidden_layers
            
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=prev_size,
            hidden_size=cell_size,
            batch_first=True
        )
        
        # Output heads
        self.policy_head = nn.Linear(cell_size, num_actions)
        self.value_head = nn.Linear(cell_size, 1)
        
        self.cell_size = cell_size

    def forward(self, obs, states, seq_lens=None):
        """
        Forward pass with recurrent state handling
        
        Args:
            obs: Tensor of shape [B, T, C, H, W] 
            states: Tuple of (h_0, c_0) LSTM states
            seq_lens: Tensor of sequence lengths for masking
        """
        # Reshape batch for conv processing
        batch_size, seq_len = obs.shape[:2]
        x = obs.view(-1, *obs.shape[2:])
        
        # Conv layers
        for conv in self.conv_layers:
            x = F.leaky_relu(conv(x))
        x = torch.flatten(x, start_dim=1)
        
        # Dense layers
        for dense in self.dense_layers:
            x = F.leaky_relu(dense(x))
            
        # Reshape for sequence processing
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM layer with state handling
        if seq_lens is not None:
            # Pack padded sequence
            x = nn.utils.rnn.pack_padded_sequence(
                x, seq_lens.cpu(), batch_first=True, enforce_sorted=False
            )
            
        lstm_out, new_states = self.lstm(x, states)
        
        if seq_lens is not None:
            # Unpack sequence
            lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
                lstm_out, batch_first=True
            )
            
        # Output heads
        action_logits = self.policy_head(lstm_out)
        values = self.value_head(lstm_out)
        
        return action_logits, values, new_states

    def get_initial_states(self, batch_size=1):
        """Returns initial LSTM states"""
        return (
            torch.zeros(1, batch_size, self.cell_size),
            torch.zeros(1, batch_size, self.cell_size)
        )

class PPOTrainer:
    def __init__(
        self,
        env,
        model_config,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.015,
        use_lstm=False,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        env.reset()
        
        self.env = env
        self.device = device
        self.n_steps = n_steps
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf or clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.use_lstm = use_lstm

        # Initialize models for both agents
        # Get the actual observation shape from a dummy state
        obs_shape = 26 #dummy_state['both_agent_obs'][0].shape  # This will give us the correct shape
        num_actions = len(Action.ALL_ACTIONS)  # Number of possible actions in Overcooked

        ModelClass = PPOLSTMModel if use_lstm else PPOModel
        self.model_agent1 = ModelClass(obs_shape, num_actions, model_config).to(device)
        self.model_agent2 = ModelClass(obs_shape, num_actions, model_config).to(device)

        # Initialize optimizers
        self.optimizer_agent1 = optim.Adam(self.model_agent1.parameters(), lr=learning_rate)
        self.optimizer_agent2 = optim.Adam(self.model_agent2.parameters(), lr=learning_rate)

    def collect_rollouts(self, n_episodes):
        """Collect environment rollouts using current policy"""
        # Storage for rollout data
        rollout_data = defaultdict(list)
        
        if self.use_lstm:
            # Initialize LSTM states
            lstm_states_agent1 = self.model_agent1.get_initial_states()
            lstm_states_agent2 = self.model_agent2.get_initial_states()

        for _ in range(n_episodes):
            state = self.env.reset()
            done = False
            episode_data = defaultdict(list)

            while not done:
                # Get observations for both agents
                obs_agent1 = torch.FloatTensor(state["both_agent_obs"][0]).unsqueeze(0).to(self.device)
                obs_agent2 = torch.FloatTensor(state["both_agent_obs"][1]).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    if self.use_lstm:
                        action_logits1, value1, lstm_states_agent1 = self.model_agent1(
                            obs_agent1, lstm_states_agent1
                        )
                        action_logits2, value2, lstm_states_agent2 = self.model_agent2(
                            obs_agent2, lstm_states_agent2
                        )
                    else:
                        action_logits1, value1 = self.model_agent1(obs_agent1)
                        action_logits2, value2 = self.model_agent2(obs_agent2)

                    # Sample actions
                    dist1 = Categorical(logits=action_logits1)
                    dist2 = Categorical(logits=action_logits2)
                    action1 = dist1.sample()
                    action2 = dist2.sample()
                    log_prob1 = dist1.log_prob(action1)
                    log_prob2 = dist2.log_prob(action2)

                # Execute actions
                next_state, reward, done, info = self.env.step(
                    [action1.item(), action2.item()]
                )

                # Store transition data
                episode_data["obs_agent1"].append(obs_agent1)
                episode_data["obs_agent2"].append(obs_agent2)
                episode_data["actions_agent1"].append(action1)
                episode_data["actions_agent2"].append(action2)
                episode_data["log_probs_agent1"].append(log_prob1)
                episode_data["log_probs_agent2"].append(log_prob2)
                episode_data["values_agent1"].append(value1)
                episode_data["values_agent2"].append(value2)
                episode_data["rewards"].append(reward)
                episode_data["dones"].append(done)

                state = next_state

            # Process episode data
            for key, values in episode_data.items():
                if isinstance(values[0], torch.Tensor):
                    episode_tensor = torch.cat(values)
                else:
                    episode_tensor = torch.tensor(values, device=self.device)
                rollout_data[key].append(episode_tensor)

        # Concatenate all episodes
        return {k: torch.stack(v) for k, v in rollout_data.items()}

    def compute_advantages_and_returns(self, values, rewards, dones):
        """Compute GAE advantages and returns"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        last_gae = 0
        last_return = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0
            else:
                next_non_terminal = 1.0 - dones[t + 1]
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            last_return = rewards[t] + self.gamma * next_non_terminal * last_return
            returns[t] = last_return

        return advantages, returns

    def update_policy(self, rollout_data, agent_num):
        """Update policy for one agent"""
        model = self.model_agent1 if agent_num == 1 else self.model_agent2
        optimizer = self.optimizer_agent1 if agent_num == 1 else self.optimizer_agent2
        
        obs = rollout_data[f"obs_agent{agent_num}"]
        actions = rollout_data[f"actions_agent{agent_num}"]
        old_log_probs = rollout_data[f"log_probs_agent{agent_num}"]
        advantages = rollout_data[f"advantages_agent{agent_num}"]
        returns = rollout_data[f"returns_agent{agent_num}"]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Create mini-batches
        dataset = TensorDataset(obs, actions, old_log_probs, advantages, returns)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Training loop
        for epoch in range(self.n_epochs):
            for batch in dataloader:
                b_obs, b_actions, b_old_log_probs, b_advantages, b_returns = batch

                # Get current policy outputs
                if self.use_lstm:
                    action_logits, values, _ = model(b_obs, model.get_initial_states())
                else:
                    action_logits, values = model(b_obs)

                # Calculate policy loss
                dist = Categorical(logits=action_logits)
                log_probs = dist.log_prob(b_actions)
                ratio = torch.exp(log_probs - b_old_log_probs)
                
                # PPO policy loss
                policy_loss1 = ratio * b_advantages
                policy_loss2 = torch.clamp(ratio, 1 - self.clip_range, 1 + self.clip_range) * b_advantages
                policy_loss = -torch.min(policy_loss1, policy_loss2).mean()

                # Value loss
                values_pred = values.squeeze()
                value_loss = F.mse_loss(values_pred, b_returns)

                # Entropy loss
                entropy_loss = -dist.entropy().mean()

                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss

                # Optimization step
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
                optimizer.step()

                # Calculate KL divergence
                approx_kl = ((ratio - 1) - log_probs).mean().item()
                if approx_kl > self.target_kl:
                    break

            if approx_kl > self.target_kl:
                break

    def train(self, total_timesteps, log_interval=100):
        """Main training loop"""
        # # Initialize wandb
        # wandb.init(project="overcooked-ppo", config={
        #     "learning_rate": self.optimizer_agent1.param_groups[0]["lr"],
        #     "n_steps": self.n_steps,
        #     "batch_size": self.batch_size,
        #     "n_epochs": self.n_epochs,
        #     "clip_range": self.clip_range,
        #     "ent_coef": self.ent_coef,
        #     "vf_coef": self.vf_coef,
        #     "use_lstm": self.use_lstm
        # })

        num_updates = total_timesteps // self.n_steps
        for update in tqdm(range(num_updates)):
            # Collect rollouts
            rollout_data = self.collect_rollouts(self.n_steps)

            # Compute advantages and returns for both agents
            for agent_num in [1, 2]:
                values = rollout_data[f"values_agent{agent_num}"]
                advantages, returns = self.compute_advantages_and_returns(
                    values, rollout_data["rewards"], rollout_data["dones"]
                )
                rollout_data[f"advantages_agent{agent_num}"] = advantages
                rollout_data[f"returns_agent{agent_num}"] = returns

            # Update both agents
            self.update_policy(rollout_data, agent_num=1)
            self.update_policy(rollout_data, agent_num=2)

            # Logging
            if update % log_interval == 0:
                mean_reward = rollout_data["rewards"].mean().item()
                # wandb.log({
                #     "mean_reward": mean_reward,
                #     "update": update
                # })

        # wandb.finish()

if __name__ == "__main__":
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
    from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

    # Initialize environment
    mdp = OvercookedGridworld.from_layout_name("cramped_room")
    env = OvercookedEnv.from_mdp(mdp, horizon=400)

    # Model configuration
    model_config = {
        "custom_model_config": {
            "NUM_HIDDEN_LAYERS": 2,
            "SIZE_HIDDEN_LAYERS": 64,
            "NUM_FILTERS": 32,
            "NUM_CONV_LAYERS": 3,
            "D2RL": True,
            "CELL_SIZE": 256  # Only used if use_lstm=True
        }
    }
    print(env)
    # Initialize trainer
    trainer = PPOTrainer(
        env=env,
        model_config=model_config,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        use_lstm=False  # Set to True to use LSTM model
    )

    # Train the agents
    trainer.train(total_timesteps=100000)