# config.py

class Config:
    """Configuration class for PPOAgent hyperparameters."""
    def __init__(self):
        self.normalize_advantage = True
        self.update_freq = 5
        self.horizon = 400
        self.layout = "cramped_room"
        self.learning_rate = 0.01
        self.reward_shaping_factor = 0.1
        self.clip_param = 0.05
        self.normalize_advantage = True
        self.num_episodes = 30

        # Gradient Clipping
        self.max_grad_norm = 0.1 
        
        self.entropy_coeff_start = 0.2
        self.entropy_coeff_end = 0.1
        self.entropy_coeff_horizon = 3e5
        self.vf_loss_coeff = 1e-4

        self.num_iters = 420
        self.num_epochs = 8
        self.num_mini_batches = 6
        
        # layouts
        self.layouts = ['cramped_room', 'asymmetric_advantages_tomato', 'coordination_ring', 'forced_coordination', 'counter_circuit']
        self.max_width = 9
        self.max_height = 5

        # GAE 
        self.gae_gamma = 0.99
        self.gae_lambda = 0.98
