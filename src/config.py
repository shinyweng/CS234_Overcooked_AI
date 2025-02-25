# config.py

class Config:
    """Configuration class for PPOAgent hyperparameters."""
    def __init__(self):
        self.normalize_advantage = True
        self.update_freq = 5
        self.eps_clip = 0.2
        self.horizon = 400
        self.layout = "cramped_room"
        self.learning_rate = 0.01
        self.num_epochs = 100
        self.num_batches = 4
        self.reward_shaping_factor = 0.1
        self.eps_clip = 0.2
        self.normalize_advantage = True
        self.num_episodes = 2