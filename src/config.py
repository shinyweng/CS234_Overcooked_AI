class Config:
    """Configuration class for PPOAgent hyperparameters."""
    def __init__(self, layout="padded_cramped_room", seed_num=0):
        # Seeds
        self.seeds = [0, 10, 20, 30,40]
        self.seed = self.seeds[seed_num] 
        
        # Shared hyperparameters 
        self.device = 'mps'
        self.normalize_advantage = True
        self.update_freq = 5
        self.horizon = 400
        self.num_iters = 420
        self.num_epochs = 8
        self.num_mini_batches = 6
        self.reward_shaping_factor = 0.1 
        self.clip_param = 0.05 
        self.num_episodes = 30
        self.max_grad_norm = 0.1 
        self.entropy_coeff_start = 0.2
        self.entropy_coeff_end = 0.1
        self.entropy_coeff_horizon = 3e5
        self.vf_loss_coeff = 0.5

        # Layouts
        self.layouts = ["padded_" + layout for layout in ['cramped_room', 'asymmetric_advantages_tomato', 'coordination_ring', 'forced_coordination', 'counter_circuit']]
        self.max_width = 9
        self.max_height = 5

        # Set layout
        self.layout = layout

        # GAE 
        self.gae_gamma = 0.99 
        self.gae_lambda = 0.98 

        # Layout-specific hyperparameters
        self._set_layout_hyperparameters(layout)


    def _set_layout_hyperparameters(self, layout):
        """Set hyperparameters based on the chosen layout."""
        if layout == "padded_cramped_room":
            self.learning_rate = 1e-3

        elif layout == "padded_asymmetric_advantages_tomato":
            self.learning_rate = 1e-4

        elif layout == "padded_coordination_ring":
            self.learning_rate = 6e-4

        elif layout == "padded_forced_coordination":
            self.learning_rate = 8e-4
      
        elif layout == "padded_counter_circuit":
            self.learning_rate = 8e-4

        else:
            raise ValueError(f"Unknown layout: {layout}")
        