"""
Config for Inverse Dynamics algorithms.
"""

from robomimic.config.base_config import BaseConfig
from robomimic.config import BCConfig


class InverseDynamicsConfig(BCConfig):
    ALGO_NAME = "invdyna"

    def algo_config(self):
        """
        This function populates the `config.algo` attribute of the config, and is given to the 
        `Algo` subclass (see `algo/algo.py`) for each algorithm through the `algo_config` 
        argument to the constructor. Any parameter that an algorithm needs to determine its 
        training and test-time behavior should be populated here.
        """

        super(InverseDynamicsConfig, self).algo_config() # populating rest of the BC config params

        # optimization parameters
        self.algo.optim_params.policy.learning_rate.initial = 1e-4      # policy learning rate
        self.algo.optim_params.policy.learning_rate.decay_factor = 0.1  # factor to decay LR by (if epoch schedule non-empty)
        self.algo.optim_params.policy.learning_rate.epoch_schedule = [] # epochs where LR decay occurs
        self.algo.optim_params.policy.regularization.L2 = 0.00          # L2 regularization strength

        # loss weights
        self.algo.loss.l2_weight = 1.0      # L2 loss weight
        self.algo.loss.l1_weight = 0.0      # L1 loss weight
        self.algo.loss.cos_weight = 0.0     # cosine loss weight

        # RNN policy settings
        self.algo.rnn.enabled = False       # whether to train RNN policy
        self.algo.rnn.horizon = 10          # unroll length for RNN - should usually match train.seq_length
        self.algo.rnn.hidden_dim = 400      # hidden dimension size    
        self.algo.rnn.rnn_type = "LSTM"     # rnn type - one of "LSTM" or "GRU"
        self.algo.rnn.num_layers = 2        # number of RNN layers that are stacked
        self.algo.rnn.open_loop = False     # if True, action predictions are only based on a single observation (not sequence)
        self.algo.rnn.kwargs.bidirectional = False            # rnn kwargs
        self.algo.rnn.kwargs.do_not_lock_keys()

        # Latent diffusion policy settings
        self.algo.latent_diffusion.enabled = False
        self.algo.latent_diffusion.encoder_mlp_dims = [400,400,256]
        self.algo.latent_diffusion.denoiser_mlp_hidden_dims = [400,400]
        self.algo.latent_diffusion.num_timesteps = 100
        self.algo.latent_diffusion.timestep_dim = 256