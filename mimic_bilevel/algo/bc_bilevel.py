import numpy as np
from collections import OrderedDict
import copy
import time
import datetime
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
import torch.distributions as D

from robomimic.algo.bc import BC_RNN
# import robomimic.models.policy_nets as PolicyNets
import mimicplay.models.policy_nets as PolicyNets
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.tensor_utils as TensorUtils
# import robomimic.utils.geometry as geometry
from robomimic.algo import register_algo_factory_func

from mimicplay.algo import Highlevel_GMM_pretrain

@register_algo_factory_func("mimic_bilevel")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the MimicPlay algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    if not algo_config.bc_bilevel.enabled: # if not using bilevel algo in this file, return mimicplay
        return Highlevel_GMM_pretrain_mimicplay, {}

    return BC_Bilevel(), {}

class Highlevel_GMM_pretrain_mimicplay(Highlevel_GMM_pretrain):

    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        assert self.algo_config.highlevel.enabled
        assert not self.algo_config.lowlevel.enabled

        # del self.obs_shapes['robot0_eef_pos_future_traj']
        self.ac_dim = self.algo_config.highlevel.ac_dim

        self.nets = nn.ModuleDict()
        self.nets["policy"] = PolicyNets.GMMActorNetwork(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            num_modes=self.algo_config.gmm.num_modes,
            min_std=self.algo_config.gmm.min_std,
            std_activation=self.algo_config.gmm.std_activation,
            low_noise_eval=self.algo_config.gmm.low_noise_eval,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.save_count = 0

        self.nets = self.nets.float().to(self.device)

    def process_batch_for_training(self, batch):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, 0, :] for k in batch["obs"]} # only keep first obs
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        assert input_batch["goal_obs"] is not None
        input_batch["actions"] = batch["actions"].view([batch["actions"].shape[0], -1]) # merge time and ac dims
        assert input_batch["actions"].shape[-1] == self.ac_dim

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))

    def postprocess_batch_for_training(self, batch, obs_normalization_stats):
        """
        Processes input batch from a data loader to filter out
        relevant information and prepare the batch for training.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader

        Returns:
            input_batch (dict): processed and filtered batch that
                will be used for training
        """

        # ensure obs_normalization_stats are torch Tensors on proper device
        obs_normalization_stats = TensorUtils.to_float(
            TensorUtils.to_device(TensorUtils.to_tensor(obs_normalization_stats), self.device))

        # we will search the nested batch dictionary for the following special batch dict keys
        # and apply the processing function to their values (which correspond to observations)
        obs_keys = ["obs", "next_obs", "goal_obs"]

        def recurse_helper(d):
            """
            Apply process_obs_dict to values in nested dictionary d that match a key in obs_keys.
            """
            for k in d:
                if k in obs_keys:
                    # found key - stop search and process observation
                    if d[k] is not None:
                        d[k] = ObsUtils.process_obs_dict(d[k])
                        if obs_normalization_stats is not None:
                            d[k] = ObsUtils.normalize_obs(d[k], obs_normalization_stats=obs_normalization_stats)
                elif isinstance(d[k], dict):
                    # search down into dictionary
                    recurse_helper(d[k])

        recurse_helper(batch)

        # TODO move line below to above function maybe; or remove fully to use what's in mimicplay
        # batch["goal_obs"]["agentview_image"] = batch["goal_obs"]["agentview_image"][:, 0]

        return TensorUtils.to_device(TensorUtils.to_float(batch), self.device)

    def _forward_training(self, batch):
        """
        Internal helper function for BC algo class. Compute forward pass
        and return network outputs in @predictions dict.

        Args:
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            predictions (dict): dictionary containing network outputs
        """

        dists = self.nets["policy"].forward_train(
            obs_dict=batch["obs"],
            goal_dict=batch["goal_obs"]
        )

        # make sure that this is a batch of multivariate action distributions, so that
        # the log probability computation will be correct
        assert len(dists.batch_shape) == 1
        log_probs = dists.log_prob(batch["actions"])

        predictions = OrderedDict(
            log_probs=log_probs,
        )
        return predictions


class BC_Bilevel(BC_RNN):
    def _create_networks(self):
        """
        Creates networks and places them into @self.nets.
        """
        self.robot = self.algo_config.implicit.robot
        self.nets = nn.ModuleDict()

        # ### LMP style - goal-conditioned output trajectory
        # self.nets["posterior"] = PolicyNets.Latent(**encoder_kwargs) #TODO encoder
        # self.nets["decoder"] = PolicyNets.Decoder(**decoder_kwargs) #TODO decoder

        ### TODO RSSM style - unconditioned dist over entire trajectory

        self.nets = self.nets.float().to(self.device)

    def _process_batch_for_training(self, batch):
        To = self.algo_config.horizon.observation_horizon # should be same as frame_stack
        Tp = self.algo_config.horizon.prediction_horizon # should be same as the entire seq size (seq_len+frame_stack1-1) 
        # TODO decide what the input frames should be -- initial and last(goal)?
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k][:, :To, :] for k in batch["obs"]}
        input_batch["goal_obs"] = batch.get("goal_obs", None) # goals may not be present
        input_batch["actions"] = batch["actions"][:, :Tp, :]

        # TODO check for action normalization

        return TensorUtils.to_device(TensorUtils.to_float(input_batch), self.device)

    def _prior(self):
        prior = D.MultivariateNormal()
        return prior

    def _posterior(self, batch):
        z = self.nets["posterior"](batch['obs'])
        posterior = D.MultivariateNormal(z['mean', z['std']])
        # TODO add to logs
        return posterior

    def _compute_loss(self):
        # Encode input obs and goal into a latent
        # Compute KL_div(posterior||prior)

        # Decode all obs and actions
        # Compute NLL loss

        # Make sure everything is pushed to logs

        pass

    def log_info(self, info):
        pass

    def get_action(self, obs_dict, goal_dict=None):
        pass


# # requires diffusers==0.11.1
# from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
# from diffusers.schedulers.scheduling_ddim import DDIMScheduler

class LatentDiffusion(BC_RNN): #TODO change base class
    def __create_networks__(self):
        # setup noise scheduler
        noise_scheduler = None
        if self.algo_config.ddpm.enabled:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=self.algo_config.ddpm.num_train_timesteps,
                beta_schedule=self.algo_config.ddpm.beta_schedule,
                clip_sample=self.algo_config.ddpm.clip_sample,
                prediction_type=self.algo_config.ddpm.prediction_type
            )
        elif self.algo_config.ddim.enabled:
            noise_scheduler = DDIMScheduler(
                num_train_timesteps=self.algo_config.ddim.num_train_timesteps,
                beta_schedule=self.algo_config.ddim.beta_schedule,
                clip_sample=self.algo_config.ddim.clip_sample,
                set_alpha_to_one=self.algo_config.ddim.set_alpha_to_one,
                steps_offset=self.algo_config.ddim.steps_offset,
                prediction_type=self.algo_config.ddim.prediction_type
            )
        else:
            raise RuntimeError()
        self.noise_scheduler = noise_scheduler

        self.nets["noise_pred"] = PolicyNets.NoisePred() # TODO
        
    def noise_latents(self, latents):
        batch_size = latents.shape[0]

        # sample noise to add to latents
        noise = torch.randn(latents.shape, device=self.device)
        
        # sample a diffusion iteration for each data point
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (batch_size,), device=self.device
        ).long()

        # (forward diffusion) add noise to the clean latents according to the noise magnitude at each diffusion iteration
        noisy_latents = self.noise_scheduler.add_noise(
            latents, noise, timesteps)
        
        return noisy_latents, noise
    
    def _compute_losses(self, batch):
        noisy_latents, noise = self.noise_latents(batch['latents'])
        noise_pred = self.nets["noise_pred"](batch)
        loss = nn.MSELoss()(noise_pred, noise)
        # TODO ablate noise_pred, latent_pred, v_pred

        # TODO add to logs
        return loss

