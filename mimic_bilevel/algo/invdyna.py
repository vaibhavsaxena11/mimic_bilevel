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

import robomimic.models.base_nets as BaseNets
from robomimic.algo.bc import BC_RNN, BC
import robomimic.models.policy_nets as PolicyNets
import mimic_bilevel.models.policy_nets as PolicyNets_Bilevel
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.loss_utils as LossUtils
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.torch_utils as TorchUtils
from robomimic.models.obs_nets import MIMO_Transformer
# import robomimic.utils.geometry as geometry
from robomimic.algo import register_algo_factory_func


@register_algo_factory_func("invdyna")
def algo_config_to_class(algo_config):
    """
    Maps algo config to the algo class to instantiate, along with additional algo kwargs.

    Args:
        algo_config (Config instance): algo config

    Returns:
        algo_class: subclass of Algo
        algo_kwargs (dict): dictionary of additional kwargs to pass to algorithm
    """

    latent_diffusion_enabled = ("latent_diffusion" in algo_config and algo_config.latent_diffusion.enabled)

    if latent_diffusion_enabled:
        return DiffusionInverseDynamics, {}

    return InverseDynamics, {}

class InverseDynamics(BC_RNN):
    # TODO add config class if want to use this class

    def _create_networks(self):
        self.nets = nn.ModuleDict()

        self.nets["policy"] = PolicyNets_Bilevel.RNNActorInvdyna(
            obs_shapes=self.obs_shapes,
            goal_shapes=self.goal_shapes,
            ac_dim=self.ac_dim,
            mlp_layer_dims=self.algo_config.actor_layer_dims,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
            **BaseNets.rnn_args_from_config(self.algo_config.rnn),
        )

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
        # TODO use next_obs to add one additional obs to batch["obs"] and throw rest of the next_obs (i.e. so it doesn't input to encoder)
        input_batch = dict()
        input_batch["obs"] = {k: batch["obs"][k] for k in batch["obs"]}
        for k in input_batch["obs"]:
            # assert time dim is at least 2, so that we have at least one inferred action
            assert input_batch["obs"][k].shape[1] >= 2, input_batch["obs"][k].shape
        input_batch["goal_obs"] = batch.get("goal_obs", None)
        assert input_batch["goal_obs"] is None, input_batch["goal_obs"].keys() # don't need goals
        input_batch["actions"] = batch["actions"]
        # input_batch["actions"] = batch["actions"][:, :-1, :] # keeping all but last action
        assert input_batch["actions"].shape[-1] == self.ac_dim, self.ac_dim

        # we move to device first before float conversion because image observation modalities will be uint8 -
        # this minimizes the amount of data transferred to GPU
        return TensorUtils.to_float(TensorUtils.to_device(input_batch, self.device))
    
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
        predictions = OrderedDict()
        # TODO don't pass the obs at last t; subsequently, no need to ignore last timestep of predictions["recons"] in _compute_losses()
        outputs = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"], predictions["recons"] = outputs
        return predictions
    
    def _compute_losses(self, predictions, batch):
        """
        Internal helper function for BC algo class. Compute losses based on
        network outputs in @predictions dict, using reference labels in @batch.

        Args:
            predictions (dict): dictionary containing network outputs, from @_forward_training
            batch (dict): dictionary with torch.Tensors sampled
                from a data loader and filtered by @process_batch_for_training

        Returns:
            losses (dict): dictionary of losses computed over the batch
        """
        losses = OrderedDict()
        a_target = batch["actions"][:, :-1]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss

        recon_losses = [nn.MSELoss()(predictions["recons"][k][:, :-1], batch["obs"][k][:, 1:]) for k in batch["obs"]]
        losses["recon_loss"] = sum(recon_losses)

        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """
        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"]+losses["recon_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        log = super(InverseDynamics, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item() + info["losses"]["recon_loss"].item()
        if "recon_loss" in info["losses"]:
            log["Recon_Loss"] = info["losses"]["recon_loss"].item()
        return log

    def get_action(self, obs_dict, goal_dict=None):
        #  Call dynamics to get next state
        # probably not needed to reimplement as policy class takes care and only outputs action

        assert not self.nets.training

        obs = copy.deepcopy(obs_dict)
        _, next_obs, _ = self.nets["policy"].forward_step(obs, goal_dict=goal_dict, predict_action=False)
        obs = TensorUtils.unsqueeze(obs, 1)
        next_obs = TensorUtils.unsqueeze(next_obs, 1)
        for k in next_obs:
            obs[k] = torch.cat([obs[k], next_obs[k]], 1)
        action, _ = self.nets["policy"](obs, goal_dict=goal_dict, predict_action=True)

        return action[:, 0]


class DiffusionInverseDynamics(InverseDynamics):
    def _create_networks(self):
        self.nets = nn.ModuleDict()

        self.nets["policy"] = PolicyNets_Bilevel.DiffusionActorInvdyna(
            obs_shapes=self.obs_shapes,
            ac_dim=self.ac_dim,
            encoder_mlp_dims=self.algo_config.latent_diffusion.encoder_mlp_dims,
            denoiser_mlp_hidden_dims=self.algo_config.latent_diffusion.denoiser_mlp_hidden_dims,
            num_timesteps=self.algo_config.latent_diffusion.num_timesteps,
            timestep_dim=self.algo_config.latent_diffusion.timestep_dim,
            goal_shapes=self.goal_shapes,
            encoder_kwargs=ObsUtils.obs_encoder_kwargs_from_config(self.obs_config.encoder),
        )

        self.nets = self.nets.float().to(self.device)

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
        predictions = OrderedDict()
        outputs = self.nets["policy"](obs_dict=batch["obs"], goal_dict=batch["goal_obs"])
        predictions["actions"], predictions["latent_diffusion"], predictions["recons"] = outputs
        return predictions

    def _compute_losses(self, predictions, batch):
        losses = OrderedDict()
        a_target = batch["actions"][:, :-1]
        actions = predictions["actions"]
        losses["l2_loss"] = nn.MSELoss()(actions, a_target)
        losses["l1_loss"] = nn.SmoothL1Loss()(actions, a_target)
        # cosine direction loss on eef delta position
        losses["cos_loss"] = LossUtils.cosine_loss(actions[..., :3], a_target[..., :3])

        action_losses = [
            self.algo_config.loss.l2_weight * losses["l2_loss"],
            self.algo_config.loss.l1_weight * losses["l1_loss"],
            self.algo_config.loss.cos_weight * losses["cos_loss"],
        ]
        action_loss = sum(action_losses)
        losses["action_loss"] = action_loss

        # TODO test
        recon_losses = [nn.MSELoss()(predictions["recons"][k][:, :-1], batch["obs"][k][:, 1:]) for k in self.obs_shapes]
        losses["recon_loss"] = sum(recon_losses)

        # TODO test
        diffusion_loss = nn.MSELoss()(predictions["latent_diffusion"][0], predictions["latent_diffusion"][1])
        losses["diffusion_loss"] = diffusion_loss

        return losses

    def _train_step(self, losses):
        """
        Internal helper function for BC algo class. Perform backpropagation on the
        loss tensors in @losses to update networks.

        Args:
            losses (dict): dictionary of losses computed over the batch, from @_compute_losses
        """
        # NOTE: DONE
        # gradient step
        info = OrderedDict()
        policy_grad_norms = TorchUtils.backprop_for_loss(
            net=self.nets["policy"],
            optim=self.optimizers["policy"],
            loss=losses["action_loss"]+losses["diffusion_loss"]+losses["recon_loss"],
            max_grad_norm=self.global_config.train.max_grad_norm,
        )
        info["policy_grad_norms"] = policy_grad_norms
        return info

    def log_info(self, info):
        """
        Process info dictionary from @train_on_batch to summarize
        information to pass to tensorboard for logging.

        Args:
            info (dict): dictionary of info

        Returns:
            loss_log (dict): name -> summary statistic
        """
        # NOTE: DONE
        log = super(InverseDynamics, self).log_info(info)
        log["Loss"] = info["losses"]["action_loss"].item() + info["losses"]["diffusion_loss"].item() + info["losses"]["recon_loss"].item()
        if "recon_loss" in info["losses"]:
            log["Recon_Loss"] = info["losses"]["recon_loss"].item()
        if "diffusion_loss" in info["losses"]:
            log["Diffusion_Loss"] = info["losses"]["diffusion_loss"].item()
        return log

    def get_action(self, obs_dict, goal_dict=None):
        """
        conditioned-generation (infilling) followed by inverse-dynamics
        """
        mod = ObsUtils.OBS_MODALITIES_TO_KEYS["rgb"][0]
        T = obs_dict[mod].shape[1]

        mask = torch.zeros([T,]).to(self.device)
        mask[0] = 1
        mask[-1] = 1
        mask_idxs = [0, -1]
        recons = copy.deepcopy(obs_dict)
        for t in reversed(range(self.algo_config.latent_diffusion.num_timesteps)):
            actions, _, recons = self.nets["policy"](obs_dict, goal_dict, timesteps=[t,])
            for k in recons:
                # recons[k] = obs_dict[k]*mask + recons[k]*(1-mask)
                for t in range(len(mask)):
                    recons[k][:,t] = obs_dict[k][:,t]*mask[t] + recons[k][:,t]*(1-mask[t])

        return actions
