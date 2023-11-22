from collections import OrderedDict

import torch
import torch.nn as nn

from robomimic.models.base_nets import Module, Sequential, MLP
from robomimic.models.obs_nets import MIMO_Transformer
import robomimic.utils.tensor_utils as TensorUtils
from mimic_bilevel.models.obs_nets import RNN_MIMO_MultiMod, MultiModalityObservationDecoder

# class LDM(MIMO_Transformer):
#     def _create_networks(self):
#         # TODO
#         pass

# class Dynamics(Module):
#     def __init__(self, **kwargs):
#         self.nets = nn.ModuleDict()
#         self.nets[""]

#     def output_shape(self):
#         # TODO
#         pass

#     def forward(self, **inputs):
#         pass

class RNNActorInvdyna(RNN_MIMO_MultiMod):
    """
    An RNN policy network that predicts next_obs from obs, and actions using learned inverse dynamics.
    """
    def __init__(
        self,
        obs_shapes,
        ac_dim,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        goal_shapes=None,
        encoder_kwargs=None,
    ):
        """
        Args:
            obs_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for observations.

            ac_dim (int): dimension of action space.

            mlp_layer_dims ([int]): sequence of integers for the MLP hidden layers sizes.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the torch.nn.LSTM / GRU

            goal_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for goal observations.

            encoder_kwargs (dict or None): If None, results in default encoder_kwargs being applied. Otherwise, should
                be nested dictionary containing relevant per-modality information for encoder networks.
                Should be of form:

                obs_modality1: dict
                    feature_dimension: int
                    core_class: str
                    core_kwargs: dict
                        ...
                        ...
                    obs_randomizer_class: str
                    obs_randomizer_kwargs: dict
                        ...
                        ...
                obs_modality2: dict
                    ...
        """
        self.ac_dim = ac_dim

        assert isinstance(obs_shapes, OrderedDict)
        self.obs_shapes = obs_shapes
        action_shapes = OrderedDict({"action": [self.ac_dim,]})

        # set up different observation groups for @RNN_MIMO_MultiMod
        observation_group_shapes = OrderedDict()
        observation_group_shapes["obs"] = OrderedDict(self.obs_shapes)

        self._is_goal_conditioned = False
        if goal_shapes is not None and len(goal_shapes) > 0:
            assert isinstance(goal_shapes, OrderedDict)
            self._is_goal_conditioned = True
            self.goal_shapes = OrderedDict(goal_shapes)
            observation_group_shapes["goal"] = OrderedDict(self.goal_shapes)
        else:
            self.goal_shapes = OrderedDict()

        output_shapes = self._get_rnn_output_shapes()
        super(RNNActorInvdyna, self).__init__(
            input_obs_group_shapes=observation_group_shapes,
            output_shapes=output_shapes,
            mlp_layer_dims=mlp_layer_dims,
            mlp_activation=nn.ReLU,
            mlp_layer_func=nn.Linear,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            rnn_kwargs=rnn_kwargs,
            per_step=False, # True
            encoder_kwargs=encoder_kwargs,
        )

        if self._has_mlp:
            self.nets["action_decoder"] = Sequential(
                MLP(
                    input_dim=2*self.rnn_output_dim,
                    output_dim=mlp_layer_dims[-1],
                    layer_dims=mlp_layer_dims[:-1],
                    output_activation=nn.ReLU,
                    layer_func=nn.Linear
                ),
                MultiModalityObservationDecoder(
                    decode_shapes=action_shapes,
                    input_feat_dim=mlp_layer_dims[-1],
                )
            )
        else:
            self.nets["action_decoder"] = MultiModalityObservationDecoder(
                decode_shapes=action_shapes,
                input_feat_dim=2*self.rnn_output_dim,
            )

    def _get_rnn_output_shapes(self):
        return self.obs_shapes

    # def _get_output_shapes(self):
    #     """
    #     Allow subclasses to re-define outputs from @RNN_MIMO_MLP, since we won't
    #     always directly predict actions, but may instead predict the parameters
    #     of a action distribution.
    #     """
    #     return OrderedDict(action=(self.ac_dim,))

    def output_shape(self, input_shape):
        # note: @input_shape should be dictionary (key: mod)
        # infers temporal dimension from input shape
        mod = list(self.obs_shapes.keys())[0]
        T = input_shape[mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="RNNActorInvdyna: input_shape inconsistent in temporal dimension")
        return [T, self.ac_dim]

    def forward(self, obs_dict, goal_dict=None, rnn_init_state=None, return_state=False, predict_action=True):
        """
        Forward a sequence of inputs through the RNN and the per-step network.

        Args:
            obs_dict (dict): batch of observations - each tensor in the dictionary
                should have leading dimensions batch and time [B, T, ...]
            goal_dict (dict): if not None, batch of goal observations
            rnn_init_state: rnn hidden state, initialize to zero state if set to None
            return_state (bool): whether to return hidden state

        Returns:
            actions (torch.Tensor): predicted action sequence
            rnn_state: return rnn state at the end if return_state is set to True
        """
        if self._is_goal_conditioned:
            assert goal_dict is not None
            # repeat the goal observation in time to match dimension with obs_dict
            mod = list(obs_dict.keys())[0]
            goal_dict = TensorUtils.unsqueeze_expand_at(goal_dict, size=obs_dict[mod].shape[1], dim=1)

        outputs = super(RNNActorInvdyna, self).forward(
            obs=obs_dict, goal=goal_dict, rnn_init_state=rnn_init_state, return_state=return_state)

        if return_state:
            recons, rnn_outputs, rnn_state = outputs
        else:
            recons, rnn_outputs = outputs
            rnn_state = None

        actions = None
        if predict_action:
            # use rnn_outputs to reverse-predict actions # TODO use rnn-outputs instead of rnn-rnn_outputs? matters for LSTM but not GRU
            assert rnn_outputs.ndim == 3, rnn_outputs.shape # [B, T, D]
            next_rnn_outputs = torch.roll(rnn_outputs, shifts=-1, dims=1)
            # TODO stop grad on rnn rnn_outputs?
            actions = TensorUtils.time_distributed(torch.cat([rnn_outputs[:, :-1], next_rnn_outputs[:, :-1]], -1), self.nets["action_decoder"])
            # apply tanh squashing to ensure actions are in [-1, 1]
            actions = torch.tanh(actions["action"])

        if return_state:
            return actions, recons, rnn_state
        else:
            return actions, recons

    def forward_step(self, obs_dict, goal_dict=None, rnn_state=None, predict_action=True):
        """
        Unroll RNN over single timestep to get actions.

        Args:
            obs_dict (dict): batch of observations. Should not contain
                time dimension.
            goal_dict (dict): if not None, batch of goal observations
            rnn_state: rnn hidden state, initialize to zero state if set to None

        Returns:
            actions (torch.Tensor): batch of actions - does not contain time dimension
            state: updated rnn state
        """
        obs_dict = TensorUtils.to_sequence(obs_dict)
        action, obs, state = self.forward(
            obs_dict, goal_dict, rnn_init_state=rnn_state, return_state=True, predict_action=predict_action)
        if predict_action:
            return action[:, 0], TensorUtils.index_at_time(obs, 0), state
        return None, TensorUtils.index_at_time(obs, 0), state

    def _to_string(self):
        """Info to pretty print."""
        return "action_dim={}".format(self.ac_dim)
