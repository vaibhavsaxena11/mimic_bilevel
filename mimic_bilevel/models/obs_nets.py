import sys
import numpy as np
import textwrap
from copy import deepcopy
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from robomimic.utils.python_utils import extract_class_init_kwargs_from_dict
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
from robomimic.models.base_nets import Module, Sequential, MLP, RNN_Base, ResNet18Conv, SpatialSoftmax, FeatureAggregator
from robomimic.models.obs_nets import MIMO_MLP, RNN_MIMO_MLP, ObservationDecoder, ObservationGroupEncoder
from robomimic.models.obs_core import VisualCore, Randomizer
from robomimic.models.transformers import PositionalEncoding, GPT_Backbone


class MultiModalityObservationDecoder(ObservationDecoder):

    def _create_layers(self):
        self.nets = nn.ModuleDict()
        for k in self.obs_shapes:
            mod = ObsUtils.OBS_KEYS_TO_MODALITIES[k]
            # TODO add more modalities as needed
            if mod == "rgb":
                # TODO use deconv image decoder
                layer_out_dim = int(np.prod(self.obs_shapes[k]))
                self.nets[k] = nn.Sequential(
                    nn.Linear(self.input_feat_dim, layer_out_dim),
                    nn.Unflatten(-1, self.obs_shapes[k])
                )
            else:
                layer_out_dim = int(np.prod(self.obs_shapes[k]))
                self.nets[k] = nn.Linear(self.input_feat_dim, layer_out_dim)

    def forward(self, feats):
        """
        Predict each modality from input features, and reshape to each modality's shape.
        """
        output = {}
        for k in self.obs_shapes:
            output[k] = self.nets[k](feats)
        return output


class RNN_MIMO_MultiMod(Module):
    """
    Structure: [encoder(multi mod) -> rnn -> mlp -> decoder(multi mod)]
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        mlp_layer_dims,
        rnn_hidden_dim,
        rnn_num_layers,
        rnn_type="LSTM",  # [LSTM, GRU]
        rnn_kwargs=None,
        mlp_activation=nn.ReLU,
        mlp_layer_func=nn.Linear,
        per_step=False, # TODO(VS) remove support for per_step as it should always be False
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.

            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.

            rnn_hidden_dim (int): RNN hidden dimension

            rnn_num_layers (int): number of RNN layers

            rnn_type (str): [LSTM, GRU]

            rnn_kwargs (dict): kwargs for the rnn model

            per_step (bool): if True, apply the MLP and observation decoder into @output_shapes
                at every step of the RNN. Otherwise, apply them to the final hidden state of the 
                RNN.

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
        # super(RNN_MIMO_Invdyna, self).__init__(*args, **kwargs)
        # self.nets["decoder"] = ImageDecoder(
        #     decode_shapes=self.output_shapes,
        #     input_feat_dim=rnn_output_dim,
        # )

        assert not per_step, "per_step should be false as we need intermediate outputs at all timesteps"

        super(RNN_MIMO_MultiMod, self).__init__()
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.per_step = per_step

        # separating action from state output shapes
        # action_keys = ["actions"] # TODO get this from config.train.action_keys
        # self.obs_output_shapes = dict([(k,v) for k, v in self.output_shapes.items() if k not in action_keys])
        # self.action_output_shapes = dict([(k,v) for k, v in self.output_shapes.items() if k in action_keys])
        # assert set(self.action_output_shapes.keys()).issubset(set(action_keys)) # TODO remove after testing

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )

        # flat encoder output dimension
        rnn_input_dim = self.nets["encoder"].output_shape()[0]

        # bidirectional RNNs mean that the output of RNN will be twice the hidden dimension
        rnn_is_bidirectional = rnn_kwargs.get("bidirectional", False)
        num_directions = int(rnn_is_bidirectional) + 1 # 2 if bidirectional, 1 otherwise
        rnn_output_dim = num_directions * rnn_hidden_dim
        self.rnn_output_dim = rnn_output_dim

        per_step_net = None
        # NOTE(VS) per_step is True by default in the code
        self._has_mlp = (len(mlp_layer_dims) > 0)
        if self._has_mlp:
            self.nets["mlp"] = MLP(
                input_dim=rnn_output_dim,
                output_dim=mlp_layer_dims[-1],
                layer_dims=mlp_layer_dims[:-1],
                output_activation=mlp_activation,
                layer_func=mlp_layer_func
            )
            self.nets["obs_decoder"] = MultiModalityObservationDecoder(
                # decode_shapes=self.obs_output_shapes,
                decode_shapes=self.output_shapes,
                input_feat_dim=mlp_layer_dims[-1],
            )
            if self.per_step:
                per_step_net = Sequential(self.nets["mlp"], self.nets["obs_decoder"]) # TODO(VS) remove
        else:
            self.nets["obs_decoder"] = MultiModalityObservationDecoder(
                # decode_shapes=self.obs_output_shapes,
                decode_shapes=self.output_shapes,
                input_feat_dim=rnn_output_dim,
            )
            if self.per_step:
                per_step_net = self.nets["obs_decoder"] # TODO(VS) remove

        # self.nets["action_decoder"] = MultiModalityObservationDecoder(
        #     decode_shapes=self.action_output_shapes,
        #     input_feat_dim=2*rnn_output_dim,
        # )

        # core network
        self.nets["rnn"] = RNN_Base(
            input_dim=rnn_input_dim,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_type=rnn_type,
            per_step_net=per_step_net, # TODO(VS) remove
            rnn_kwargs=rnn_kwargs
        )

    def get_rnn_init_state(self, batch_size, device):
        """
        Get a default RNN state (zeros)

        Args:
            batch_size (int): batch size dimension

            device: device the hidden state should be sent to.

        Returns:
            hidden_state (torch.Tensor or tuple): returns hidden state tensor or tuple of hidden state tensors
                depending on the RNN type
        """
        # NOTE(VS) unchanged
        return self.nets["rnn"].get_rnn_init_state(batch_size, device=device)

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.

        Args:
            input_shape (dict): dictionary of dictionaries, where each top-level key
                corresponds to an observation group, and the low-level dictionaries
                specify the shape for each modality in an observation dictionary
        """
        # NOTE(VS) unchanged

        # infers temporal dimension from input shape
        obs_group = list(self.input_obs_group_shapes.keys())[0]
        mod = list(self.input_obs_group_shapes[obs_group].keys())[0]
        T = input_shape[obs_group][mod][0]
        TensorUtils.assert_size_at_dim(input_shape, size=T, dim=0, 
                msg="RNN_MIMO_MLP: input_shape inconsistent in temporal dimension")
        # returns a dictionary instead of list since outputs are dictionaries
        return { k : [T] + list(self.output_shapes[k]) for k in self.output_shapes }

    def forward(self, rnn_init_state=None, return_state=False, **inputs):
        """
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.

            rnn_init_state: rnn hidden state, initialize to zero state if set to None

            return_state (bool): whether to return hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.

            rnn_state (torch.Tensor or tuple): return the new rnn state (if @return_state)
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        # use encoder to extract flat rnn inputs
        rnn_inputs = TensorUtils.time_distributed(inputs, self.nets["encoder"], inputs_as_kwargs=True)
        assert rnn_inputs.ndim == 3  # [B, T, D]
        if self.per_step: # TODO(VS) remove as we need to return rnn_outputs and not just decoded ones
            return self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
        
        rnn_outputs = self.nets["rnn"].forward(inputs=rnn_inputs, rnn_init_state=rnn_init_state, return_state=return_state)
        if return_state:
            rnn_outputs, rnn_state = rnn_outputs
        assert rnn_outputs.ndim == 3 # [B, T, D]

        # # apply MLP + decoder to last RNN output
        # if self._has_mlp:
        #     outputs = self.nets["obs_decoder"](self.nets["mlp"](outputs[:, -1]))
        # else:
        #     outputs = self.nets["obs_decoder"](outputs[:, -1])

        # apply MLP + decoder to all rnn outputs
        if self._has_mlp:
            outputs = TensorUtils.time_distributed(rnn_outputs, Sequential(self.nets["mlp"], self.nets["obs_decoder"]))
        else:
            outputs = TensorUtils.time_distributed(rnn_outputs, self.nets["obs_decoder"])

        if return_state:
            return outputs, rnn_outputs, rnn_state
        return outputs, rnn_outputs

    def forward_step(self, rnn_state, **inputs):
        """
        Unroll network over a single timestep.

        Args:
            inputs (dict): expects same modalities as @self.input_shapes, with
                additional batch dimension (but NOT time), since this is a 
                single time step.

            rnn_state (torch.Tensor): rnn hidden state

        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Does not contain time dimension.

            rnn_state: return the new rnn state
        """
        # ensure that the only extra dimension is batch dim, not temporal dim 
        assert np.all([inputs[k].ndim - 1 == len(self.input_shapes[k]) for k in self.input_shapes])

        inputs = TensorUtils.to_sequence(inputs)
        outputs, rnn_state = self.forward(
            inputs, 
            rnn_init_state=rnn_state,
            return_state=True,
        )
        # NOTE(VS) if per_step==True, outputs at all timesteps are passed through decoder
        # NOTE(VS) otherwise, only the last output has been passed through decoder -- and time dim removed
        if self.per_step:
            # if outputs are not per-step, the time dimension is already reduced
            outputs = outputs[:, 0]
        return outputs, rnn_state

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        # NOTE(VS) unchanged
        return ''

    def __repr__(self):
        """Pretty print network."""
        # NOTE(VS) unchanged
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        msg += textwrap.indent("\n" + self._to_string(), indent)
        msg += textwrap.indent("\n\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\nrnn={}".format(self.nets["rnn"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


class MIMO_Transformer(Module):
    """
    Extension to Transformer (based on GPT architecture) to accept multiple observation 
    dictionaries as input and to output dictionaries of tensors. Inputs are specified as 
    a dictionary of observation dictionaries, with each key corresponding to an observation group.
    This module utilizes @ObservationGroupEncoder to process the multiple input dictionaries and
    @ObservationDecoder to generate tensor dictionaries. The default behavior
    for encoding the inputs is to process visual inputs with a learned CNN and concatenating
    the flat encodings with the other flat inputs. The default behavior for generating 
    outputs is to use a linear layer branch to produce each modality separately
    (including visual outputs).
    """
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        transformer_embed_dim,
        transformer_num_layers,
        transformer_num_heads,
        transformer_context_length,
        transformer_emb_dropout=0.1,
        transformer_attn_dropout=0.1,
        transformer_block_output_dropout=0.1,
        transformer_sinusoidal_embedding=False,
        transformer_activation="gelu",
        transformer_nn_parameter_for_timesteps=False,
        encoder_kwargs=None,
    ):
        """
        Args:
            input_obs_group_shapes (OrderedDict): a dictionary of dictionaries.
                Each key in this dictionary should specify an observation group, and
                the value should be an OrderedDict that maps modalities to
                expected shapes.
            output_shapes (OrderedDict): a dictionary that maps modality to
                expected shapes for outputs.
            transformer_embed_dim (int): dimension for embeddings used by transformer
            transformer_num_layers (int): number of transformer blocks to stack
            transformer_num_heads (int): number of attention heads for each
                transformer block - must divide @transformer_embed_dim evenly. Self-attention is 
                computed over this many partitions of the embedding dimension separately.
            transformer_context_length (int): expected length of input sequences
            transformer_activation: non-linearity for input and output layers used in transformer
            transformer_emb_dropout (float): dropout probability for embedding inputs in transformer
            transformer_attn_dropout (float): dropout probability for attention outputs for each transformer block
            transformer_block_output_dropout (float): dropout probability for final outputs for each transformer block
            encoder_kwargs (dict): observation encoder config
        """
        super(MIMO_Transformer, self).__init__()
        
        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)

        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes

        self.nets = nn.ModuleDict()
        self.params = nn.ParameterDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
            feature_activation=None,
        )

        # flat encoder output dimension
        transformer_input_dim = self.nets["encoder"].output_shape()[0]

        self.nets["embed_encoder"] = nn.Linear(
            transformer_input_dim, transformer_embed_dim
        )

        max_timestep = transformer_context_length

        if transformer_sinusoidal_embedding:
            self.nets["embed_timestep"] = PositionalEncoding(transformer_embed_dim)
        elif transformer_nn_parameter_for_timesteps:
            assert (
                not transformer_sinusoidal_embedding
            ), "nn.Parameter only works with learned embeddings"
            self.params["embed_timestep"] = nn.Parameter(
                torch.zeros(1, max_timestep, transformer_embed_dim)
            )
        else:
            self.nets["embed_timestep"] = nn.Embedding(max_timestep, transformer_embed_dim)

        # layer norm for embeddings
        self.nets["embed_ln"] = nn.LayerNorm(transformer_embed_dim)
        
        # dropout for input embeddings
        self.nets["embed_drop"] = nn.Dropout(transformer_emb_dropout)

        # GPT transformer
        self.nets["transformer"] = GPT_Backbone(
            embed_dim=transformer_embed_dim,
            num_layers=transformer_num_layers,
            num_heads=transformer_num_heads,
            context_length=transformer_context_length,
            attn_dropout=transformer_attn_dropout,
            block_output_dropout=transformer_block_output_dropout,
            activation=transformer_activation,
        )

        # decoder for output modalities
        self.nets["decoder"] = ObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=transformer_embed_dim,
        )

        self.transformer_context_length = transformer_context_length
        self.transformer_embed_dim = transformer_embed_dim
        self.transformer_sinusoidal_embedding = transformer_sinusoidal_embedding
        self.transformer_nn_parameter_for_timesteps = transformer_nn_parameter_for_timesteps

    def output_shape(self, input_shape=None):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.
        """
        return { k : list(self.output_shapes[k]) for k in self.output_shapes }

    def embed_timesteps(self, embeddings):
        """
        Computes timestep-based embeddings (aka positional embeddings) to add to embeddings.
        Args:
            embeddings (torch.Tensor): embeddings prior to positional embeddings are computed
        Returns:
            time_embeddings (torch.Tensor): positional embeddings to add to embeddings
        """
        timesteps = (
            torch.arange(
                0,
                embeddings.shape[1],
                dtype=embeddings.dtype,
                device=embeddings.device,
            )
            .unsqueeze(0)
            .repeat(embeddings.shape[0], 1)
        )
        assert (timesteps >= 0.0).all(), "timesteps must be positive!"
        if self.transformer_sinusoidal_embedding:
            assert torch.is_floating_point(timesteps), timesteps.dtype
        else:
            timesteps = timesteps.long()

        if self.transformer_nn_parameter_for_timesteps:
            time_embeddings = self.params["embed_timestep"]
        else:
            time_embeddings = self.nets["embed_timestep"](
                timesteps
            )  # these are NOT fed into transformer, only added to the inputs.
            # compute how many modalities were combined into embeddings, replicate time embeddings that many times
            num_replicates = embeddings.shape[-1] // self.transformer_embed_dim
            time_embeddings = torch.cat([time_embeddings for _ in range(num_replicates)], -1)
            assert (
                embeddings.shape == time_embeddings.shape
            ), f"{embeddings.shape}, {time_embeddings.shape}"
        return time_embeddings

    def input_embedding(
        self,
        inputs,
    ):
        """
        Process encoded observations into embeddings to pass to transformer,
        Adds timestep-based embeddings (aka positional embeddings) to inputs.
        Args:
            inputs (torch.Tensor): outputs from observation encoder
        Returns:
            embeddings (torch.Tensor): input embeddings to pass to transformer backbone.
        """
        embeddings = self.nets["embed_encoder"](inputs)
        time_embeddings = self.embed_timesteps(embeddings)
        embeddings = embeddings + time_embeddings
        embeddings = self.nets["embed_ln"](embeddings)
        embeddings = self.nets["embed_drop"](embeddings)

        return embeddings

    
    def forward(self, **inputs):
        """
        Process each set of inputs in its own observation group.
        Args:
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.
        Returns:
            outputs (dict): dictionary of output torch.Tensors, that corresponds
                to @self.output_shapes. Leading dimensions will be batch and time [B, T, ...]
                for each tensor.
        """
        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                if inputs[obs_group][k] is None:
                    continue
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        inputs = inputs.copy()

        transformer_encoder_outputs = None
        transformer_inputs = TensorUtils.time_distributed(
            inputs, self.nets["encoder"], inputs_as_kwargs=True
        )
        assert transformer_inputs.ndim == 3  # [B, T, D]

        if transformer_encoder_outputs is None:
            transformer_embeddings = self.input_embedding(transformer_inputs)
            # pass encoded sequences through transformer
            transformer_encoder_outputs = self.nets["transformer"].forward(transformer_embeddings)

        transformer_outputs = transformer_encoder_outputs
        # apply decoder to each timestep of sequence to get a dictionary of outputs
        transformer_outputs = TensorUtils.time_distributed(
            transformer_outputs, self.nets["decoder"]
        )
        transformer_outputs["transformer_encoder_outputs"] = transformer_encoder_outputs
        return transformer_outputs

    def _to_string(self):
        """
        Subclasses should override this method to print out info about network / policy.
        """
        return ''

    def __repr__(self):
        """Pretty print network."""
        header = '{}'.format(str(self.__class__.__name__))
        msg = ''
        indent = ' ' * 4
        if self._to_string() != '':
            msg += textwrap.indent("\n" + self._to_string() + "\n", indent)
        msg += textwrap.indent("\nencoder={}".format(self.nets["encoder"]), indent)
        msg += textwrap.indent("\n\ntransformer={}".format(self.nets["transformer"]), indent)
        msg += textwrap.indent("\n\ndecoder={}".format(self.nets["decoder"]), indent)
        msg = header + '(' + msg + '\n)'
        return msg


################################
#### Latent Diffusion Model ####
################################

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        assert x.shape[-1] == 1, x.shape
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x * emb
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Diffusion_MIMO_MultiMod(Module):
    def __init__(
        self,
        input_obs_group_shapes,
        output_shapes,
        encoder_mlp_dims,
        denoiser_mlp_hidden_dims,
        num_timesteps,
        timestep_dim,
        encoder_kwargs=None
    ):
        super(Diffusion_MIMO_MultiMod, self).__init__()

        assert isinstance(input_obs_group_shapes, OrderedDict)
        assert np.all([isinstance(input_obs_group_shapes[k], OrderedDict) for k in input_obs_group_shapes])
        assert isinstance(output_shapes, OrderedDict)
        self.input_obs_group_shapes = input_obs_group_shapes
        self.output_shapes = output_shapes
        self.num_timesteps = num_timesteps

        self.nets = nn.ModuleDict()

        # Encoder for all observation groups.
        self.nets["encoder"] = ObservationGroupEncoder(
            observation_group_shapes=input_obs_group_shapes,
            encoder_kwargs=encoder_kwargs,
        )
        self.nets["encoder_mlp"] = MLP(
            input_dim=self.nets["encoder"].output_shape()[0],
            output_dim=encoder_mlp_dims[-1],
            layer_dims=encoder_mlp_dims[:-1],
            output_activation=nn.Identity,
            layer_func=nn.Linear
        )

        # flat encoder output dimension
        # self.latent_dim = self.nets["encoder"].output_shape()[0]
        self.latent_dim = encoder_mlp_dims[-1]
        
        # Latent denoiser
        assert timestep_dim % 2 == 0
        self.timestep_encoder = SinusoidalPosEmb(timestep_dim)
        self.nets["denoiser"] = MLP(
            input_dim=self.latent_dim+timestep_dim, # TODO(VS) also input timestep_dim
            output_dim=self.latent_dim,
            layer_dims=denoiser_mlp_hidden_dims,
            output_activation=nn.Identity,
            layer_func=nn.Linear
        )

        self.nets["obs_decoder"] = MultiModalityObservationDecoder(
            decode_shapes=self.output_shapes,
            input_feat_dim=self.latent_dim,
        )

    def output_shape(self, input_shape):
        """
        Returns output shape for this module, which is a dictionary instead
        of a list since outputs are dictionaries.

        Args:
            input_shape (dict): dictionary of dictionaries, where each top-level key
                corresponds to an observation group, and the low-level dictionaries
                specify the shape for each modality in an observation dictionary
        """
        return OrderedDict(noise=self.latent_dim, noise_pred=self.latent_dim, **input_shape)

    def forward(self, timesteps=None, **inputs):
        """
        Args:
            timesteps: TODO
            inputs (dict): a dictionary of dictionaries with one dictionary per
                observation group. Each observation group's dictionary should map
                modality to torch.Tensor batches. Should be consistent with
                @self.input_obs_group_shapes. First two leading dimensions should
                be batch and time [B, T, ...] for each tensor.

        Returns:
            latents: TODO
            (noise, noise_pred): TODO
            decoded_obs: TODO
        """
        # input a dict of observations
        # output: latent from the encoder
        # output: decoded (reconstructed) observations for all timesteps

        for obs_group in self.input_obs_group_shapes:
            for k in self.input_obs_group_shapes[obs_group]:
                # first two dimensions should be [B, T] for inputs
                assert inputs[obs_group][k].ndim - 2 == len(self.input_obs_group_shapes[obs_group][k])

        # use encoder to extract flat inputs
        feats = TensorUtils.time_distributed(inputs, self.nets["encoder"], inputs_as_kwargs=True)
        latents = self.nets["encoder_mlp"](feats)
        assert latents.ndim == 3  # [B, T, D]

        # (latent diffusion model) iteratively add noise to latents (only on T's not masked) (TODO)
        # and denoise to obtain new latents
        # subsequently, decode latents to generate observations and return

        if timesteps == None: # train to denoise all timesteps
            # timestep_wrong_actions = torch.rand([batch_size, num_wrong_exs, 1]).to(self.device)
            # timestep_wrong_actions = torch.tensor(np.random.choice(ts, [batch_size, num_wrong_exs, 1]).astype(np.float32)).to(self.device)
            B, T = latents.shape[:2]
            ts = np.array([t for t in range(self.num_timesteps)])
            timesteps = torch.tensor(np.random.choice(ts, [B, T, 1]).astype(np.float32)).to(latents.device)
        else:
            timesteps = torch.tensor(timesteps).to(latents.device)
            assert timesteps.ndim == 1 and timesteps.shape[0] == 1
            timesteps = torch.unsqueeze(torch.unsqueeze(timesteps, 0), 0)
            B, T = latents.shape[:2]
            timesteps = torch.tile(timesteps, (B,T,1))
        assert timesteps.ndim == 3  # [B, T, D]
        beta = 0.002 + (0.1 * timesteps / self.num_timesteps)
        noise = torch.randn(list(latents.shape))
        noisy_latents = latents + (beta**0.5)*noise
        noise_pred = self.nets["denoiser"](torch.cat([noisy_latents, self.timestep_encoder(timesteps)], -1))

        decoded_obs = self.nets["obs_decoder"](latents)

        return latents, (noise, noise_pred), decoded_obs
