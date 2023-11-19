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
from robomimic.models.base_nets import Module, Sequential, MLP, RNN_Base, ResNet18Conv, SpatialSoftmax, \
    FeatureAggregator
from robomimic.models.obs_core import VisualCore, Randomizer
from robomimic.models.transformers import PositionalEncoding, GPT_Backbone


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