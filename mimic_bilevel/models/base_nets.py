import torch
import torch.nn as nn

from robomimic.models.base_nets import ConvBase, Module, Sequential, MLP

# class UNetConv(ConvBase):
#     def __init__(
#         self,
#         input_channel=3,
#         pretrained=False, #TODO(VS)
#         init_features=32,
#         recon_enabled=False,
#     ):
#         """
#         Architecture from: https://github.com/mateuszbuda/brain-segmentation-pytorch
#         """
#         super(UNetConv, self).__init__()
#         self._input_channel = input_channel
#         self._output_channel = input_channel
#         self.init_features = init_features
#         self.recon_enabled = recon_enabled

#         # self.net = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=input_channel, out_channels=input_channel, init_features=self.init_features, pretrained=pretrained)
#         self.net = self._create_network()

#     def _create_network(self):
#         features = self.init_features
#         in_channels = self._input_channel
#         out_channels = self._output_channel

#         self.encoder1 = UNetConv._block(in_channels, features, name="enc1")
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder2 = UNetConv._block(features, features * 2, name="enc2")
#         self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder3 = UNetConv._block(features * 2, features * 4, name="enc3")
#         self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.encoder4 = UNetConv._block(features * 4, features * 8, name="enc4")
#         self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

#         self.bottleneck = UNetConv._block(features * 8, features * 16, name="bottleneck")

#         self.upconv4 = nn.ConvTranspose2d(
#             features * 16, features * 8, kernel_size=2, stride=2
#         )
#         self.decoder4 = UNetConv._block((features * 8) * 2, features * 8, name="dec4")
#         self.upconv3 = nn.ConvTranspose2d(
#             features * 8, features * 4, kernel_size=2, stride=2
#         )
#         self.decoder3 = UNetConv._block((features * 4) * 2, features * 4, name="dec3")
#         self.upconv2 = nn.ConvTranspose2d(
#             features * 4, features * 2, kernel_size=2, stride=2
#         )
#         self.decoder2 = UNetConv._block((features * 2) * 2, features * 2, name="dec2")
#         self.upconv1 = nn.ConvTranspose2d(
#             features * 2, features, kernel_size=2, stride=2
#         )
#         self.decoder1 = UNetConv._block(features * 2, features, name="dec1")

#         self.conv = nn.Conv2d(
#             in_channels=features, out_channels=out_channels, kernel_size=1
#         )

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict(
#                 [
#                     (
#                         name + "conv1",
#                         nn.Conv2d(
#                             in_channels=in_channels,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     # (name + "norm1", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu1", nn.ReLU(inplace=True)),
#                     (
#                         name + "conv2",
#                         nn.Conv2d(
#                             in_channels=features,
#                             out_channels=features,
#                             kernel_size=3,
#                             padding=1,
#                             bias=False,
#                         ),
#                     ),
#                     # (name + "norm2", nn.BatchNorm2d(num_features=features)),
#                     (name + "relu2", nn.ReLU(inplace=True)),
#                 ]
#             )
#         )

#     def output_shape(self, input_shape):
#         """
#         Function to compute output shape from inputs to this module. 

#         Args:
#             input_shape (iterable of int): shape of input. Does not include batch dimension.
#                 Some modules may not need this argument, if their output does not depend 
#                 on the size of the input, or if they assume fixed size input.

#         Returns:
#             out_shape ([int]): list of integers corresponding to output shape
#         """
#         assert(len(input_shape) == 3)
#         out_h = int(math.ceil(input_shape[1] / 16.))
#         out_w = int(math.ceil(input_shape[2] / 16.))
#         return [self.init_features * 16, out_h, out_w]

#     def forward(self, inputs):
#         enc1 = self.encoder1(inputs)
#         enc2 = self.encoder2(self.pool1(enc1))
#         enc3 = self.encoder3(self.pool2(enc2))
#         enc4 = self.encoder4(self.pool3(enc3))

#         bottleneck = self.bottleneck(self.pool4(enc4))

#         if not self.recon_enabled:
#             return {"feats": bottleneck, "recon": None}

#         dec4 = self.upconv4(bottleneck)
#         dec4 = torch.cat((dec4, enc4), dim=1)
#         dec4 = self.decoder4(dec4)
#         dec3 = self.upconv3(dec4)
#         dec3 = torch.cat((dec3, enc3), dim=1)
#         dec3 = self.decoder3(dec3)
#         dec2 = self.upconv2(dec3)
#         dec2 = torch.cat((dec2, enc2), dim=1)
#         dec2 = self.decoder2(dec2)
#         dec1 = self.upconv1(dec2)
#         dec1 = torch.cat((dec1, enc1), dim=1)
#         dec1 = self.decoder1(dec1)
#         recon = torch.sigmoid(self.conv(dec1))

#         return {"feats": bottleneck, "recon": recon}

#     def __repr__(self):
#         """Pretty print network."""
#         header = '{}'.format(str(self.__class__.__name__))
#         return header + '(input_channel={})'.format(self._input_channel)


class DeConv(ConvBase):
    def __init__(
        self,
        in_features,
        out_shape,
        input_channel=3,
        pretrained=False, #TODO(VS)
    ):
        self.in_features = in_features
        self.out_shape = out_shape
        self.nets = nn.ModuleDict()
        self._create_networks()

    def _create_network(self):
        self.nets["mlp"] = MLP(
            input_dim=self.in_features,
            output_dim=1000,
            layer_dims=[100, 100, 100],
            layer_func=nn.Linear,
            activation=nn.ReLU,
            output_activation=None,
        )
        self.nets["upconv"] = Sequential(
            nn.ConvTranspose2d() # TODO
        )

    def output_shape(self, input_shape):
        """
        Function to compute output shape from inputs to this module. 

        Args:
            input_shape (iterable of int): shape of input. Does not include batch dimension.
                Some modules may not need this argument, if their output does not depend 
                on the size of the input, or if they assume fixed size input.

        Returns:
            out_shape ([int]): list of integers corresponding to output shape
        """
        return self.out_shape

    def forward(self, inputs):
        mlp_out = self.nets["mlp"](inputs)
        in_feats = torch.reshape(mlp_out, [*inputs.shape[:-2], 1, 10, 10])
        out_feats = self.nets["upconv"](in_feats)
        return out_feats