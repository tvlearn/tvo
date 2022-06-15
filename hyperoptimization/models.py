import numpy as np
from warnings import warn
from typing import List

import torch as to
import torch.nn as nn
import logging

logging.basicConfig(level=logging.DEBUG)


class FCDeConvNet(to.nn.Module):
    def __init__(
        self,
        n_deconv_layers: int,
        n_fc_layers: int,
        W_shapes: List[int],
        fc_activations: List,
        dc_activations: List,
        n_filters: List[int],
        dropouts: List[bool],
        batch_norms: List[bool],
        output_shape: int,
        input_size,
        dropout_rate=0.25,
        filters_from_fc=1,
        kernels=None,
        paddings=None,
        sanity_checks=False,
    ):
        """
        Adjustable deconvolutional network class. It builds an optionally deconvolutional
         generative model with a fully connected base. Both options are set in blocks of
         [(fully connected/deconv) layer,(dropout/batchnorm) regularizer, nonlinearity].
         # Todo make fc base completely optional?

        :param n_deconv_layers: number of transposed convolutions to be applied to the
         embedding (after fc layers)
        :param n_fc_layers: number of fully connected layers to be applied to S
        :param W_shapes: weight shapes of the fully connected layers. Currenly one shape
         is used for all layers.
        :param fc_activations: set of activations for the fully connected layers.
        :param dc_activations: set of activations for the deconv layers.
        :param n_filters:  number of filters per deconv layer
        :param dropouts: List of dropout booleans. Only applied to fc layers.
        :param batch_norms: List of batch norm booleans. Only applied to deconv blocks.
        :param output_shape: X.shape[-1]
        :param input_size: S.shape
        :param dropout_rate: global dropout rate # todo enable local dropout?
        :param filters_from_fc: the amount of filters to use for the hidden representation
         of the linear stack
        :param sanity_checks: BOOL, decides whether sanity checks are run at init.
        """
        super().__init__()

        # # override the first filter size?
        # if initial_filters: # yes
        #     n_filters[0]=initial_filters
        # else: # no
        #     initial_filters = n_filters[0]

        if sanity_checks:
            fc_sanitize = (W_shapes, n_fc_layers, dropouts)
            dc_sanitize = (n_deconv_layers, n_filters, batch_norms, filters_from_fc)
            self.test_sanity(input_size, fc_sanitize, dc_sanitize)

        self.shape = [input_size]

        self.n_linear_layers = n_fc_layers
        self.n_deconv_layers = n_deconv_layers
        self.linear_stack = nn.Sequential()
        self.deconv_stack = nn.Sequential()

        # setup fully connected blocks
        in_features = input_size
        if not n_deconv_layers:
            W_shapes[-1] = output_shape

        # build fc blocks
        for i, (n_hidden, activation, dropout) in enumerate(
            zip(W_shapes, fc_activations, dropouts)
        ):
            self.shape.append(n_hidden)  # store shape for TVEM
            self.linear_stack.add_module(
                "linear_{}".format(i),
                nn.Linear(in_features, out_features=n_hidden),
            )
            if dropout:
                self.linear_stack.add_module("dropout_layer{}".format(i), nn.Dropout(dropout_rate))
            self.linear_stack.add_module("activation_{}".format(i), eval(activation)())
            in_features = n_hidden

        # transposed convolution blocks
        input_len = int(np.sqrt(in_features))
        input_shape = (input_len, input_len, filters_from_fc)

        if not kernels:
            # calculate total increase in dimensionality
            total_upsampling = int(np.sqrt(output_shape) - np.sqrt(input_shape[0] * input_shape[1]))
            assert total_upsampling == np.sqrt(output_shape) - np.sqrt(
                input_shape[0] * input_shape[1]
            )

            if total_upsampling < 0:
                warn("Transposed convolution used for downsampling")

            # calculate kernel sizes and paddings, such as the outputs match the
            # dimensionality of the output
            kernels, paddings = self.deconvolution_hypers_from_upsampling(
                upsampling=total_upsampling, min_kernel=3, n_layers=n_deconv_layers
            )

        if not paddings:
            paddings = [0] * len(kernels)

        # print(total_upsampling, kernels, paddings)

        # add the transposed convolution blocks
        for i in range(n_deconv_layers):

            self.shape.append((n_filters[i], kernels[i]))
            self.deconv_stack.add_module(
                "conv_transpose_{}".format(i),
                nn.ConvTranspose2d(
                    in_channels=input_shape[-1],
                    out_channels=n_filters[i],
                    kernel_size=kernels[i],
                    padding=paddings[i],
                ),
            )

            if batch_norms[i]:
                self.deconv_stack.add_module(
                    "batch_norm_{}".format(i), nn.BatchNorm2d(n_filters[i])
                )

            self.deconv_stack.add_module(
                "deconv_activation_{}".format(i), eval(dc_activations[i])()
            )

            input_shape = self.deconv_output_shape(
                input_len=input_shape[0],
                filters=n_filters[i],
                kernel=kernels[i],
                padding=paddings[i],
            )

        assert input_shape[0] == input_shape[1]
        assert output_shape == np.prod(input_shape), 'output ({}) not equal to product of input ({})'.format(output_shape,input_shape)
        self.dropout = nn.Dropout(p=dropout_rate)

        # todo: change self.shape functionality appropriately after the TVAE changes
        self.shape.append(output_shape)

    def forward(self, x):
        x = x.double()
        # x.to('cuda')
        h = self.linear_stack(x)

        n, S_k, D = x.shape[0], x.shape[1], self.D

        if self.n_deconv_layers > 0:

            h = h.reshape(n, S_k, int(np.sqrt(h.shape[-1])), int(np.sqrt(h.shape[-1])))
            out = to.empty(size=(n, S_k, D), device=h.device, dtype=h.dtype)
            for s in range(S_k):
                h_s = self.deconv_stack(h[:, s, :, :].unsqueeze(axis=1))
                # h_s = to.sum(h_s, dim=1)
                # todo force last filter to match the dimensionality
                out[:, s, :] = to.reshape(h_s, (n, D))
        else:
            out = h

        return out

    def number_of_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def conv_output_shape(self, input_len, filters, kernel, stride=1, padding=0, dilation=1):
        """
        returns the output shape of a convolutional layer
        :param input_len: length of input: for 3x28x28 coloured MNIST it's 28
        :param filters: number of filters
        :param kernel: 1D length of kernel: for a 3x3 kernel it's a 3
        :param stride: stride of the filter
        :param padding: 1D size of padding around the input.
        :return: the output shape of a convolutional layer
        """
        assert type(input_len + filters + stride + kernel) is int
        out = (input_len + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1
        out = int(out)
        return (out, out, filters)

    def deconv_output_shape(
        self,
        input_len,
        filters,
        kernel,
        stride=1,
        padding=0,
        dilation=1,
        output_padding=0,
    ):
        """
        returns the output shape of a transposed convolutional layer
        :param input_len: length of input: for 3x28x28 coloured MNIST it's 28
        :param filters: number of filters
        :param kernel: 1D length of kernel: for a 3x3 kernel it's a 3
        :param stride: stride of the filter
        :param padding: 1D size of padding around the input.
        :return: the output shape of a transposed convolutional layer
        """
        assert type(input_len + filters + stride + kernel + dilation + output_padding) is int
        out = int(
            stride * (input_len - 1) - 2 * padding + dilation * (kernel - 1) + output_padding + 1
        )
        return (out, out, filters)

    @staticmethod
    def deconvolution_hypers_from_upsampling(upsampling: int, min_kernel=3, n_layers=1):
        """
        :param upsampling: dimentionality needed to upsample image
        :param min_kernel: minimum kernel size
        :param n_layers: number of layers on which to spread the upsampling on
        :return: the kernel size and padding for each layer such that the upsampling is obeyed.
        """

        # todo: decide how to treat bad n_layers input
        if n_layers <= 0:
            return [], []
        assert n_layers, "no layers provided"

        if upsampling == 0:
            return [3] * n_layers, [1] * n_layers

        # each layer gets the same amount of upsampling
        layer_upsampling = upsampling // n_layers

        # any remaining upsampling goes to the final layer to reduce compute
        last_layer = layer_upsampling + upsampling % n_layers

        assert layer_upsampling * n_layers + upsampling % n_layers == upsampling

        # assign upsampling to layers
        layers_upsampling = [layer_upsampling for _ in range(n_layers)]
        layers_upsampling[-1] = last_layer
        kernels, paddings = [], []

        # compute kernel/padding combination for each upsampling value
        for upsampling_ in layers_upsampling:
            # padding adds 2
            # kernel adds -1, with kernel=1 -> upsampling=0
            kernel = min_kernel
            padding = (min_kernel - upsampling_ - 1) / 2
            if padding != int(padding):
                kernel -= 1 * np.sign(padding)
                padding = int(padding)

            if padding < 0:
                kernel += 2 * abs(padding)
                padding = 0

            # print('upsampling for layer: kernels={}, padding={}'.format(kernel, padding))
            assert (
                -2 * padding + (kernel - 1) == upsampling_
            ), "dafuq? padding = {}, kernel = {}, upsamplings = {}".format(
                padding, kernel, layers_upsampling
            )
            # todo decide whether to allow negative padding
            assert padding >= 0, "upsampling {} with negative padding={}".format(
                upsampling_, padding
            )
            kernels.append(int(kernel))
            paddings.append(int(padding))

        actual = np.sum(np.array(kernels) - 1) - 2 * np.sum(np.array(paddings))
        diff = actual - upsampling

        assert (
            diff == 0
        ), "Unexpected diff={} between expected result ({})" "and actual ({})".format(
            diff, upsampling, actual
        )
        return kernels, paddings

    @staticmethod  # static to make mypy happy
    def test_sanity(input_size, fc_sanitize, dc_sanitize):

        # make sanity checks
        # Todo: decide whether and which sanity checks can be removed
        W_shapes, n_fc_layers, dropouts = fc_sanitize
        n_deconv_layers, n_filters, batch_norms, filters_from_fc = dc_sanitize

        assert (
            len(W_shapes) == n_fc_layers == len(dropouts)
        ), "add information for all fc layers (dropout can be 0)"
        assert (
            len(n_filters) == n_deconv_layers == len(batch_norms)
        ), "add information for all deconv layers"

        if n_filters and n_fc_layers:
            initial_deconv_dim = int(np.sqrt(W_shapes[-1] / n_filters[0]))
            assert n_filters[0] == W_shapes[-1] / initial_deconv_dim ** 2, (
                "the output of the final fully connected layer should be "
                "a product of squares, where the product is the number "
                "of filters and the square is the shape of the filters. "
            )
        elif n_filters:
            initial_deconv_dim = int(np.sqrt(input_size / n_filters[0]))
            assert initial_deconv_dim == np.sqrt(
                input_size / n_filters[0]
            ), "pure deconvnet can only be used if the input size is a product of squares"

        assert (
            filters_from_fc == int(filters_from_fc) and filters_from_fc > 0
        ), "filters need to be positive"


# added in models.py
class FCDeConvNetSigOut(FCDeConvNet):
    def forward(self, x):
        return to.sigmoid(super(FCDeConvNetSigOut, self).forward(x))


# this function computes the feature map of a convolutional layer.
def feature_map(w, h, d, n_kernels, kernel_size):
    w2 = w - kernel_size + 1
    h2 = h - kernel_size + 1
    d2 = n_kernels
    volume = w2, h2, d2
    n_weights = kernel_size ** 2 * d * n_kernels
    return volume, n_weights


def deconv_2_l(n_filters):
    volume1, n_weights1 = feature_map(32, 32, 3, 3, 15)
    w, h, d = volume1
    volume2, n_weights2 = feature_map(w, h, d, n_filters, 15)
    v1 = volume1[0] * volume1[1] * volume1[2]
    v2 = volume2[0] * volume2[1] * volume2[2]
    return v1 + v2, n_weights2 + n_weights1
