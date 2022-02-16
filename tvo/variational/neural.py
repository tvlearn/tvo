import numpy as np
import torch as to

from itertools import combinations
from typing import Callable, Tuple, Optional, Sequence
from torch import Tensor

import tvo
from tvo.utils import get
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.variational._utils import update_states_for_batch, set_redundant_lpj_to_low

from tvo.utils.model_protocols import Optimized, Trainable
import warnings


class NeuralVariationalStates(TVOVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        S_new: int,
        precision: to.dtype,
        K_init_file: str = None,
        update_decoder: bool = True,
        encoder: str = "MLP",
        sampling: str = "Gumbel",
        lr: float = 1e-3,
        **kwargs,
    ):
        """
        :param N: Number of data points
        :param H: Output dimension of the encoder
        :param S: Number of samples in the K set
        :param S_new: Number of samples to draw for new variational states
        :param precision: floting point precision
        :param K_init_file: File to load the initial K matrix from
        :param update_decoder: If true, the decoder is updated during training
        :param encoder: Type of encoder to use. Either "MLP" or "CNN".
        :param sampling: Type of sampling to use. Currently only "Gumbel Softmax" is supported.
        """

        self.update_decoder = update_decoder
        self.encoder = encoder
        self.sampling = sampling
        self.n_samples = S  # number of samples to draw from gumbel distribution per update cycle

        # picking a model
        if encoder == "MLP":
            self.encoder = MLP(**kwargs)
        elif encoder == "CNN":
            self.encoder = CNN(**kwargs)
        else:
            raise ValueError(f"Unknown model: {encoder}")  # pragma: no cover

        # picking sampling method
        if sampling == "Gumbel":
            assert (
                kwargs["output_activation"] == to.nn.Identity
            ), "output_activation must be nn.Identity for gumbel-Softmax"
            self.sampling = self.gumbel_softmax_sampling
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")  # pragma: no cover

        # picking an optimizer
        self.optimizer = to.optim.Adam(self.encoder.parameters(), lr=lr)

        # building full config
        config = dict(
            N=N,
            H=H,
            S=S,
            S_new=S_new,
            precision=precision,
            encoder=encoder,
            sampling=sampling,
            k_init_file=K_init_file,
            update_decoder=update_decoder,
        )
        for k, v in kwargs.items():
            config[k] = v

        super().__init__(config)

    def gumbel_softmax_sampling(
        self, logits: Tensor, temperature: float = 1.0, hard: bool = False
    ) -> Tensor:
        """
        Implements Gumbel Softmax
        """
        # sample from gumbel distribution
        return to.nn.functional.gumbel_softmax(logits, temperature, hard)

    def init_states(K: int, N: int, H: int, S: int, precision: to.dtype):
        raise NotImplementedError  # pragma: no cover

    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        """Generate new variational states, update K and lpj with best samples and their lpj.

        :param idx: data point indices of batch w.r.t. K
        :param batch: batch of data points
        :param model: the model being used
        :returns: average number of variational state substitutions per datapoint performed
        """
        if isinstance(model, Optimized):
            lpj_fn = model.log_pseudo_joint
            sort_by_lpj = model.sorted_by_lpj
        else:
            lpj_fn = model.log_joint
            sort_by_lpj = {}

        n_samples = self.n_samples
        lpj = self.lpj

        self.optimizer.zero_grad()
        loss = 0
        q = self.encoder(batch)
        for i in range(n_samples):
            # sample new variational states
            states = self.sampling(q)

            # get lpj of new variational states
            lpj_new = lpj_fn(batch, states)
            loss -=lpj_new

        loss.backward()

        print('Average batch diff ={}'.format(lpj-loss/n_samples))

        # update K and lpj

    def get_variational_states(self, q):
        """
        Wraps the logic of getting variational states.
        param: q: output of encoder

        """
        if self.update_decoder:
            return self.sampling(q)
        else:
            with to.no_grad():
                warnings.warn("Sampling is used with no grad")
                return self.sampling(q)

    def update_k_lpj(self):
        pass


class MLP(to.nn.Module):
    '''
    Builds an MLP.
    :param input_size: input dimensionality
    :param n_hidden: number of units per hidden layer
    :param output_size: output dimensionality
    :param activations: list of activation functions for the hidden layers in the format of to.nn.activation
    :param output_activation: activation function for the output layer
    :param dropouts: List of boolean values indicating whether to apply dropout to each layer
    :param dropout_rate: global dropout rate
    :param **kwargs: unused, for compatibility with other models
    '''
    def __init__(
        self,
        input_size: int,
        n_hidden: Sequence[int],
        output_size: int,
        activations: Sequence[Callable],
        dropouts: Sequence[bool],
        dropout_rate: int,
        output_activation=to.nn.Identity,
        **kwargs,
    ):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = n_hidden
        self.output_size = output_size
        self.activations = activations
        self.output_activation = output_activation
        self.dropouts = dropouts
        self.dropout_rate = dropout_rate

        self.sanity_check()

        self.build_mlp()

    def build_mlp(self):

        # define a temporary flatten class
        class Flatten(to.nn.Module):
            def forward(self, input):
                return input.view(input.size(0), -1)

        # initialize layer input size
        feature_size_in = self.input_size

        # flatten input
        self.add_module("flatten", Flatten())

        # add the hidden layer blocks
        for i in range(len(self.hidden_size)):

            # add  Wx + b module
            self.add_module("layer_" + str(i), to.nn.Linear(feature_size_in, self.hidden_size[i]))

            # optionally apply dropout
            if self.dropouts[i]:
                self.add_module("dropout_" + str(i), to.nn.Dropout(p=self.dropout_rate))

            # add nonlinearity
            self.add_module("activation_" + str(i), self.activations[i]())

            # update layer input size for the next layer
            feature_size_in = self.hidden_size[i]

        # add the output layer
        self.add_module("output_layer", to.nn.Linear(feature_size_in, self.output_size))
        self.output_activation = self.output_activation()

    def forward(self, x):
        # pass the input through the network
        for module in list(self.modules())[1:]:
            x = module(x)
        return x

    def sanity_check(self):
        assert len(self.hidden_size) == len(self.activations) == len(self.dropouts)
        assert self.output_activation in [
            to.nn.Sigmoid,
            to.nn.Identity,
        ], "Only Sigmoid or Identity activations are supported"


class CNN(to.nn.Module):
    def __init__(self, **kwargs):
        super(CNN, self).__init__()
        raise NotImplementedError  # pragma: no cover

