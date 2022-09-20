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

from tvo.variational.evo import batch_sparseflip, batch_randflip

class NeuralVariationalStates(TVOVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        precision: to.dtype,
        K_init=None,
        update_decoder: bool = True,
        encoder: str = "MLP",
        sampling: str = "Gumbel",
        bitflipping: str = "sparseflip",
        n_samples: int = 1,
        lr: float = 1e-3,
        training=True,
        k_updating=True,
        **kwargs,
    ):
        """
        :param N: Number of data points
        :param H: Output dimension of the encoder
        :param S: Number of samples in the K set
        :param n_samples: Number of samples to draw for new variational states
        :param precision: floting point precision
        :param K_init: File to load the initial K matrix from. to.Tensor of shape (N, S, H)
        :param update_decoder: If true, the decoder is updated during training
        :param encoder: Type of encoder to use. Either "MLP" or "CNN".
        :param sampling: Type of sampling to use. Currently only "Gumbel Softmax" is supported.
        """

        # build full config
        config = dict(
            N=N,
            H=H,
            S=S,
            n_samples=n_samples,
            precision=precision,
            encoder=encoder,
            sampling=sampling,
            K_init=K_init,
            update_decoder=update_decoder,
        )
        # add any  kwargs to config
        for k, v in kwargs.items():
            config[k] = v
        # init TVOariationalStates
        super().__init__(config, K_init=K_init)

        # save parameters
        self.update_decoder = update_decoder
        self.encoder = encoder
        self.sampling = sampling
        self.n_samples = n_samples  # number of samples to draw from sampling distribution per update cycle
        self.N = N
        self.S = S
        self.n_samples = n_samples
        self.H = H

        # picking a model
        if encoder == "MLP":
            self.encoder = MLP(**kwargs)
        elif encoder == "CNN":
            self.encoder = CNN(**kwargs)
        else:
            raise ValueError(f"Unknown model: {encoder}")  # pragma: no cover

        # select sampling method
        if sampling == "Gumbel":
            assert (
                kwargs["output_activation"] == to.nn.Identity
            ), "output_activation must be nn.Identity for gumbel-Softmax"
            self.sampling = self.gumbel_softmax_sampling
        elif sampling =="Independent Bernoullis":
            assert(kwargs["output_activation"] == to.nn.Sigmoid), "output_activation must be nn.Sigmoid for Independent Bernoullis"
            #assert(kwargs["loss_name"] in ["BCE", "CE"]), "loss_function must be Binary Crossentropy"
            self.sampling = self.independent_bernoullis_sampling
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")  # pragma: no cover

        # select bit flipping method
        if bitflipping ==  'randflip':
            self.bitflipping = batch_randflip
        elif bitflipping == 'sparseflip':
            self.bitflipping = batch_sparseflip
        elif bitflipping == 'none':
            self.bitflipping = None
        print(bitflipping)

        # select a loss function
        self.loss_fname = kwargs["loss_name"]
        assert self.loss_fname in ["BCE", "LPJ", 'CE', 'log_bernoulli_loss'], "loss_function not listed"

        if self.loss_fname == "BCE":
            self.loss_function = to.nn.BCELoss()
        elif self.loss_fname == "LPJ":
            self.loss_function = self.lpj_loss
        elif self.loss_fname == "CE":
            self.loss_function = to.nn.CrossEntropyLoss()
        elif self.loss_fname =='log_bernoulli_loss':
            assert sampling=='Independent Bernoullis', 'sampling is {}'.format(sampling)
            self.loss_function = self.log_bernoulli_loss
        else:
            raise ValueError(f"Unknown loss function: {self.loss_name}")  # pragma: no cover

        # picking an optimizer
        self.optimizer = to.optim.Adam(self.encoder.parameters(), lr=lr)

        # set training or evaluation mode
        self.set_nn_training(training)
        self.set_k_updating(k_updating)

        # populate train set if any
        if getattr(self, "decoder_train_labels", None):
            self.K = self.decoder_train_labels

    def gumbel_softmax_sampling(
        self, logits: Tensor, temperature: float = 0.3, hard: bool = True
    ) -> Tensor:
        """
        Implements Gumbel Softmax on logits of a neural network.
        :param logits: logits of a neural network
        :param temperature: temperature of the gumbel softmax. Annealing schedule is not implemented here.
        :param hard: If true, the returned samples are one-hot encoded.
        """
        # sample from gumbel distribution
        assert len(logits) // 2 == len(logits) / 2, "logits must be of even length"

        # reformat logits to be of shape (batch_size, model_H, 2)
        logits_reshaped = logits.reshape(logits.shape[0], logits.shape[1] // 2, 2)
        # init states tensor
        states = to.nn.functional.gumbel_softmax(logits_reshaped, temperature, hard)
        # keep only the first element of the last dimension (p(x=1))
        states = states[:, :, 0]
        # reformat states to be of shape (batch_size, model_H)
        states = states.reshape(logits.shape[0], logits.shape[1] // 2)

        return states.type(to.uint8) if hard else states

    def independent_bernoullis_sampling(self, logits: Tensor, hard: bool = True) -> Tensor:
        # assert that pies are in [0, 1]
        assert to.min(logits) >= 0 and to.max(logits) <= 1, "pies must be in [0, 1]"
        # sample from bernoullis
        states = to.bernoulli(logits).type(to.uint8) if hard else logits
        return states

    def init_states(K: int, N: int, H: int, S: int, precision: to.dtype):
        raise NotImplementedError  # pragma: no cover

    def neural_update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        """Generate new variational states, update K and lpj with best samples and their lpj.

        :param idx: data point indices of batch w.r.t. K
        :param batch: batch of data points
        :param model: the model being used
        :returns: average number of variational state substitutions per datapoint performed
        """

        # get log (pseudo)joints of the current variational states
        if isinstance(model, Optimized):
            lpj_fn = model.log_pseudo_joint
            sort_by_lpj = model.sorted_by_lpj
        else:
            lpj_fn = model.log_joint
            sort_by_lpj = {}

        K, lpj = self.K, self.lpj
        batch_size, H = batch.shape[0], K.shape[2]
        lpj[idx] = lpj_fn(batch, K[idx])


        # init states tensor
        n, k, h = K[idx].shape # n: batch size, k: number of states, h: size of state
        new_states = to.empty((n, self.n_samples, h), dtype=to.uint8, device=self.lpj.device)

        # prepare training process
        if self._training:
            self.optimizer.zero_grad()

        # get parameters of sampling distribution
        with self.gradients_context_manager:
            # if self._training:
            #     batch.requires_grad_(True)
            q = self.encoder(batch) # n x h

        # sample new variational states
        for i in range(self.n_samples):
            # get new variational states
            using_neural_marginal = True #to.rand(1) < 2
            using_marginal = 0
            using_prior = 0 #~ (using_marginal | using_neural_marginal)

            if using_neural_marginal:
                new_states[:, i, :] = self.sampling(q)
            elif using_marginal:
                p = to.mean(self.K[idx].to(self.precision), axis=1)
                new_states[:, i, :] = self.sampling(p)
            elif using_prior:
                new_K_prior = (
                        to.rand(batch.shape[0],new_states.shape[-1], device=self.K.device)
                        < model.theta["pies"]
                ).byte()
                new_states[:, i, :] = new_K_prior

        # tvs prior
        # new_states = (
        #                     to.rand(batch.shape[0],self.n_samples,new_states.shape[-1], device=self.K.device)
        #                     < model.theta["pies"]
        #             ).byte()


        # flip bits for stochasticity
        new_states_bfd = new_states  #
        if self.bitflipping:
            new_states_bfd=self.bitflipping(new_states, n_children=8, sparsity=0.1, p_bf=0.1)

        # get lpj of new variational states
        new_lpj = lpj_fn(batch, new_states_bfd)
        self.K[idx]

        # update encoder
        if self._training:
            with self.gradients_context_manager:
                # accumulate loss
                if self.loss_fname=='LPJ':
                    loss = self.loss_function(new_lpj,lpj[idx].clone())
                elif self.loss_fname in ('BCE','CE'):
                    p = to.mean(self.K[idx].to(self.precision), axis=1)
                    p.requires_grad_(True)
                    loss = self.loss_function(p,q)
                elif self.loss_fname=='log_bernoulli_loss':
                    loss = self.loss_function(q, self.accepted(old_lpj=lpj[idx], new_lpj=new_lpj))
                else:
                    raise ValueError(f"Unknown loss function: {self.loss_fname}")  # pragma: no cover

                # if to.rand(1) > 0.95:
                #     print('Loss={}'.format(loss.detach().numpy()))

                self.encoder.losses.append(loss.detach().numpy())
                try:
                    to.autograd.set_detect_anomaly(True)
                    if self.encoder.layer_0.weight.grad:
                        loss.backward(retain_graph=True)
                except RuntimeError as e:
                    loss = self.loss_function(p, q)
                    loss.backward()
                    raise e

        # update the variational states
        # if self.k_updating:
        # print(new_states.shape, new_lpj.shape, self.K[idx].shape)
        # print(new_states_bfd.max(), new_states_bfd.min(), new_states_bfd.max()/to.prod(to.tensor(new_states_bfd.shape)))
        # print(new_lpj.min(), new_lpj.max(), new_lpj.mean(), (new_lpj-lpj.min()).max())
        set_redundant_lpj_to_low(new_states_bfd, new_lpj, old_states=self.K[idx])

        return update_states_for_batch(
            new_states=new_states_bfd.to(device=self.K.device),
            new_lpj=new_lpj.to(device=self.lpj.device),
            idx=idx,
            all_states=self.K,
            all_lpj=self.lpj,
            sort_by_lpj=sort_by_lpj,
        )

    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        return self.neural_update(idx, batch, model)

    def tvs_update(self, idx: to.Tensor, batch: to.Tensor, model: Trainable) -> int:
        """minimal tvs prior sampling. Used for debugging"""
        if isinstance(model, Optimized):
            lpj_fn = model.log_pseudo_joint
            sort_by_lpj = model.sorted_by_lpj
        else:
            lpj_fn = model.log_joint
            sort_by_lpj = {}

        K, lpj = self.K, self.lpj
        batch_size, H = batch.shape[0], K.shape[2]
        lpj[idx] = lpj_fn(batch, K[idx])

        new_K_prior = (
            to.rand(batch_size, self.n_samples, H, device=K.device)
            < model.theta["pies"]
        ).byte()

        new_K = new_K_prior

        new_lpj = lpj_fn(batch, new_K)

        set_redundant_lpj_to_low(new_K, new_lpj, K[idx])

        return update_states_for_batch(
            new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj
        )

    def lpj_loss(self, new_lpj, old_lpj):
        s_min = old_lpj.min(dim=1)[0].clone()
        loss=to.sum(s_min[:,None]-new_lpj)
        return loss

    def log_bernoulli_loss(self, pies, accepted):
        '''
        :param pies: pies of bernoulli distribution
        :param accepted: whether associated log_joing is accepted
        :return:
        '''
        log_p_phi = to.sum(to.log(pies))
        # p_phi = to.prod(pies)
        delta_accepted = accepted.sum(dim=1)
        N = accepted.shape[1]
        return to.log(1/to.tensor(N))+ log_p_phi+to.log(delta_accepted) # 1/n sum(accepted_bool * p_phi)


    def set_nn_training(self, training: bool):
        if training:
            self.gradients_context_manager = to.set_grad_enabled(True) # NullContextManager()
            self.encoder.train()
            self._training = True
        else:
            self.gradients_context_manager = to.no_grad()
            self.encoder.eval()
            self._training = False

    def set_k_updating(self, k_updating:bool):
        self.k_updating=k_updating

    def accepted(self, old_lpj, new_lpj):
        min_lpj = to.min(old_lpj, dim=1)[0]
        return new_lpj > min_lpj[:,None]

class MLP(to.nn.Module):
    """
    Builds an MLP.
    :param input_size: input dimensionality
    :param n_hidden: number of units per hidden layer
    :param output_size: output dimensionality
    :param activations: list of activation functions for the hidden layers in the format of to.nn.activation
    :param output_activation: activation function for the output layer
    :param dropouts: List of boolean values indicating whether to apply dropout to each layer
    :param dropout_rate: global dropout rate
    :param **kwargs: unused, for compatibility with other models
    """

    def __init__(
        self,
        input_size: int,
        n_hidden: Sequence[int],
        output_size: int,
        activations: Sequence[Callable],
        dropouts: Sequence[bool],
        dropout_rate: int,
        output_activation=to.nn.Identity,
        precision: to.dtype = to.float32,
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
        self.precision = precision

        self.sanity_check()

        self.build_mlp()

        self.losses = []
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
            self.add_module(
                "layer_" + str(i), to.nn.Linear(feature_size_in, self.hidden_size[i], dtype=self.precision, device=tvo.get_device())
            )

            # optionally apply dropout
            if self.dropouts[i]:
                self.add_module("dropout_" + str(i), to.nn.Dropout(p=self.dropout_rate))

            # add nonlinearity
            self.add_module("activation_" + str(i), self.activations[i]())

            # update layer input size for the next layer
            feature_size_in = self.hidden_size[i]

        # add the output layer
        self.add_module("output_layer", to.nn.Linear(feature_size_in, self.output_size, dtype=self.precision,
                                                     device=tvo.get_device()))
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

class NullContextManager(object):
    def __init__(self, object=None):
        self.object = object
    def __enter__(self):
        return self.object
    def __exit__(self, *args):
        pass
