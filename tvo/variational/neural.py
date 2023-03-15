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

from rmbvae import rmbvae

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
        # self.encoder = encoder
        # self.sampling = sampling
        # self.n_samples = n_samples  # number of samples to draw from sampling distribution per update cycle
        self.encoder=config['encoder']
        self.sampling=config['sampling']
        self.loss_function=config['loss_function']
        self.loss_name=config['loss_name']
        self.bitflipping=bitflipping
        self.RBMVAE = False

        self.N = N
        self.S = S
        self.n_samples = n_samples
        self.H = H

        self.hacky_ignore_substitution = False


        # picking an optimizer
        self.optimizer = to.optim.Adam(self.encoder.parameters(), lr=lr)

        # RBMAVE
        # todo: remove rmbvae hack

        if self.loss_name=='n_accepted' and self.RBMVAE:
            self.REPARAMETERIZED=True
            self.m=rmbvae.RMBVAE(
                n_hidden=config['n_hidden'],
                input_dimensionality=25,
                d=config['output_size'],
                r=config['output_size'],
                lambda_=0.5
            )
            self.encoder=self.m.encode
            self.sampling=self.m.reparameterized_rmb_sample
            self.optimizer=to.optim.Adam(list(self.m.variance_stack.parameters())+
                                          list(self.m.alpha_stack.parameters()), lr=lr)

        # set training or evaluation mode
        self.set_nn_training(training)
        self.set_k_updating(k_updating)

        # populate train set if any
        if getattr(self, "decoder_train_labels", None):
            self.K = self.decoder_train_labels

    def init_states(K: int, N: int, H: int, S: int, precision: to.dtype):
        raise NotImplementedError  # pragma: no cover

    @to.enable_grad()
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
        # with self.gradients_context_manager:
        #     q = self.encoder(batch) # n x h

        q = self.encoder(batch)


        # sample new variational states
        for i in range(self.n_samples):
            # get new variational states
            using_neural_marginal = True #to.rand(1) < 2
            using_marginal = 0
            using_prior = 0 #~ (using_marginal | using_neural_marginal)

            if using_neural_marginal:
                if self.RBMVAE:
                    new_states[:, i, :] = self.sampling(q[0], q[1]) if hasattr(self, 'RBMVAE') else self. sampling(q)
                else:
                    try:
                        new_states[:, i, :] = self.sampling(q)
                    except AssertionError:
                        new_states_i=False
            elif using_marginal:
                p = to.mean(self.K[idx].to(self.precision), axis=1)
                new_states[:, i, :] = self.sampling(p)
            elif using_prior:
                new_K_prior = (
                        to.rand(batch.shape[0],new_states.shape[-1], device=self.K.device)
                        < model.theta["pies"]
                ).byte()
                new_states[:, i, :] = new_K_prior

        # flip bits for stochasticity
        self.new_states_bfd = new_states  #
        if self.bitflipping:
            self.new_states_bfd=self.bitflipping(new_states, n_children=8, sparsity=0.1, p_bf=0.1)

        # get lpj of new variational states
        self.new_lpj = lpj_fn(batch, self.new_states_bfd)

        # update K set
        set_redundant_lpj_to_low(self.new_states_bfd, self.new_lpj, old_states=self.K[idx])
        n_subs, vector_subs = update_states_for_batch(
            new_states=self.new_states_bfd.clone().to(device=self.K.device).detach(),
            new_lpj=self.new_lpj.clone().to(device=self.lpj.device).detach(),
            idx=idx,
            all_states=self.K,
            all_lpj=self.lpj,
            sort_by_lpj=sort_by_lpj,
            vector_subs=True
        )



        # update encoder
        if self._training:
            # with self.gradients_context_manager:
            #     # accumulate loss

            if self.loss_name=='LPJ':
                loss = self.loss_function(self.new_lpj,lpj[idx].clone())
            elif self.loss_name in ('BCE','CE'):
                p = to.mean(self.K[idx].to(self.precision), axis=1)
                p.requires_grad_(True)
                loss = self.loss_function(p,q)
            elif self.loss_name=='log_bernoulli_loss':
                '''
                '''
                # get positions of accepted lpj
                # least_accepted=vector_subs.sum(axis=1).float()
                # sorted_lpj=to.sort(self.new_lpj,axis=1)[0]
                # minf=to.ones(sorted_lpj.shape[0])-to.inf
                # to.cat((minf.unsqueeze(1).T, sorted_lpj.T), 0).T
                # accepted = sorted_lpj >= sorted_lpj[least_accepted]
                # self.new_states_bfd=self.new_states_bfd.float().requires_grad = True # remove
                loss = self.loss_function(states=self.new_states_bfd, pies=q, accepted=vector_subs.float())

            elif self.loss_name=='n_accepted':
                states=self.new_states_bfd
                logits=q # log probabilities (!)
                accepted=vector_subs.float()
                Batch, M_samples, H2 = states.shape  # BxNxH
                # logits = logits.unsqueeze(axis=1).expand(Batch, M_samples, H2)
                logits_reshaped = logits.reshape(logits.shape[0], logits.shape[1] // 2, 2)
                p_gumbel = to.distributions.Categorical(logits=logits_reshaped)
                log_p_phi = p_gumbel.log_prob(states.squeeze()).sum() # cat prob [s, not(s)]

                delta_lpj = self.new_lpj - self.lpj[idx].min(axis=1)[0]
                expectation = (1 / to.tensor(M_samples)) + to.sum(delta_lpj * log_p_phi, axis=-1)
                loss = -expectation.sum()
                # for i in range(self.n_samples):
                #     z = new_states[:, i, :]
                #     # loss +=self.m.BinaryConcretePMF(d=z, a=q[0], l=0.5)
                #     # loss += self.m.mrb_kl(q[0], z).sum()
                #     n_accepted=vector_subs.sum(axis=1).float()
                #     a_loss = q[0].T@n_accepted
                #     S_loss = q[1].T@n_accepted
                #     loss+=a_loss.sum() + S_loss.sum()
                #     # self.encoder.losses.append(loss.detach().numpy())
                loss=loss.sum()
            elif self.loss_name=='e_logjoint_under_phi':
                s='Factorized_bernoulli'
                if s=='Factorized_bernoulli':
                    log_p_phi=to.sum(q, axis=1) # do this correctly
                elif s=='GS':
                    p_phi=to.A
                loss = 0
                for i in range(self.n_samples):
                    self.new_lpj=self.new_lpj.detach()
                    loss += self.loss_function(lpj=self.new_lpj[:,i].detach(), p_phi=p_phi)


            else:
                raise ValueError(f"Unknown loss function: {self.loss_fname}")  # pragma: no cover





                # if to.rand(1) > 0.95:
                #     print('Loss={}'.format(loss.detach().numpy()))

                # self.encoder.losses.append(loss.detach().numpy())# todo: remove rmbvae hack
                # reenable above
            # to.autograd.set_detect_anomaly(True)


            loss.backward()
            # clip gradient
            to.nn.utils.clip_grad_norm_(self.encoder.parameters(), 10)
            self.optimizer.step()

                # if self.encoder.layer_0.weight.grad:
                #     loss.backward(retain_graph=True)

        # clean states and lpjs:
        # self.new_states_bfd = self.new_states_bfd
        # self.new_lpj = self.new_lpj.detach()
        # update the variational states
        # if self.k_updating:
        # print(new_states.shape, new_lpj.shape, self.K[idx].shape)
        # print(new_states_bfd.max(), new_states_bfd.min(), new_states_bfd.max()/to.prod(to.tensor(new_states_bfd.shape)))
        # print(new_lpj.min(), new_lpj.max(), new_lpj.mean(), (new_lpj-lpj.min()).max())

        return n_subs

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




    def set_nn_training(self, training: bool):
        if training:
            self.gradients_context_manager = to.set_grad_enabled(True) # NullContextManager()
            # self.encoder.train() # todo: fix
            self._training = True
        else:
            self.gradients_context_manager = to.no_grad()
            self.encoder.eval()
            self._training = False

    def set_k_updating(self, k_updating:bool):
        self.k_updating=k_updating

    def accepted(self, old_lpj, new_lpj, vector_subs):
        min_lpj = to.min(old_lpj, dim=1)[0]
        raise NotImplementedError('not correct')
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
        dropout_rate: float,
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
                "layer_" + str(i), to.nn.Linear(feature_size_in, self.hidden_size[i],
                                                dtype=self.precision,
                                                device=tvo.get_device())
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
