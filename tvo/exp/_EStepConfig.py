# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0
import os.path
from abc import ABC, abstractmethod
from typing import Dict, Any, Sequence, List
from typing import Callable, Tuple, Optional, Sequence
from torch import Tensor
from tvo.variational.neural import MLP
from tvo.variational.evo import batch_sparseflip, batch_randflip
import torch as to

class EStepConfig(ABC):
    def __init__(self, n_states: int):
        """Abstract base configuration object for experiments' E-steps.

        :param n_states: Number of variational states per datapoint to keep in memory.
        """
        self.n_states = n_states

    @abstractmethod
    def as_dict(self) -> Dict[str, Any]:
        raise NotImplementedError  # pragma: no cover


class EVOConfig(EStepConfig):
    def __init__(
        self,
        n_states: int,
        n_parents: int,
        n_generations: int,
        parent_selection: str = "fitness",
        crossover: bool = True,
        n_children: int = None,
        mutation: str = "uniform",
        bitflip_frequency: float = None,
        K_init_file: str = None,
    ):
        """Configuration object for EVO E-step.

        :param n_states: Number of variational states per datapoint to keep in memory.
        :param n_parents: Number of parent states to select at each EVO generation.
                          Must be <= n_states.
        :param parent_selection: Parent selection algorithm for EVO. Must be one of:

                                 - 'fitness': fitness-proportional parent selection
                                 - 'uniform': random uniform parent selection
        :param crossover: Whether crossover should be applied or not.
                          Must be False if n_children is specified.
        :param n_children: Number of children per parent to generate via mutation
                           at each EVO generation. Required if crossover is False.
        :param mutation: Mutation algorithm for EVO. Must be one of:

                         - 'sparsity': bits are flipped so that states tend
                           towards current model sparsity.
                         - 'uniform': random uniform selection of bits to flip.
        :param bitflip_frequency: Probability of flipping a bit during the mutation step (e.g.
                                  2/H for an average of 2 bitflips per mutation). Required when
                                  using the 'sparsity' mutation algorithm.
        :param K_init_file: Full path to H5 file providing initial states
        """
        assert (
            not crossover or n_children is None
        ), "Exactly one of n_children and crossover may be provided."
        valid_selections = ("fitness", "uniform")
        assert (
            parent_selection in valid_selections
        ), f"Unknown parent selection {parent_selection}"
        valid_mutations = ("sparsity", "uniform")
        assert mutation in valid_mutations, f"Unknown mutation {mutation}"
        assert (
            n_parents <= n_states
        ), f"n_parents ({n_parents}) must be lower than n_states ({n_states})"
        assert (
            mutation != "sparsity" or bitflip_frequency is not None
        ), "bitflip_frequency is required for mutation algorithm 'sparsity'"

        self.n_parents = n_parents
        self.n_children = n_children
        self.n_generations = n_generations
        self.parent_selection = parent_selection
        self.crossover = crossover
        self.mutation = mutation
        self.bitflip_frequency = bitflip_frequency
        self.K_init_file = K_init_file

        super().__init__(n_states)

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


class NeuralEMConfig(EStepConfig):
    """
    Configuration object for Neural EM E-step.
    """

    def __init__(
        self,
        encoder: str,
        n_states: int,
        n_samples: int,
        input_size: int,
        activations: Sequence,
        output_size: int,
        n_hidden: List[int] = None,
        dropouts: List[bool] = None,
        dropout_rate: float = None,
        output_activation=None,
        lr: float = None,
        sampling: str = "Gumbel",
        bitflipping: str = None,
        K_init=None,
        loss_name: str =None,
        n_parents=None,
        n_children=None,
        **kwargs
    ):
        """
        :param encoder: encoder type to use. Must be one of: MLP, CNN
        :param n_states: number of states in the K set
        :param n_samples: number of samples to use for the E step
        :param input_size: input size of the encoder (D)
        :param activations: list of activations to use for the encoder
        :param output_size: output size of the encoder (H)
        :param n_hidden: list of hidden layer sizes for the encoder
        :param dropouts: list of dropouts for the encoder
        :param dropout_rate: dropout rate for the encoder
        :param output_activation: activation for the output layer.
        :param lr: learning rate for the optimizer of the encoder
        :param sampling: sampling method to use. Must be one of: Gumbel, Independent Bernoullis
        :param bitflipping: vector permutation method. See tvo/variational/evo.py for details.
        :param K_init: initial K set to use. If None, a random set of states is used. To.tensor.
        :param n_parents: number of initial states selected for permutation. See EVO for details.
        :param n_children: number of children per initial state. See EVO for details.
        """

        self.n_samples = n_samples

        self.K_init = K_init
        self.loss_name = loss_name
        self.n_parents = n_parents
        self.n_children = n_children

        assert bitflipping in ['sparseflip', 'randflip', None], '{} is unrecognized'.format(bitflipping)
        self.bitflipping = bitflipping

        if encoder == "MLP":

            self.input_size = input_size
            self.n_hidden = n_hidden
            self.activations = activations
            self.dropouts = dropouts
            self.dropout_rate = dropout_rate
            self.output_size = output_size
            self.output_activation = output_activation
            self.lr = lr
            self.sampling = sampling

            self.MLP_sanity_check()

            super().__init__(n_states)

        elif encoder == "CNN":
            raise NotImplementedError  # pragma: no cover
        else:
            raise ValueError(f"Unknown encoder {encoder}")

        if sampling == "GumbelSM":
            self.output_size *= 2
            output_size *= 2     # shift to bitwise categorical representation
        elif sampling == "Independent Bernoullis":
            pass
        else:
            raise ValueError(f"Unknown sampling method {sampling}")

        # picking a model
        if encoder == "MLP":
            self.encoder = MLP(input_size, n_hidden, output_size, activations, dropouts, dropout_rate, output_activation)
        elif encoder == "CNN":
            self.encoder = CNN(**kwargs)
        else:
            raise ValueError(f"Unknown model: {encoder}")  # pragma: no cover

        # select sampling method
        if sampling == "GumbelSM":
            # assert (
            #     kwargs["output_activation"] == to.nn.Identity
            # ), "output_activation must be nn.Identity for gumbel-Softmax"
            assert (output_activation == to.nn.Identity), "output_activation must be nn.Identity for gumbel-Softmax"
            self.sampling = self.gumbel_softmax_sampling
        elif sampling =="Independent Bernoullis":
            assert(output_activation == to.nn.Sigmoid), "output_activation must be nn.Sigmoid for Independent Bernoullis"
            #assert(kwargs["loss_name"] in ["BCE", "CE"]), "loss_function must be Binary Crossentropy"
            self.sampling = self.independent_bernoullis_sampling
        else:
            raise ValueError(f"Unknown sampling method: {sampling}")  # pragma: no cover

        # select bit flipping method
        if bitflipping ==  'randflip':
            self.bitflipping = batch_randflip
        elif bitflipping == 'sparseflip':
            self.bitflipping = batch_sparseflip
        else:
            bitflipping == 'none'
            self.bitflipping = None

        # select a loss function
        assert self.loss_name in ["BCE", "LPJ", 'CE', 'log_bernoulli_loss',
                                  'e_logjoint_under_phi','n_accepted'], "loss_function not listed"

        if self.loss_name == "BCE":
            self.loss_function = to.nn.BCELoss()
        elif self.loss_name == "LPJ":
            self.loss_function = self.lpj_loss
        elif self.loss_name == "CE":
            self.loss_function = to.nn.CrossEntropyLoss()
        elif self.loss_name =='log_bernoulli_loss':
            assert sampling=='Independent Bernoullis', 'sampling is {}'.format(sampling)
            self.loss_function = self.log_bernoulli_loss
        elif self.loss_name== 'e_logjoint_under_phi':
            self.loss_function = self.e_logjoint_under_phi
        elif self.loss_name== 'n_accepted':
            assert sampling == 'GumbelSM', 'sampling is {}'.format(sampling)
            self.loss_function= self.n_accepted
        else:
            raise ValueError(f"Unknown loss function: {self.loss_name}")  # pragma: no cover

    def MLP_sanity_check(self):
        assert self.n_hidden is not None, "n_hidden must be specified for MLP encoder"
        assert self.dropouts is not None, "dropouts must be specified for MLP encoder"
        assert (
            self.dropout_rate is not None
        ), "dropout_rate must be specified for MLP encoder"
        assert (
            len(self.n_hidden) == len(self.activations) == len(self.dropouts)
        ), "hidden units, activations and dropouts must be equal."

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)

    def log_bernoulli_loss(self,states, pies, accepted):
        '''
        :param pies: pies of bernoulli distribution (in case of matrix input, this function assumes independence)
        :param accepted: whether associated log_joing is accepted
        :return:
        '''
        # inputs are : q, vector_subs, n_accepted=vector_subs.sum(axis=1).float()

        Batch, M_samples, H = states.shape # BxNxH
        pies=pies.unsqueeze(axis=1).expand(Batch, M_samples, H)
        # sum over H
        log_p_phi = to.sum(states*(to.log(pies)-to.log(1-pies))+to.log(1-pies), axis=-1)

        delta_lpj = accepted
        # sum over M
        # derivative taken by calling backward, as only log_p_phi depends on phi
        expectation=(1/to.tensor(M_samples))+to.sum(delta_lpj*log_p_phi, axis=-1)
        loss=-expectation.sum()
        return loss

    def e_logjoint_under_phi(self, lpj, p_phi):
        return to.dot(lpj, p_phi).sum()

    def n_accepted(self, n_accepted, q):
        # assert n_accepted.shape[0]==q.shape[0]
        return n_accepted*q

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
        assert len(logits[1]) // 2 == len(logits[1]) / 2, "logits must be of even length, but were {}. " \
                                                    "Is sampling set as Gumbel-Softmax?".format(len(logits))

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

        assert to.min(logits) >= 0 and to.max(logits) <= 1, \
            "Pies should have a min of at least 0 and a max of at most 1. Instead, they are {} and {}".format(
                to.min(logits),
                to.max(logits)
            )
        # sample from bernoullis
        states = to.bernoulli(logits).type(to.uint8) if hard else logits
        return states

    def lpj_loss(self, new_lpj, old_lpj):
        s_min = old_lpj.min(dim=1)[0].clone()
        loss=to.sum(s_min[:,None]-new_lpj)
        return loss


class PreAmortizedConfig(EStepConfig):

    def __init__(
            self,
            N: int,
            H: int,
            S: int,
            model_path: str,
            nsamples: int,
            K_init_file: str = None,
    ):
        super().__init__(nsamples)

        self.N = N
        self.H = H
        self.S = S
        self.model_path = model_path
        self.nsamples = nsamples
        self.K_init_file = K_init_file



    def as_dict(self) -> Dict[str, Any]:
        return vars(self)

# class PreAmortizedConfig(EStepConfig):
#     def __init__(
#         self,
#         n_states: int,
#         model_path: str,
#         nsamples: int = 10,
#         K_init_file: str = None,
#     ):
#         """Configuration object for TVS E-step.
#
#         :param n_states: Number of variational states per datapoint to keep in memory.
#         :param n_prior_samples: Number of new variational states to be sampled from prior.
#         :param n_marginal_samples: Number of new variational states to be sampled from\
#                                    approximated marginal p(s_h=1|vec{y}^{(n)}, Theta).
#         :param K_init_file: Full path to H5 file providing initial states
#         """
#         model_path = os.path.expanduser(model_path)
#         assert n_states > 0, f"n_states must be positive integer ({n_states})"
#         assert(
#             nsamples > -1
#         ), f"nsamples must be positive integer ({nsamples})"
#
#         self.nsamples = nsamples
#         self.model_path = model_path
#         self.K_init_file = K_init_file
#
#         super().__init__(n_states)
#
#     def as_dict(self) -> Dict[str, Any]:
#         return vars(self)

class TVSConfig(EStepConfig):
    def __init__(
        self,
        n_states: int,
        n_prior_samples: int,
        n_marginal_samples: int,
        K_init_file: str = None,
    ):
        """Configuration object for TVS E-step.

        :param n_states: Number of variational states per datapoint to keep in memory.
        :param n_prior_samples: Number of new variational states to be sampled from prior.
        :param n_marginal_samples: Number of new variational states to be sampled from\
                                   approximated marginal p(s_h=1|vec{y}^{(n)}, Theta).
        :param K_init_file: Full path to H5 file providing initial states
        """
        assert n_states > 0, f"n_states must be positive integer ({n_states})"
        assert (
            n_prior_samples > -1
        ), f"n_prior_samples must be positive integer ({n_prior_samples})"
        assert (
            n_marginal_samples > -1
        ), f"n_marginal_samples must be positive integer ({n_marginal_samples})"

        assert n_marginal_samples + n_prior_samples > 0, 'total samples must be a natural number'
        self.n_prior_samples = n_prior_samples
        self.n_marginal_samples = n_marginal_samples
        self.K_init_file = K_init_file

        super().__init__(n_states)

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


class FullEMConfig(EStepConfig):
    def __init__(self, n_latents: int):
        """Full EM configuration."""
        super().__init__(2 ** n_latents)

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


class FullEMSingleCauseConfig(EStepConfig):
    def __init__(self, n_latents: int):
        """Full EM configuration."""
        super().__init__(n_latents)

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


class RandomSamplingConfig(EStepConfig):
    def __init__(
        self,
        n_states: int,
        n_samples: int,
        sparsity: float = 0.5,
        K_init_file: str = None,
    ):
        """Configuration object for random sampling.

        :param n_states: Number of variational states per datapoint to keep in memory.
        :param n_samples: Number of new variational states to randomly draw.
        :param sparsity: average fraction of active units in sampled states.
        :param K_init_file: Full path to H5 file providing initial states
        """
        assert n_states > 0, f"n_states must be positive integer ({n_states})"
        assert n_samples > 0, f"n_samples must be positive integer ({n_samples})"
        assert sparsity > 0 and sparsity < 1, f"sparsity must be in [0, 1] ({sparsity})"

        self.n_samples = n_samples
        self.sparsity = sparsity
        self.K_init_file = K_init_file

        super().__init__(n_states)

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)

class NeuralEvoConfig(NeuralEMConfig, EVOConfig):
    def __init__(
                    self,
                    encoder: str,
                    n_states: int,
                    n_samples: int,
                    input_size: int,
                    activations: Sequence,
                    output_size: int,
                    n_generations,
                    n_hidden: List[int] = None,
                    dropouts: List[bool] = None,
                    dropout_rate: float = None,
                    output_activation=None,
                    lr: float = None,
                    sampling: str = "Gumbel",
                    K_init=None,
                    loss_name: str = None,
                    n_parents=None,
                    n_children=None,
                    parent_selection: str = "fitness",
                    crossover: bool = True,
                    mutation: str = "uniform",
                    bitflip_frequency: float = None,
                    K_init_file: str = None,
                    **kwargs):

        super().__init__(n_states, n_parents,n_generations,
        parent_selection,
        crossover,
        n_children,
        mutation,
        bitflip_frequency,
        K_init_file,
        K_init,
        loss_name,
        encoder,
        sampling,
        output_activation,
        n_samples,
        lr,
        dropout_rate,
        dropouts,
        n_hidden,
        output_size,
        activations,
        input_size,
        **kwargs)

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)