# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC, abstractmethod
from typing import Dict, Any


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
        assert parent_selection in valid_selections, f"Unknown parent selection {parent_selection}"
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
        assert n_prior_samples > 0, f"n_prior_samples must be positive integer ({n_prior_samples})"
        assert (
            n_marginal_samples > 0
        ), f"n_marginal_samples must be positive integer ({n_marginal_samples})"

        self.n_prior_samples = n_prior_samples
        self.n_marginal_samples = n_marginal_samples
        self.K_init_file = K_init_file

        super().__init__(n_states)

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)


class FullEMConfig(EStepConfig):
    def __init__(self, n_latents: int):
        """Full EM configuration."""
        super().__init__(2**n_latents)

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
        self, n_states: int, n_samples: int, sparsity: float = 0.5, K_init_file: str = None
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
