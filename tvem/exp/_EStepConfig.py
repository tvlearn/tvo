# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC


class EStepConfig(ABC):
    def __init__(self, n_states: int):
        """Abstract base configuration object for experiments' E-steps.

        :param n_states: Number of variational states per datapoint to keep in memory.
        """
        self.n_states = n_states


class EEMConfig(EStepConfig):
    def __init__(
        self,
        n_states: int,
        n_parents: int,
        n_children: int,
        n_generations: int,
        parent_selection: str = "fitness",
        crossover: bool = True,
        mutation: str = "uniform",
        bitflip_frequency: float = None,
    ):
        """Configuration object for EEM E-step.

        :param n_states: Number of variational states per datapoint to keep in memory.
        :param n_parents: Number of parent states to select at each EEM generation.
                          Must be <= n_states.
        :param n_children: Number of children per parent to generate via mutation
                           at each EEM generation.
        :param parent_selection: Parent selection algorithm for EEM. Must be one of:
                                 - 'fitness': fitness-proportional parent selection
        :param crossover: Whether crossover should be applied or not.
        :param mutation: Mutation algorithm for EEM. Must be one of:
                         - 'sparsity': bits are flipped so that states tend
                            towards current model sparsity.
                         - 'uniform': random uniform selection of bits to flip.
        :param bitflip_frequency: Probability of flipping a bit during the mutation step (e.g.
                                  2/H for an average of 2 bitflips per mutation). Required when
                                  using the 'sparsity' mutation algorithm.
        """
        valid_selections = ("fitness",)
        assert parent_selection in valid_selections, f"Unknown parent selection {parent_selection}"
        valid_mutations = ("sparsity", "uniform")
        assert mutation in valid_mutations, f"Unknown mutation {mutation}"
        assert (
            n_parents <= n_states
        ), f"n_parents ({n_parents}) must be lower than n_states ({n_states})"
        assert (
            mutation != "sparsity" or bitflip_frequency is not None
        ), "bitflip_frequency is required for mutation algorithm 'sparsity'"

        self.n_new_states = (
            n_parents * (n_parents - 1) * n_children * n_generations
            if crossover
            else n_parents * n_children * n_generations
        )
        self.n_parents = n_parents
        self.n_children = n_children
        self.n_generations = n_generations
        self.parent_selection = parent_selection
        self.crossover = crossover
        self.mutation = mutation
        self.bitflip_frequency = bitflip_frequency

        super().__init__(n_states)


class FullEMConfig(EStepConfig):
    def __init__(self, n_latents: int):
        """Full EM configuration."""
        self.n_new_states = 0

        super().__init__(2 ** n_latents)
