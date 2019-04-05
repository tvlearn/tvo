# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from typing import Callable, Dict, Any
from torch import Tensor

from .TVEMVariationalStates import TVEMVariationalStates, update_states_for_batch


class RandomSampledVarStates(TVEMVariationalStates):
    def __init__(self, n_new_states: int, conf: Dict[str, Any], sparsity: float = 0.5):
        """A TVEMVariationalStates implementation that performs random sampling.

        :param n_new_states: number of states to be sampled at every call to ~update
        :param conf: dictionary with hyper-parameters. See\
          :func:`~tvem.variational.TVEMVariationalStates` for a list of required keys.
        :param sparsity: average fraction of active units in sampled states.
        """
        super().__init__(conf)
        self.n_new_states = n_new_states
        self.sparsity = sparsity

    def update(self, idx: Tensor, batch: Tensor,
               lpj_fn: Callable[[Tensor, Tensor], Tensor],
               sort_by_lpj: Dict[str, Tensor] = {}) -> int:
        """See :func:`TVEMVariationalStates.update <tvem.variational.TVEMVariationalStates.update>`.
        """
        K = self.K[idx]
        batch_size, S, H = K.shape
        self.lpj[idx] = lpj_fn(batch, K)
        new_K = to.rand(batch_size, self.n_new_states, H, device=K.device) < self.sparsity
        new_lpj = lpj_fn(batch, new_K)

        nsubs = update_states_for_batch(new_K, new_lpj, idx,
                                        self.K, self.lpj, sort_by_lpj=sort_by_lpj)

        self.lpj2pjc(idx)

        return nsubs
