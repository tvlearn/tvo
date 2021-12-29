# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch import Tensor
from tvem.utils.model_protocols import Trainable, Optimized

from ._utils import update_states_for_batch
from .TVEMVariationalStates import TVEMVariationalStates


class RandomSampledVarStates(TVEMVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        precision: to.dtype,
        S_new: int,
        sparsity: float = 0.5,
        K_init_file: str = None,
    ):
        """A TVEMVariationalStates implementation that performs random sampling.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        :param S_new: number of states to be sampled at every call to ~update
        :param sparsity: average fraction of active units in sampled states.
        :param K_init_file: Full path to H5 file providing initial states
        """
        conf = dict(
            N=N,
            H=H,
            S=S,
            precision=precision,
            S_new=S_new,
            sparsity=sparsity,
            K_init_file=K_init_file,
        )
        super().__init__(conf)

    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        """See :func:`tvem.variational.TVEMVariationalStates.update`."""
        if isinstance(model, Optimized):
            lpj_fn = model.log_pseudo_joint
            sort_by_lpj = model.sorted_by_lpj
        else:
            lpj_fn = model.log_joint
            sort_by_lpj = {}
        K = self.K[idx]
        batch_size, S, H = K.shape
        self.lpj[idx] = lpj_fn(batch, K)
        new_K = (
            to.rand(batch_size, self.config["S_new"], H, device=K.device) < self.config["sparsity"]
        ).byte()
        new_lpj = lpj_fn(batch, new_K)

        return update_states_for_batch(
            new_K, new_lpj, idx, self.K, self.lpj, sort_by_lpj=sort_by_lpj
        )
