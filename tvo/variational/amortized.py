# -*- coding: utf-8 -*-
# Copyright (C) 2024 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from tvo.variational.TVOVariationalStates import TVOVariationalStates
from tvo.variational._utils import mean_posterior
from tvo.utils.model_protocols import Trainable, Optimized
from ._utils import update_states_for_batch, set_redundant_lpj_to_low


class AmortizedTVOStates(TVOVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        S_new: int,
        precision: to.dtype,
    ):
        """Amortized TVO sampling class.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param S_new: number of states to be sampled at every call to ~update
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        """
        conf = {
            "N": N,
            "H": H,
            "S": S,
            "S_new": S_new,
            "precision": precision,
        }
        super().__init__(conf)
        self._posterior_sampler = None  # amortized posterior sampler


    def set_posterior_sampler(self, sampler):
        assert isinstance(sampler, to.nn.Module)
        self._posterior_sampler = sampler


    def update(self, idx: to.Tensor, batch: to.Tensor, model: Trainable) -> int:
        """Generate new variational states, update K and lpj with best samples and their lpj.

        :param idx: data point indices of batch w.r.t. K
        :param batch: batch of data points
        :param model: the model being used
        :returns: average number of variational state substitutions per datapoint performed
        """
        assert self._posterior_sampler is not None

        if isinstance(model, Optimized):
            lpj_fn = model.log_pseudo_joint
            sort_by_lpj = model.sorted_by_lpj
        else:
            lpj_fn = model.log_joint
            sort_by_lpj = {}

        #batch_size, H = batch.shape[0], K.shape[2]
        K, lpj = self.K, self.lpj
        lpj[idx] = lpj_fn(batch, K[idx])

        new_K = self._posterior_sampler.sample_q(batch, nsamples=self.config["S_new"])
        new_K = new_K.permute(1, 0, 2).to(self.K.dtype)
        new_lpj = lpj_fn(batch, new_K)

        set_redundant_lpj_to_low(new_K, new_lpj, K[idx])
        return update_states_for_batch(new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj)
