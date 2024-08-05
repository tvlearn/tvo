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
        precision: to.dtype,
        S_new_prior: int,
        S_new_marg: int,
        K_init_file: str = None,
    ):
        """Amortized TVO sampling class.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        :param S_new_prior: number of states to be sampled from prior at every call to ~update
        :param S_new_marg: number of states to be sampled from approximated marginal\
                           p(s_h=1|vec{y}^{(n)}, Theta) at every call to ~update
        :param K_init_file: Full path to H5 file providing initial states
        """
        conf = {
            "N": N,
            "H": H,
            "S": S,
            "S_new_prior": S_new_prior,
            "S_new_marg": S_new_marg,
            "S_new": S_new_prior + S_new_marg,
            "precision": precision,
            "K_init_file": K_init_file,
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

        new_K = self._posterior_sampler(batch, self.config["N_posterior_samples"])
        new_lpj = lpj_fn(batch, new_K)

        set_redundant_lpj_to_low(new_K, new_lpj, K[idx])
        return update_states_for_batch(new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj)
