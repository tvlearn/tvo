# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import mean_posterior
from tvem.utils.model_protocols import Trainable, Optimized
from ._utils import update_states_for_batch, set_redundant_lpj_to_low


class TVSVariationalStates(TVEMVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        precision: to.dtype,
        S_new_prior: int,
        S_new_marg: int,
    ):
        """Truncated Variational Sampling class.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        :param S_new_prior: number of states to be sampled from prior at every call to ~update
        :param S_new_marg: number of states to be sampled from approximated marginal\
                           p(s_h=1|vec{y}^{(n)}, Theta) at every call to ~update
        """
        conf = {
            "N": N,
            "H": H,
            "S": S,
            "S_new_prior": S_new_prior,
            "S_new_marg": S_new_marg,
            "S_new": S_new_prior + S_new_marg,
            "precision": precision,
        }
        super().__init__(conf)

    def update(self, idx: to.Tensor, batch: to.Tensor, model: Trainable) -> int:
        """See :func:`tvem.variational.TVEMVariationalStates.update`."""
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
            to.rand(batch_size, self.config["S_new_prior"], H, device=K.device)
            < model.theta["pies"]
        ).byte()

        approximate_marginals = (
            mean_posterior(K[idx].type_as(lpj), lpj[idx])
            .unsqueeze(1)
            .expand(batch_size, self.config["S_new_marg"], H)
        )  # approximates p(s_h=1|\yVecN, \Theta), shape is (batch_size, S_new_marg, H)
        new_K_marg = (
            to.rand(batch_size, self.config["S_new_marg"], H, device=K.device)
            < approximate_marginals
        ).byte()

        new_K = to.cat((new_K_prior, new_K_marg), dim=1)

        new_lpj = lpj_fn(batch, new_K)

        set_redundant_lpj_to_low(new_K, new_lpj, K[idx])

        return update_states_for_batch(new_K, new_lpj, idx, K, lpj, sort_by_lpj=sort_by_lpj)
