# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0


import torch as to
import math
from torch.distributions.one_hot_categorical import OneHotCategorical

from torch import Tensor
from typing import Union, Tuple

import tvem
from tvem.utils.parallel import pprint, all_reduce, broadcast
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import mean_posterior, mean_tempered_posterior, lpj2pjc
from tvem.utils.model_protocols import Optimized, Sampler, Reconstructor
from tvem.utils.sanity import fix_theta
from tvem.models import FULL_GMM

class REM1_FULL_GMM(FULL_GMM):
    def __init__(
        self,
        H: int,
        D: int,
        W_init: Tensor = None,
        Sigma_init: Tensor = None,
        pies_init: Tensor = None,
        precision: to.dtype = to.float64,
    ):
        super().__init__(H=H, D=D, W_init=W_init, Sigma_init=Sigma_init, pies_init=pies_init, precision=precision)
    
    def update_param_batch(self, idx: Tensor, batch: Tensor, states: Tensor, beta: float) -> None:
        lpj = states.lpj[idx]
        K = states.K[idx]
        batch_size, S, _ = K.shape

        Kfloat = K.to(dtype=lpj.dtype)  # TODO Find solution to avoid byte->float casting
        Wbar = to.matmul(
            Kfloat, self.theta["W"].t()
        )  # N,S,D # TODO Find solution to re-use evaluations from E-step

        batch_s_pjc = mean_tempered_posterior(Kfloat, lpj, beta)  # is (batch_size,H) mean_posterior(Kfloat, lpj) 
        # basically responsibilitys: r^(n)_c
        batch_Wp = batch.unsqueeze(2) * batch_s_pjc.unsqueeze(1)  # is (batch_size,D,H)
        data_covariance = (batch[:,None,:] - Wbar).unsqueeze(3) @ (batch[:,None,:] - Wbar).unsqueeze(2) # is batch_size, S, D, D
        # batch_Sigma = to.einsum('nsde, ns->nsde',data_covariance, batch_s_pjc) # is batch_size, D, D the same as next line
        batch_Sigma = data_covariance * batch_s_pjc.unsqueeze(2).unsqueeze(2)


        # batch_Sigma = mean_posterior(to.einsum('nsd, nse-> nsde', batch[:, None, :] - Wbar, batch[:, None, :] - Wbar) , lpj)

        self.my_pies.add_(to.sum(batch_s_pjc, dim=0))
        self.my_Wp.add_(to.sum(batch_Wp, dim=0))
        self.my_Wq.add_(to.sum(batch_s_pjc, dim=0)) # sum(r(n,h),dim=0)
        self.my_Sigma.add_(to.sum(batch_Sigma, dim=0).permute(1,2,0))#my_Sigma is D,D,H; batch_Sigma is N,H,D,D
        self.my_N.add_(batch_size)

        return None
    
    def annealed_free_energy(
            self, idx: to.Tensor, batch: to.Tensor, states: "TVEMVariationalStates", beta: float = 1
    ) -> float:
        """Evaluate free energy for the given batch of datapoints.

        :param idx: indexes of the datapoints in batch within the full dataset
        :param batch: batch of datapoints, Tensor with shape (N,D)
        :param states: all TVEMVariationalStates states for this dataset

        .. note::
        This default implementation of free_energy is only appropriate for Optimized models
        """
        with to.no_grad():
            log_joints = self.log_joint(batch, states.K[idx], states.lpj[idx])
            lpj = states.lpj[idx]
            post = lpj2pjc(beta * lpj)
            F = beta * mean_tempered_posterior(log_joints, lpj, beta) - mean_tempered_posterior(to.log(post), lpj, beta)
            res = to.sum(F).item()
        return res

