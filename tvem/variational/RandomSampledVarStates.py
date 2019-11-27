# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from typing import Dict, Any, TYPE_CHECKING
from torch import Tensor

from ._utils import update_states_for_batch
from .TVEMVariationalStates import TVEMVariationalStates

if TYPE_CHECKING:
    from tvem.models.TVEMModel import TVEMModel


class RandomSampledVarStates(TVEMVariationalStates):
    def __init__(self, S_new: int, conf: Dict[str, Any], sparsity: float = 0.5):
        """A TVEMVariationalStates implementation that performs random sampling.

        :param S_new: number of states to be sampled at every call to ~update
        :param conf: dictionary with hyper-parameters. See\
          :func:`~tvem.variational.TVEMVariationalStates` for a list of required keys.
        :param sparsity: average fraction of active units in sampled states.
        """
        conf["S_new"] = S_new
        super().__init__(conf)
        self.sparsity = sparsity

    def update(self, idx: Tensor, batch: Tensor, model: "TVEMModel") -> int:
        """See :func:`TVEMVariationalStates.update <tvem.variational.TVEMVariationalStates.update>`.
        """
        lpj_fn = (
            model.log_joint if model.log_pseudo_joint is NotImplemented else model.log_pseudo_joint
        )
        sort_by_lpj = model.sorted_by_lpj
        K = self.K[idx]
        batch_size, S, H = K.shape
        self.lpj[idx] = lpj_fn(batch, K)
        new_K = (
            to.rand(batch_size, self.config["S_new"], H, device=K.device) < self.sparsity
        ).byte()
        new_lpj = lpj_fn(batch, new_K)

        return update_states_for_batch(
            new_K, new_lpj, idx, self.K, self.lpj, sort_by_lpj=sort_by_lpj
        )
