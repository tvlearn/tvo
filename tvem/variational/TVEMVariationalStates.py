# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any

from tvem.variational._util import generate_unique_states
from tvem.util import get
import tvem


class TVEMVariationalStates(ABC):
    def __init__(self, conf: Dict[str, Any], K_init: Tensor = None):
        """Abstract base class for TVEM realizations.

        :param conf: dictionary with hyper-parameters. Required keys: N, H, S, dtype, device
        :param K_init: if specified, self.K will be initialized with this Tensor of shape (N,S,H)
        """
        required_keys = ("N", "H", "S", "precision")
        for c in required_keys:
            assert c in conf and conf[c] is not None
        self.conf = conf

        N, H, S, precision = get(conf, *required_keys)

        if K_init is not None:
            assert K_init.shape == (N, S, H)
            self.K = K_init.clone()
        else:
            self.K = generate_unique_states(S, H).repeat(N, 1, 1)  # (N, S, H)
        self.lpj = to.empty((N, S), dtype=precision, device=tvem.get_device())
        self.precision = precision

    @abstractmethod
    def update(
        self,
        idx: Tensor,
        batch: Tensor,
        lpj_fn: Callable[[Tensor, Tensor], Tensor],
        sort_by_lpj: Dict[str, Tensor] = {},
    ) -> int:
        """Generate new variational states, update K and lpj with best samples and their lpj.

        :param idx: data point indices of batch w.r.t. K
        :param batch: batch of data points
        :param lpj_fn: function to evaluate lpj
        :param sort_by_lpj: optional dictionary of tensors with shape (N,S,...) that will be\
            sorted by all_lpj, the same way all_lpj and all_states are sorted.
        :returns: average number of variational state substitutions per datapoint performed
        """
        pass  # pragma: no cover
