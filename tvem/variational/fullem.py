# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from torch import Tensor
from itertools import combinations
from typing import Callable, Dict, Any

from tvem.util import get
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
import tvem


def state_matrix(H: int, device: to.device = None):
    """Get full combinatorics of H-dimensional binary vecor.

    :param H: vector length
    :device: torch.device of output Tensor. Defaults to tvem.get_device().
    :returns: tensor containing full combinatorics, shape (2**H,H)
    """
    if device is None:
        device = tvem.get_device()
    sl = []
    for g in range(0, H + 1):
        for s in combinations(range(H), g):
            sl.append(to.tensor(s, dtype=to.int64))
    SM = to.zeros((len(sl), H), dtype=to.uint8, device=device)
    for i in range(len(sl)):
        s = sl[i]
        SM[i, s] = 1
    return SM


class FullEM(TVEMVariationalStates):
    def __init__(self, conf: Dict[str, Any]):
        """Full EM class.

        :param conf: dictionary with hyper-parameters
        """
        required_keys = ("N", "H")
        for c in required_keys:
            assert c in conf and conf[c] is not None
        N, H = get(conf, *required_keys)

        super().__init__(conf, state_matrix(H)[None, :, :].expand(N, -1, -1))

    def update(
        self,
        idx: Tensor,
        batch: Tensor,
        lpj_fn: Callable[[Tensor, Tensor], Tensor],
        sort_by_lpj: Dict[str, Tensor] = {},
    ) -> int:

        K = self.K
        lpj = self.lpj

        lpj[idx] = lpj_fn(batch, K[idx])

        return 0
