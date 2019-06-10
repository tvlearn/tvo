# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from torch import Tensor
from typing import Callable, Dict

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

    all_states = to.empty((2 ** H, H), dtype=to.uint8, device=device)
    for state in range(2 ** H):
        bit_sequence = tuple(int(bit) for bit in f"{state:0{H}b}")
        all_states[state] = to.tensor(bit_sequence, dtype=to.uint8, device=device)
    return all_states


class FullEM(TVEMVariationalStates):
    def __init__(self, N: int, H: int, precision: to.dtype):
        """Full EM class.

        :param N: Number of datapoints
        :param H: Number of latent variables
        :param precision: The floating point precision of the lpj values.
                          Must be one of to.float32 or to.float64
        """
        conf = dict(N=N, S=2 ** H, H=H, precision=precision)
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
