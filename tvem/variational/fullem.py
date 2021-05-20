# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from torch import Tensor

from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
import tvem
from tvem.utils.model_protocols import Trainable, Optimized


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
    def __init__(self, N: int, H: int, precision: to.dtype, K_init=None):
        """Full EM class.

        :param N: Number of datapoints
        :param H: Number of latent variables
        :param precision: The floating point precision of the lpj values.
                          Must be one of to.float32 or to.float64
        :param K_init: Optional initialization of states
        """
        if K_init is None:
            K_init = state_matrix(H)[None, :, :].expand(N, -1, -1)
            S = K_init.shape[1]
        else:  # Single cause EM
            S = H

        conf = dict(N=N, S=S, S_new=0, H=H, precision=precision)
        super().__init__(conf, K_init)

    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        lpj_fn = model.log_pseudo_joint if isinstance(model, Optimized) else model.log_joint

        K = self.K
        lpj = self.lpj

        lpj[idx] = lpj_fn(batch, K[idx])

        return 0


class FullEMSingleCauseModels(FullEM):
    def __init__(self, N: int, H: int, precision: to.dtype):
        """Full EM class for single causes models.

        :param N: Number of datapoints
        :param C: Number of latent variables
        :param precision: The floating point precision of the lpj values.
                          Must be one of to.float32 or to.float64
        """
        K_init = to.eye(H, dtype=precision, device=tvem.get_device())[None, :, :].expand(N, -1, -1)
        super().__init__(N, H, precision, K_init=K_init)

    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        lpj_fn = model.log_pseudo_joint if isinstance(model, Optimized) else model.log_joint
        assert to.any(self.K.sum(axis=1) == 1), "Multiple causes detected."
        K = self.K
        lpj = self.lpj

        lpj[idx] = lpj_fn(batch, K[idx])

        return 0

