# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from torch import Tensor

import tvo
from tvo.utils.model_protocols import Trainable, Optimized
from tvo.variational.TVOVariationalStates import TVOVariationalStates


def state_matrix(H: int, device: to.device = None):
    """Get full combinatorics of H-dimensional binary vecor.

    :param H: vector length
    :device: torch.device of output Tensor. Defaults to tvo.get_device().
    :returns: tensor containing full combinatorics, shape (2**H,H)
    """
    if device is None:
        device = tvo.get_device()

    all_states = to.empty((2 ** H, H), dtype=to.uint8, device=device)
    for state in range(2 ** H):
        bit_sequence = tuple(int(bit) for bit in f"{state:0{H}b}")
        all_states[state] = to.tensor(bit_sequence, dtype=to.uint8, device=device)
    return all_states


class FullEM(TVOVariationalStates):
    def __init__(self, N: int, H: int, precision: to.dtype, K_init=None):
        """Full EM class.

        :param N: Number of datapoints
        :param H: Number of latent variables
        :param precision: The floating point precision of the lpj values.
                          Must be one of to.float32 or to.float64
        :param K_init: Optional initialization of states
        """
        conf = dict(N=N, S=None, S_new=None, H=H, precision=precision)
        required_keys = ("N", "H", "precision")
        for c in required_keys:
            assert c in conf and conf[c] is not None
        self.config = conf
        self.lpj = to.empty((N, 2 ** H), dtype=precision, device=tvo.get_device())
        self.precision = precision
        self.K = state_matrix(H)[None, :, :].expand(N, -1, -1)

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
        conf = dict(N=N, S=None, S_new=None, H=H, precision=precision)
        required_keys = ("N", "H", "precision")
        for c in required_keys:
            assert c in conf and conf[c] is not None
        self.config = conf
        self.lpj = to.empty((N, H), dtype=precision, device=tvo.get_device())
        self.precision = precision
        self.K = to.eye(H, dtype=precision, device=tvo.get_device())[None, :, :].expand(N, -1, -1)

    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        assert to.any(self.K.sum(axis=1) == 1), "Multiple causes detected."
        super(FullEMSingleCauseModels, self).update(idx, batch, model)
        return 0
