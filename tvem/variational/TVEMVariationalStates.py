# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch as to

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
from torch import Tensor

from tvem.util import get
import tvem


def unique_ind(x: Tensor, dim: int = None) -> Tensor:
    """Find indices of unique elements in x along specific dimension.

    :param x: torch tensor
    :param dim: dimension to apply unique
    """
    unique, inverse = to.unique(x, sorted=False, return_inverse=True, dim=dim)
    perm = to.arange(inverse.size(0), device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


def generate_unique_states(n_states: int, H: int, crowdedness: float = 1.,
                           device: to.device = None) -> Tensor:
    """Generate a torch tensor containing random and unique binary vectors.

    :param n_states: number of unique vectors to be generated
    :param H: size of binary vector
    :param crowdedness: average crowdedness per state
    :param device: torch.device of output Tensor. Defaults to tvem.get_device()

    Requires that n_states <= 2**H. Return has shape (n_states, H).
    """
    if device is None:
        device = tvem.get_device()
    assert n_states <= 2**H, "n_states must be smaller than 2**H"
    s_set = {tuple(s) for s in np.random.binomial(1, p=crowdedness/H, size=(n_states//2, H))}
    while len(s_set) < n_states:
        s_set.update({tuple(s) for s in np.random.binomial(
            1, p=crowdedness/H, size=(n_states//2, H))})
    while len(s_set) > n_states:
        s_set.pop()
    return to.from_numpy(np.array(tuple(s for s in s_set), dtype=int)).to(
        dtype=to.uint8, device=device)


def update_states_for_batch(new_states: Tensor, new_lpj: Tensor, idx: Tensor,
                            all_states: Tensor, all_lpj: Tensor,
                            sort_by_lpj: Dict[str, Tensor] = {}) -> int:
    """Perform substitution of old and new states (and lpj, ...)
       according to TVEM criterion.

    :param new_states: set of new variational states (idx.size, newS, H)
    :param new_lpj: corresponding log-pseudo-joints (idx.size, newS)
    :param idx: indeces of the datapoints that compose the batch within the dataset
    :param all_states: set of all variational states (N, S, H)
    :param all_lpj: corresponding log-pseudo-joints (N, S)
    :param sort_by_lpj: optional list of tensors with shape (n,s,...) that will be
        sorted by all_lpj, the same way all_lpj and all_states are sorted.

    S is the number of variational states memorized for each of the N
    data-points. idx contains the ordered list of indexes for which the
    new_states have been evaluated (i.e. the states in new_states[0] are to
    be put into all_s[idx[0]]. all_s[n] is updated to contain the set of
    variational states with best log-pseudo-joints.

    TODO Find out why lpj precision decreases for states without substitutions
    (difference on the order of 1e-15).
    """

    S = all_states.shape[1]
    batch_size, newS, H = new_states.shape

    old_states = all_states[idx]
    old_lpj = all_lpj[idx]

    assert old_states.shape == (batch_size, S, H)
    assert old_lpj.shape == (batch_size, S)

    conc_states = to.cat((old_states, new_states), dim=1)
    conc_lpj = to.cat((old_lpj, new_lpj), dim=1)  # (batch_size, S+newS)

    sorted_idx = to.flip(to.topk(conc_lpj, k=S, dim=1, largest=True,
                                 sorted=True)[1], [1])  # is (batch_size, S)
    flattened_sorted_idx = sorted_idx.flatten()

    idx_n = idx.repeat(S, 1).t().flatten()
    idx_s = to.arange(S, device=all_states.device).repeat(batch_size)
    idx_sc = to.arange(batch_size, device=all_states.device).repeat(
        S, 1).t().flatten()

    all_states[idx_n, idx_s] = conc_states[idx_sc, flattened_sorted_idx]
    all_lpj[idx_n, idx_s] = conc_lpj[idx_sc, flattened_sorted_idx]

    for t in sort_by_lpj.values():
        idx_n_ = to.arange(t.shape[0]).repeat(S, 1).t().flatten()
        t[idx_n_, idx_s] = t[idx_n_, flattened_sorted_idx]

    return (sorted_idx >= old_states.shape[1]).sum().item()  # nsubs


def set_redundant_lpj_to_low(new_states: Tensor, new_lpj: Tensor,
                             old_states: Tensor):
    """Find redundant states in new_states w.r.t. old_states and set
       corresponding lpg to low.

    :param new_states: set of new variational states (batch_size, newS, H)
    :param new_lpj: corresponding log-pseudo-joints (batch_size, newS)
    :param old_states: (batch_size, S, H)
    """

    N, S, H = old_states.shape
    newS = new_states.shape[1]

    # old_states must come first for np.unique to discard redundant new_states
    # TODO Check if still holds for to.unique
    old_and_new = to.cat((old_states, new_states), dim=1)
    for n in range(N):
        uniq_idx = unique_ind(old_and_new[n], dim=0)
        # indexes of states in new_states[n] that are not in old_states[n]
        new_uniq_idx = uniq_idx[uniq_idx >= S] - S
        mask = to.ones(newS, dtype=to.uint8, device=new_lpj.device)
        # indexes of all non-unique states in new_states (complementary of
        # new_uniq_idx)
        mask[new_uniq_idx.to(device=new_lpj.device)] = 0
        # set lpj of redundant states to an arbitrary low value
        new_lpj[n][mask] = to.tensor(
            [-1e100], dtype=new_lpj.dtype, device=new_lpj.device)


class TVEMVariationalStates(ABC):
    def __init__(self, conf: Dict[str, Any], K_init: Tensor = None):
        """Abstract base class for TVEM realizations.

        :param conf: dictionary with hyper-parameters. Required keys: N, H, S, dtype, device
        :param K_init: if specified, self.K will be initialized with this Tensor of shape (N,S,H)
        """
        required_keys = ('N', 'H', 'S', 'dtype')
        for c in required_keys:
            assert c in conf and conf[c] is not None
        self.conf = conf

        N, H, S, dtype = get(conf, *required_keys)

        if K_init is not None:
            assert K_init.shape == (N, S, H)
            self.K = K_init.clone()
        else:
            self.K = generate_unique_states(S, H).repeat(N, 1, 1)  # (N, S, H)
        self.lpj = to.empty((N, S), dtype=dtype, device=tvem.get_device())

    @abstractmethod
    def update(self, idx: Tensor, batch: Tensor,
               lpj_fn: Callable[[Tensor, Tensor], Tensor],
               sort_by_lpj: Dict[str, Tensor] = {}) -> int:
        """Generate new variational states, update K and lpj with best samples and their lpj.

        :param idx: data point indices of batch w.r.t. K
        :param batch: batch of data points
        :param lpj_fn: function to evaluate lpj
        :param sort_by_lpj: optional dictionary of tensors with shape (N,S,...) that will be\
            sorted by all_lpj, the same way all_lpj and all_states are sorted.
        :returns: average number of variational state substitutions per datapoint performed
        """
        pass  # pragma: no cover
