# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch as to

from abc import ABC, abstractmethod
from itertools import combinations
from typing import Dict, Callable
from torch import Tensor


def state_matrix(H: int, device: to.device = to.device('cpu')) -> Tensor:
    """Generate matrix containing full combinatorics of an H-dimensional
       binary variable.

    H -- length of binary vector
    device -- default is CPU
    """
    sl = []
    for g in range(0, H+1):
        for s in combinations(range(H), g):
            sl.append(to.tensor(s, dtype=to.int64))
    SM = to.zeros((len(sl), H), dtype=to.uint8, device=device)
    for i in range(len(sl)):
        s = sl[i]
        SM[i, s] = 1
    return SM


def unique_ind(x: Tensor, dim: int = None) -> Tensor:
    """Find indices of unique elements in x along specific dimension.

    x -- torch tensor
    dim -- dimension to apply unique
    """
    unique, inverse = to.unique(x, sorted=False, return_inverse=True, dim=dim)
    perm = to.arange(inverse.size(0), device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


def generate_unique_states(n_states: int, H: int, device: to.device =
                           to.device('cpu')) -> Tensor:
    """Generate a torch tensor containing random and unique binary vectors.

    n_states -- number of unique vectors to be generated
    H -- size of binary vector
    device -- default is CPU

    Requires that n_states <= 2**H. Return has shape (n_states, H).
    """
    assert n_states <= 2**H, "n_states must be smaller than 2**H"
    s_set = {tuple(s) for s in np.random.randint(2, size=(n_states*2, H))}
    while len(s_set) < n_states:
        s_set.update({tuple(s) for s in np.random.binomial(
            1, p=1./H, size=(n_states//2, H))})
    while len(s_set) > n_states:
        s_set.pop()
    return to.from_numpy(np.array(tuple(s for s in s_set), dtype=int)).to(
        dtype=to.uint8, device=device)


def update_states_for_batch(new_states: Tensor, new_lpj: Tensor, idx: Tensor,
                            all_states: Tensor, all_lpj: Tensor,
                            mstep_factors: Dict[str, Tensor] = None) -> int:
    """Perform substitution of old and new states (and lpj, ...)
       according to TVEM criterion.

    new_states -- set of new variational states (idx.size, newS, H)
    new_lpj -- corresponding log-pseudo-joints (idx.size, newS)
    idx -- indeces of the datapoints that compose the batch within the dataset
    all_states -- set of all variational states (N, S, H)
    all_lpj -- corresponding log-pseudo-joints (N, S)

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

    if mstep_factors is not None:
        for key in mstep_factors:
            mstep_factors[key][idx_n, idx_s] = mstep_factors[key]
            [idx_n, flattened_sorted_idx]

    return (sorted_idx >= old_states.shape[1]).sum().item()  # nsubs


def set_redundant_lpj_to_low(new_states: Tensor, new_lpj: Tensor,
                             old_states: Tensor):
    """Find redundant states in new_states w.r.t. old_states and set
       corresponding lpg to low.

    new_states -- set of new variational states (batch_size, newS, H)
    new_lpj -- corresponding log-pseudo-joints (batch_size, newS)
    old_states -- (batch_size, S, H)
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
    """Abstract base class for TVEM realizations."""

    def __init__(self, conf: Dict):
        """Construct a TVEM realization.

        conf -- dictionary with hyper-parameters
        """
        for c in ['my_N', 'H', 'S', 'dtype_f', 'device']:
            assert c in conf and c is not None
        self.conf = conf

        self.K = generate_unique_states(conf['S'], conf['H'],
                                        device=conf['device']).repeat(
            conf['my_N'], 1, 1)  # (N, S, H)
        self.lpj = to.empty((conf['my_N'], conf['S']),
                            dtype=conf['dtype_f'], device=conf['device'])

    @abstractmethod
    def update(self, idx: Tensor, batch: Tensor,
               lpj_fn: Callable[[Tensor, Tensor, Dict], Tensor],
               mstep_factors: Dict[str, Tensor]) -> int:
        """ Evaluate lpj of old states, generate new states and return states
        with highest lpj.

        idx -- data point indices of batch w.r.t. K
        batch -- batch of data points
        lpj_fn -- function to evaluate lpj
        mstep_factors -- optional dictionary containing tensors involved
                         in M-step
        """
        pass
