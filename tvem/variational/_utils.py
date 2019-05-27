import torch as to
import tvem
import numpy as np
from typing import Dict


def unique_ind(x: to.Tensor, dim: int = None) -> to.Tensor:
    """Find indices of unique elements in x along specific dimension.

    :param x: torch tensor
    :param dim: dimension to apply unique
    """
    unique, inverse = to.unique(x, sorted=False, return_inverse=True, dim=dim)
    perm = to.arange(inverse.size(0), device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    return inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)


def generate_unique_states(
    n_states: int, H: int, crowdedness: float = 1.0, device: to.device = None
) -> to.Tensor:
    """Generate a torch tensor containing random and unique binary vectors.

    :param n_states: number of unique vectors to be generated
    :param H: size of binary vector
    :param crowdedness: average crowdedness per state
    :param device: torch.device of output Tensor. Defaults to tvem.get_device()

    Requires that n_states <= 2**H. Return has shape (n_states, H).
    """
    if device is None:
        device = tvem.get_device()
    assert n_states <= 2 ** H, "n_states must be smaller than 2**H"
    s_set = {tuple(s) for s in np.random.binomial(1, p=crowdedness / H, size=(n_states // 2, H))}
    while len(s_set) < n_states:
        s_set.update(
            {tuple(s) for s in np.random.binomial(1, p=crowdedness / H, size=(n_states // 2, H))}
        )
    while len(s_set) > n_states:
        s_set.pop()
    return to.from_numpy(np.array(tuple(s for s in s_set), dtype=int)).to(
        dtype=to.uint8, device=device
    )


def update_states_for_batch(
    new_states: to.Tensor,
    new_lpj: to.Tensor,
    idx: to.Tensor,
    all_states: to.Tensor,
    all_lpj: to.Tensor,
    sort_by_lpj: Dict[str, to.Tensor] = {},
) -> int:
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
    """
    # TODO Find out why lpj precision decreases for states without substitutions
    # (difference on the order of 1e-15).

    S = all_states.shape[1]
    batch_size, newS, H = new_states.shape

    old_states = all_states[idx]
    old_lpj = all_lpj[idx]

    assert old_states.shape == (batch_size, S, H)
    assert old_lpj.shape == (batch_size, S)

    conc_states = to.cat((old_states, new_states), dim=1)
    conc_lpj = to.cat((old_lpj, new_lpj), dim=1)  # (batch_size, S+newS)

    # is (batch_size, S)
    sorted_idx = to.flip(to.topk(conc_lpj, k=S, dim=1, largest=True, sorted=True)[1], [1])
    flattened_sorted_idx = sorted_idx.flatten()

    idx_n = idx.repeat(S, 1).t().flatten()
    idx_s = to.arange(S, device=all_states.device).repeat(batch_size)
    idx_sc = to.arange(batch_size, device=all_states.device).repeat(S, 1).t().flatten()

    all_states[idx_n, idx_s] = conc_states[idx_sc, flattened_sorted_idx]
    all_lpj[idx_n, idx_s] = conc_lpj[idx_sc, flattened_sorted_idx]

    for t in sort_by_lpj.values():
        idx_n_ = to.arange(batch_size).repeat(S, 1).t().flatten()
        t[idx_n_, idx_s] = t[idx_n_, flattened_sorted_idx]

    return (sorted_idx >= old_states.shape[1]).sum().item()  # nsubs


def set_redundant_lpj_to_low(new_states: to.Tensor, new_lpj: to.Tensor, old_states: to.Tensor):
    """Find redundant states in new_states w.r.t. old_states and set
       corresponding lpg to low.

    :param new_states: set of new variational states (batch_size, newS, H)
    :param new_lpj: corresponding log-pseudo-joints (batch_size, newS)
    :param old_states: (batch_size, S, H)
    """

    N, S, H = old_states.shape
    newS = new_states.shape[1]

    # old_states must come first for np.unique to discard redundant new_states
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
        new_lpj[n][mask] = to.tensor([-1e100], dtype=new_lpj.dtype, device=new_lpj.device)


def _lpj2pjc(lpj: to.Tensor):
    """Shift log-pseudo-joint and convert log- to actual probability

    :param lpj: log-pseudo-joint tensor
    :returns: probability tensor
    """
    up_lpg_bound = 0.0
    shft = up_lpg_bound - lpj.max(dim=1)[0]  # TODO could use keepdim
    tmp = to.exp(lpj + shft[:, None])
    return tmp / tmp.sum(dim=1)[:, None]


def _mean_post_einsum(g: to.Tensor, lpj: to.Tensor) -> to.Tensor:
    """Compute expectation value of g(s) w.r.t truncated variational distribution q(s).

    :param g: Values of g(s) with shape (N,S,...).
    :param lpj: Log-pseudo-joint with shape (N,S).
    :returns: tensor with shape (N,...).
    """
    return to.einsum("ns...,ns->n...", (g, _lpj2pjc(lpj)))


def _mean_post_mul(g: to.Tensor, lpj: to.Tensor) -> to.Tensor:
    """Compute expectation value of g(s) w.r.t truncated variational distribution q(s).

    :param g: Values of g(s) with shape (N,S,...).
    :param lpj: Log-pseudo-joint with shape (N,S).
    :returns: tensor with shape (N,...).
    """
    # reshape lpj from (N,S) to (N,S,1,...), to match dimensionality of g
    lpj = lpj.view(*lpj.shape, *(1 for _ in range(g.ndimension() - 2)))
    return _lpj2pjc(lpj).mul(g).sum(dim=1)


def mean_posterior(g: to.Tensor, lpj: to.Tensor) -> to.Tensor:
    """Compute expectation value of g(s) w.r.t truncated variational distribution q(s).

    :param g: Values of g(s) with shape (N,S,...).
    :param lpj: Log-pseudo-joint with shape (N,S).
    :returns: tensor with shape (N,...).
    """
    if tvem.get_device().type == "cpu":
        means = _mean_post_einsum(g, lpj)
    else:
        means = _mean_post_mul(g, lpj)

    assert means.shape == (g.shape[0], *g.shape[2:])
    assert not to.isnan(means).any() and not to.isinf(means).any()
    return means
