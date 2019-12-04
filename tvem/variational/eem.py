# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import numpy as np
import torch as to

from itertools import combinations
from typing import Callable, Tuple, Optional, TYPE_CHECKING
from torch import Tensor

import tvem
from tvem.utils import get
from tvem.variational.TVEMVariationalStates import TVEMVariationalStates
from tvem.variational._utils import update_states_for_batch, set_redundant_lpj_to_low

if TYPE_CHECKING:
    from tvem.models.TVEMModel import TVEMModel


class EEMVariationalStates(TVEMVariationalStates):
    def __init__(
        self,
        N: int,
        H: int,
        S: int,
        precision: to.dtype,
        parent_selection: str,
        mutation: str,
        n_parents: int,
        n_generations: int,
        n_children: int = None,
        crossover: bool = False,
        bitflip_frequency: float = None,
    ):
        """Evolutionary Expectation Maximization class.

        :param N: number of datapoints
        :param H: number of latents
        :param S: number of variational states
        :param precision: floating point precision to be used for log_joint values.
                          Must be one of to.float32 or to.float64.
        :param selection: one of "batch_fitparents" or "randparents"
        :param mutation: one of "randflip" or "sparseflip"
        :param n_parents: number of parent states to select
        :param n_generations: number of EA generations to produce
        :param n_children: if crossover is False, number of children states to produce per
                           generation. Must be None if crossover is True.
        :param crossover: if True, apply crossover. Must be False if n_children is specified.
        :param bitflip_frequency: Probability of flipping a bit during the mutation step (e.g.
                                  2/H for an average of 2 bitflips per mutation). Required when
                                  using the 'sparsity' mutation algorithm.
        """
        assert (
            not crossover or n_children is None
        ), "Exactly one of n_children and crossover may be provided."
        if crossover:
            mutation = f"cross_{mutation}"
            n_children = n_parents - 1
        assert n_children is not None  # make mypy happy
        S_new = get_n_new_states(mutation, n_parents, n_children, n_generations)

        conf = dict(
            N=N,
            H=H,
            S=S,
            S_new=S_new,
            precision=precision,
            parent_selection=parent_selection,
            mutation=mutation,
            n_parents=n_parents,
            n_children=n_children,
            n_generations=n_generations,
            p_bf=bitflip_frequency,
        )
        super().__init__(conf)

    def update(self, idx: Tensor, batch: Tensor, model: "TVEMModel") -> int:

        lpj_fn = (
            model.log_joint if model.log_pseudo_joint is NotImplemented else model.log_pseudo_joint
        )
        sort_by_lpj = model.sorted_by_lpj
        K = self.K
        lpj = self.lpj

        parent_selection, mutation, n_parents, n_children, n_generations = get(
            self.config, "parent_selection", "mutation", "n_parents", "n_children", "n_generations"
        )

        lpj[idx] = lpj_fn(batch, K[idx])

        def lpj_fn_(states):
            return lpj_fn(batch, states)

        new_states, new_lpj = evolve_states(
            lpj=lpj[idx].to(device="cpu"),
            states=K[idx].to(device="cpu"),
            lpj_fn=lpj_fn_,
            n_parents=n_parents,
            n_children=n_children,
            n_generations=n_generations,
            parent_selection=parent_selection,
            mutation=mutation,
            sparsity=model.theta["pies"].mean() if "sparseflip" in mutation else None,
            p_bf=self.config.get("p_bf"),
        )

        return update_states_for_batch(
            new_states.to(device=K.device), new_lpj.to(device=lpj.device), idx, K, lpj, sort_by_lpj
        )


def evolve_states(
    lpj: Tensor,
    states: Tensor,
    lpj_fn: Callable[[Tensor], Tensor],
    n_parents: int,
    n_children: int,
    n_generations: int,
    parent_selection: str = "batch_fitparents",
    mutation: str = "cross_randflip",
    sparsity: Optional[float] = None,
    p_bf: Optional[float] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Take old variational states states (N,K,H) with lpj values (N,K) and
    return new states and their log-pseudo-joints for each datapoint. The
    helper function `eem.get_n_new_states` can be used to retrieve the
    exact number S of states generated depending on the chosen genetic
    algorithm. lpj_fn must be a callable that takes a set of states with
    shape (N,M,H) as arguments and returns a tuple of log-pseudo-joint
    evaluations for those states (shape (N,M)). This function does not
    guarantee that all new_states returned are unique and not already
    contained in states, but it does guarantee that all redundant states
    will have lpj'states lower than the minimum lpj of all states already
    in states, for each datapoint.

    Pre-conditions: H >= n_children, K >= n_parents

    Return: new_states (N,S,H), new_lpj (N,S)

    Each generation of new states is obtained by selecting `n_parents`
    parents from the previous generation following the strategy indicated
    by `parent_selection` and then mutating each parent `n_children` times
    following the strategy indicated by `genetic_algorithm`.

    parent_selection can be one of the following:
    - 'batch_fitparents'
        parents are selected using fitness-proportional sampling
        _with replacement_.
    - 'randparents'
        random uniform selection of parents.

    genetic_algorithm can be one of the following:
    - 'randflip'
        each children is obtained by flipping one bit of the parent.
        every bit has the same probability of being flipped.
    - 'sparseflip'
        each children is obtained by flipping bits in the parent.
        the probability of each bit being flipped depends on the sparsity
        and p_bh parameters (see method'states description)
    - 'cross'
        children are generated by one-point-crossover of the parents. each
        parent is crossed-over with each other parent
        at a point chosen via random uniform sampling.
    - 'cross_randflip'
        as above, but the children additionally go through 'randflip'
    - 'cross_sparseflip'
        as 'cross', but the children additionally go through 'sparseflip'
    """
    dtype_f, device = lpj.dtype, lpj.device
    N, K, H = states.shape
    max_new_states = get_n_new_states(mutation, n_parents, n_children, n_generations)
    new_states_per_gen = max_new_states // n_generations

    # Pre-allocations
    # It'states probable that not all new_states will be filled with a
    # new unique state. Unfilled new_states will remain uninitialized and
    # their corresponding new_lpj will be lower than any state in states[n].
    new_states = to.empty((N, max_new_states, H), dtype=to.uint8, device=device)
    new_lpj = to.empty((N, max_new_states), dtype=dtype_f, device=device)
    parents = to.empty((N, n_parents, H), dtype=to.uint8, device=device)

    select, mutate = get_EA(parent_selection, mutation)

    for g in range(n_generations):
        # parent selection
        gen_idx = to.arange(g * new_states_per_gen, (g + 1) * new_states_per_gen, device=device)
        if g == 0:
            parents[:] = select(states, n_parents, lpj)
        else:
            old_gen_idx = gen_idx - new_states_per_gen
            parents[:] = select(new_states[:, old_gen_idx], n_parents, new_lpj[:, old_gen_idx])

        # children generation
        for n in range(N):
            new_states[n, gen_idx] = mutate(parents[n], n_children, sparsity, p_bf)

        # children fitness evaluation
        # new_lpj[:, gen_idx] = lpj_fn(new_states[:, gen_idx])
        new_lpj[:, gen_idx] = lpj_fn(new_states[:, gen_idx].to(device=tvem.get_device())).to(
            device="cpu"
        )

    set_redundant_lpj_to_low(new_states, new_lpj, states)

    return new_states, new_lpj


def get_n_new_states(mutation: str, n_parents: int, n_children: int, n_gen: int) -> int:
    if mutation[:5] == "cross":
        return n_parents * (n_parents - 1) * n_gen
    else:
        return n_parents * n_children * n_gen


def get_EA(parent_selection: str, mutation: str) -> Tuple:
    """Refer to the doc of `evolve_states` for the list of valid arguments"""
    parent_sel_dict = {"batch_fitparents": batch_fitparents, "randparents": randparents}
    mutation_dict = {
        "randflip": randflip,
        "sparseflip": sparseflip,
        "cross": cross,
        "cross_randflip": cross_randflip,
        "cross_sparseflip": cross_sparseflip,
    }
    # input validation
    valid_parent_sel = parent_sel_dict.keys()
    if parent_selection not in valid_parent_sel:  # pragma: no cover
        raise ValueError(
            f'Parent selection "{parent_selection}" \
not supported. Valid options: {list(valid_parent_sel)}'
        )
    valid_mutations = mutation_dict.keys()
    if mutation not in valid_mutations:  # pragma: no cover
        raise ValueError(
            f'Mutation operator "{mutation}" not \
supported. Valid options: {list(valid_mutations)}'
        )

    return (parent_sel_dict[parent_selection], mutation_dict[mutation])


def randflip(
    parents: Tensor, n_children: int, sparsity: Optional[float] = None, p_bf: Optional[float] = None
) -> Tensor:
    """Generate n_children new states from parents by flipping one different bit per children."""

    precision, device = to.float64, parents.device

    # Select k indices to be flipped by generating H random numbers per parent
    # and taking the indexes of the largest k.
    # This ensures that, per parent, each child is different.
    n_parents, H = parents.shape
    ind_flip = to.topk(
        to.rand((n_parents, H), dtype=precision, device=device),
        k=n_children,
        dim=1,
        largest=True,
        sorted=False,
    )[1]
    ind_flip_flat = ind_flip.flatten()  # [ parent1bitflip1, parent1bitflip2,
    # parent2bitflip1, parent2bitflip2 ]

    # Each parent is "repeated" n_children times and inserted in children.
    # We then flips bits in the children states
    children = parents.repeat(1, n_children).view(-1, H)
    # is (n_parents*n_children, H)

    # for each new state (0 to n_children*n_parents-1), flip bit at the
    # position indicated by ind_flip_flat
    ind_slice_flat = to.arange(n_children * n_parents, device=parents.device)

    children[ind_slice_flat, ind_flip_flat] = 1 - children[ind_slice_flat, ind_flip_flat]

    return children


def batch_randflip(
    parents: Tensor, n_children: int, sparsity: Optional[float] = None, p_bf: Optional[float] = None
) -> Tensor:
    """Generate n_children new states from parents by flipping one different bit per children.

    :param parents: Tensor with shape (N, n_parents, H)
    :param n_children: How many children to generate per parent per datapoint
    :returns: children, a Tensor with shape (N, n_parents * n_children, H)
    """
    device = parents.device

    # Select k indices to be flipped by generating H random numbers per parent
    # and taking the indexes of the largest k.
    # This ensures that, per parent, each child is different.
    N, n_parents, H = parents.shape
    ind_flip = to.topk(
        to.rand((N, n_parents, H), device=device), k=n_children, dim=2, sorted=False
    )[1]
    ind_flip = ind_flip.view(N, n_parents * n_children)

    # Each parent is "repeated" n_children times and inserted in children.
    # We then flips bits in the children states
    children = parents.repeat(1, 1, n_children).view(N, -1, H)  # is (N, n_parents*n_children, H)

    n_idx = to.arange(N)[:, None]  # broadcastable to ind_flip shape
    s_idx = to.arange(n_parents * n_children)[None, :]  # broadcastable to ind_flip shape
    children[n_idx, s_idx, ind_flip] = 1 - children[n_idx, s_idx, ind_flip]

    return children


def sparseflip(
    parents: Tensor, n_children: int, sparsity: Optional[float], p_bf: Optional[float]
) -> Tensor:
    """ Take a set of parent bitstrings, generate n_children new bitstrings
        by performing bitflips on each of the parents.

    The returned object has shape(parents.shape[0]*n_children,
    parents.shape[1])

    sparsity and p_bf regulate the probabilities of flipping each bit:
    - sparsity: the algorithm will strive to produce children with the
      given sparsity
    - p_bf: overall probability that a bit is flipped. the average number
      of bitflips per children is p_bf*parents.shape[1]
    """
    # Initialization
    precision, device = to.float64, parents.device
    n_parents, H = parents.shape
    s_abs = parents.sum(dim=1)  # is (n_parents)
    children = parents.repeat(1, n_children).view(-1, H)
    eps = 1e-100
    crowdedness = sparsity * H

    H = float(H)
    s_abs = s_abs.to(dtype=precision)

    # # Probability to flip a 1 to a 0 and vice versa (Joerg's idea)
    # p_0 = H / ( 2 * ( H - s_abs) + eps) * p_bf,  # is (n_parents,)
    # p_1 = H / ( 2 * s_abs + eps) * p_bf # is (n_parents,)

    # Probability to flip a 1 to a 0 and vice versa
    # (modification of Joerg's idea)
    alpha = (
        (H - s_abs)
        * ((H * p_bf) - (crowdedness - s_abs))
        / ((crowdedness - s_abs + H * p_bf) * s_abs + eps)
    )  # is (n_parents)
    p_0 = (H * p_bf) / (H + (alpha - 1.0) * s_abs) + eps  # is (n_parents,)
    p_1 = (
        (alpha * p_0)[:, None].expand(-1, int(H)).repeat(1, n_children).view(-1, int(H))
    )  # is (n_parents*n_children, H)
    p_0 = p_0[:, None].expand(-1, int(H)).repeat(1, n_children).view(-1, int(H))
    # is (n_parents*n_children, H)
    p = to.empty(p_0.shape, dtype=precision, device=device)
    # BoolTensor in pytorch>=1.2, ByteTensor otherwise
    bool_or_byte = (to.empty(0) < 0).dtype
    children_idx = children.to(bool_or_byte)
    p[children_idx] = p_1[children_idx]
    p[~children_idx] = p_0[~children_idx]

    # Determine bits to be flipped and do the bitflip
    flips = to.rand((n_parents * n_children, int(H)), dtype=precision, device=device) < p
    children[flips] = 1 - children[flips]

    return children


def batch_sparseflip(
    parents: Tensor, n_children: int, sparsity: Optional[float], p_bf: Optional[float]
) -> Tensor:
    """ Take a set of parent bitstrings, generate n_children new bitstrings
        by performing bitflips on each of the parents.

    :param parents: Tensor with shape (N, n_parents, H)
    :param n_children: number of children to produce per parent per datapoint
    :param sparsity: the algorithm will strive to produce children with the given sparsity
    :param p_bf: overall probability that a bit is flipped. the average number
                 of bitflips per children is p_bf*parents.shape[1]
    :returns: Tensor with shape (N, n_parents*n_children, H)
    """
    # Initialization
    precision, device = to.float64, parents.device
    N, n_parents, H = parents.shape
    eps = 1e-100
    crowdedness = sparsity * H

    H = float(H)
    s_abs = parents.sum(dim=2).to(dtype=precision)  # is (N, n_parents)

    # # Probability to flip a 1 to a 0 and vice versa (Joerg's idea)
    # p_0 = H / ( 2 * ( H - s_abs) + eps) * p_bf,  # is (n_parents,)
    # p_1 = H / ( 2 * s_abs + eps) * p_bf # is (n_parents,)

    # Probability to flip a 1 to a 0 and vice versa (modification of Joerg's idea)
    # is (n_parents)
    alpha = (
        (H - s_abs)
        * ((H * p_bf) - (crowdedness - s_abs))
        / ((crowdedness - s_abs + H * p_bf) * s_abs + eps)
    )
    p_0 = (H * p_bf) / (H + (alpha - 1.0) * s_abs) + eps  # is (N, n_parents)
    p_1 = alpha * p_0
    p_0 = p_0[:, :, None].expand(-1, -1, int(H)).repeat(1, 1, n_children).view(N, -1, int(H))
    p_1 = p_1[:, :, None].expand(-1, -1, int(H)).repeat(1, 1, n_children).view(N, -1, int(H))

    # start from children equal to the parents (with each parent repeated n_children times)
    children = parents.repeat(1, 1, n_children).view(N, n_parents * n_children, int(H))
    assert children.shape == (N, n_parents * n_children, H)
    bool_or_byte = (to.empty(0) < 0).dtype  # BoolTensor in pytorch>=1.2, ByteTensor otherwise
    children_idx = children.to(bool_or_byte)
    p = to.where(children_idx, p_1, p_0)

    # Determine bits to be flipped and do the bitflip
    flips = to.rand((N, n_parents * n_children, int(H)), dtype=precision, device=device) < p
    children[flips] = 1 - children[flips]

    return children


def cross(parents: Tensor) -> Tensor:
    """Each pair of parents is crossed generating two children.

    :param parents: Tensor with shape (n_parents, H)
    :returns: Tensor with shape (n_parents*(n_parents - 1), H)

    The crossover is performed by selecting a "cut point" and switching the
    contents of the parents after the cut point.
    """
    n_parents, H = parents.shape
    n_children = n_parents * (n_parents - 1)
    cutting_points = np.random.randint(low=1, high=H, size=(n_children // 2,))
    parent_pairs = np.array(list(combinations(range(n_parents), 2)), dtype=np.int64)

    # The next lines build (n_children, H) indexes that swap parent entries to produce
    # the desired crossover.
    crossed_idxs = np.empty(n_children * H, dtype=np.int64)
    parent_pair_idxs = np.arange(n_children // 2)
    parent1_starts = parent_pair_idxs * (2 * H)
    cutting_points_1 = parent1_starts + cutting_points
    cutting_points_2 = cutting_points_1 + H
    parent2_ends = parent1_starts + 2 * H
    for pp_idx, o1, o2, o3, o4 in zip(
        parent_pair_idxs, parent1_starts, cutting_points_1, cutting_points_2, parent2_ends
    ):
        parent1, parent2 = parent_pairs[pp_idx]
        crossed_idxs[o1:o2] = parent1
        crossed_idxs[o2:o3] = parent2
        crossed_idxs[o3:o4] = parent1
    crossed_idxs = crossed_idxs.reshape(n_children, H)

    children = parents[crossed_idxs, range(H)]
    return children


# TODO probably to be made a cython helper function for performance
def _fill_crossed_idxs_for_batch(
    parent_pairs, crossed_idxs, parent1_starts, cutting_points_1, cutting_points_2, parent2_ends
):
    n_pairs = parent_pairs.shape[0]
    N = parent1_starts.shape[0]
    for n in range(N):
        for pp_idx in range(n_pairs):
            parent1, parent2 = parent_pairs[pp_idx]
            o1 = parent1_starts[n, pp_idx]
            o2 = cutting_points_1[n, pp_idx]
            o3 = cutting_points_2[n, pp_idx]
            o4 = parent2_ends[n, pp_idx]
            crossed_idxs[n, o1:o2] = parent1
            crossed_idxs[n, o2:o3] = parent2
            crossed_idxs[n, o3:o4] = parent1


def batch_cross(parents: Tensor) -> Tensor:
    """For each datapoint, each pair of parents is crossed generating two children.

    :param parents: Tensor with shape (N, n_parents, H)
    :returns: Tensor with shape (N, n_parents*(n_parents - 1), H)

    The crossover is performed by selecting a "cut point" and switching the
    """
    N, n_parents, H = parents.shape
    parent_pairs = np.array(list(combinations(range(n_parents), 2)), dtype=np.int64)
    n_pairs = parent_pairs.shape[0]
    cutting_points = np.random.randint(low=1, high=H, size=(N, n_pairs))
    n_children = n_pairs * 2  # will produce 2 children per pair

    # The next lines build (N, n_children, H) indexes that swap
    # parent elements to produce the desired crossover.
    crossed_idxs = np.empty((N, n_children * H), dtype=np.int64)
    parent_pair_idxs = np.arange(n_pairs)
    parent1_starts = np.tile(parent_pair_idxs * (2 * H), (N, 1))  # (N, n_children * H)
    cutting_points_1 = parent1_starts + cutting_points
    cutting_points_2 = cutting_points_1 + H
    parent2_ends = parent1_starts + 2 * H
    _fill_crossed_idxs_for_batch(
        parent_pairs, crossed_idxs, parent1_starts, cutting_points_1, cutting_points_2, parent2_ends
    )
    crossed_idxs = crossed_idxs.reshape(N, n_children, H)

    children = parents[np.arange(N)[:, None, None], crossed_idxs, np.arange(H)[None, None, :]]
    return children


def cross_randflip(
    parents: Tensor, n_children: int, sparsity: float = None, p_bf: float = None
) -> Tensor:
    children = randflip(cross(parents), 1)
    return children


def cross_sparseflip(parents: Tensor, n_children: int, sparsity: float, p_bf: float) -> Tensor:
    children = sparseflip(cross(parents), 1, sparsity, p_bf)
    return children


def fitparents(candidates: Tensor, n_parents: int, lpj: Tensor) -> Tensor:

    device = candidates.device

    # compute fitness (per data point)
    lpj_fitness = lpj - 2 * to.min([to.min(lpj), 0.0])  # is (no_candidates,)
    lpj_fitness = lpj_fitness / lpj_fitness.sum()

    # sample (indices of) parents according to fitness
    # TODO Find solution without numpy conversion
    ind_children = np.random.choice(
        candidates.shape[0], size=n_parents, replace=False, p=lpj_fitness.to(device="cpu").numpy()
    )
    # is (n_parents, H)
    return candidates[to.from_numpy(ind_children).to(device=device)]


def batch_fitparents(candidates: Tensor, n_parents: int, lpj: Tensor) -> Tensor:
    # NOTE: this a fitness-proportional parent selection __with replacement__

    precision, device = lpj.dtype, candidates.device
    assert candidates.shape[:2] == lpj.shape, "candidates and lpj must have same shape"

    # compute fitness (per batch)
    lpj_fitness = lpj - 2 * to.min(to.tensor([to.min(lpj).item(), 0.0])).item()
    # is (batch_size, no_candidates).
    lpj_fitness = lpj_fitness / lpj_fitness.sum()
    assert lpj_fitness.shape == lpj.shape

    # we will look for the indeces n for which cum_p[n-1] < x < cump[n]
    # last dimension of x < cum_p will be of the form [False,...,False,
    # True,...,True]
    # summing along the last dimension gives the number of elements greater
    # than x subtracting that from the size of the dimension gives the
    # desired index n
    cum_p = to.cumsum(lpj_fitness, dim=-1)  # (x, y, ..., z), same shape as lpj
    x = to.rand((*cum_p.shape[:-1], n_parents), dtype=precision, device=device)
    # (x, y, ..., n_parents)

    # TODO Find simpler solution
    x_view = tuple(x.shape) + (-1,)
    cum_p_view = list(cum_p.shape)
    cum_p_view.insert(-1, -1)
    cum_p_view = tuple(cum_p_view)  # type: ignore

    chosen_idx = cum_p.shape[-1] - 1 - (x.view(x_view) < cum_p.view(cum_p_view)).sum(dim=-1)

    # TODO Find solution without numpy conversion
    all_idx = to.from_numpy(np.indices(tuple(chosen_idx.shape))).to(device=device)
    # TODO Find solution without numpy conversion
    all_idx[-1] = chosen_idx
    choices = candidates[tuple(i for i in all_idx)]
    assert choices.shape == (candidates.shape[0], n_parents, candidates.shape[2])
    return choices


def randparents(candidates: Tensor, n_parents: int, lpj: Tensor = None) -> Tensor:
    device = candidates.device
    batch_size, n_candidates, H = candidates.shape
    # for each batch, choose n_parents random idxs, concatenate all idxs together
    ind_children = to.cat(
        tuple(to.randperm(n_candidates, device=device)[:n_parents] for _ in range(batch_size))
    )
    # generate indxs for the first dimention of candidates that match ind_children, e.g.
    # [0,0,1,1,2,2] for batch_size=3 and n_parents=2
    # (for each element in the batch, we have 2 ind_children)
    # TODO: change to `repeat_interleave(to.arange(batch_size), n_parents)` when
    # a new-enough pytorch version becomes available at Oldenburg.
    ind_batch = to.arange(batch_size).unsqueeze(1).repeat(1, n_parents).view(-1)
    # need a reshape because the fancy indexing flattens the first two dimensions
    parents = candidates[ind_batch, ind_children].reshape(batch_size, n_parents, H)
    return parents
