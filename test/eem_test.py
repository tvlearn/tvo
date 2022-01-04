# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import unittest
import torch as to

from torch import Tensor

from tvem.variational.TVEMVariationalStates import generate_unique_states
from tvem.variational import eem
import tvem
import pytest
from itertools import combinations
import numpy as np
from tvem.utils.model_protocols import Trainable


def reset_rng_state(seed):
    # to be sure to get the same behavior
    to.manual_seed(seed)
    to.cuda.manual_seed_all(seed)
    np.random.seed(seed)


class DummyModel(Trainable):
    def update_param_batch(self):
        pass

    def log_joint(self, data: Tensor, states: Tensor, lpj: Tensor = None) -> Tensor:
        """Dummy log-pseudo-joint."""
        N, S, H = states.shape

        s_ids = to.empty((H,), dtype=to.int64, device=states.device)
        for h in range(H):
            s_ids[h] = 2 ** h

        return (
            to.mul(states.to(dtype=to.int64), s_ids[None, None, :].expand(N, S, -1))
            .sum(dim=2)
            .to(dtype=to.float64)
        )


@pytest.mark.gpu
class TestEEM(unittest.TestCase):
    """Define unittests for tvem.variational.TVEMVariationalStates module.

    Can be executed individually with:
        ```
            python -m unittest test/eem_test.py
        ```
    """

    def setUp(self):
        self.precision = to.float64
        self.n_runs = 30

    def test_randflip(self):

        H, n_parents, n_children = 5, 4, 2

        for x in range(self.n_runs):
            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)
            children = eem.randflip(parents, n_children)  # is (n_parents*n_children, H)

            self.assertEqual(children.shape[0], n_parents * n_children)

            flips_per_child = (parents.repeat(1, n_children).view(-1, H) != children).sum(dim=1)
            self.assertTrue(to.all(flips_per_child == 1))

    def test_batch_randflip(self):

        N, H, n_parents, n_children = 3, 5, 4, 2

        for x in range(self.n_runs):
            # parents have shape (N, n_parents, H)
            parents = generate_unique_states(n_states=n_parents, H=H)
            parents = parents.unsqueeze(0).expand((N, -1, -1))
            # children have shape (N, n_parents*n_children, H)
            children = eem.batch_randflip(parents, n_children)

            self.assertEqual(children.shape, (N, n_parents * n_children, H))

            for n in range(N):
                flips_per_child = (parents[n].repeat(1, n_children).view(-1, H) != children[n]).sum(
                    dim=1
                )
                self.assertTrue(to.all(flips_per_child == 1))

    def test_randflip_vs_batch_randflip(self):
        # make sure they return the same thing for N == 1
        N, H, n_parents, n_children = 1, 5, 4, 2

        parents = generate_unique_states(n_states=n_parents, H=H)
        parents = parents.unsqueeze(0).expand((N, -1, -1))
        seed = np.random.randint(10000)
        reset_rng_state(seed)
        children_batch = eem.batch_randflip(parents, n_children)
        reset_rng_state(seed)
        children = eem.randflip(parents[0], n_children)

        self.assertTrue(to.equal(children, children_batch[0]))

    def test_sparseflip(self):

        H, n_parents, n_children, sparsity, p_bf = 5, 4, 2, 1.0 / 5, 0.5

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)
            children = eem.sparseflip(parents, n_children, sparsity, p_bf)

            self.assertEqual(children.shape[0], n_parents * n_children)
            self.assertTrue(
                (
                    (parents.repeat(1, n_children).view(-1, H) == children).sum()
                    / (n_parents * n_children)
                ).item()
                > 0
            )  # TODO Find better test

    def test_batch_sparseflip(self):

        N, H, n_parents, n_children = 3, 5, 4, 2
        sparsity, p_bf = 1 / H, 0.5

        for x in range(self.n_runs):

            # parents have shape (N, n_parents, H)
            parents = generate_unique_states(n_states=n_parents, H=H)
            parents = parents.unsqueeze(0).expand((N, -1, -1))
            # children have shape (N, n_parents*n_children, H)
            children = eem.batch_sparseflip(parents, n_children, sparsity, p_bf)

            self.assertEqual(children.shape, (N, n_parents * n_children, H))
            for n in range(N):
                # TODO Find better test
                self.assertTrue(
                    (
                        (parents[n].repeat(1, n_children).view(-1, H) == children[n])
                        .sum(dim=1)
                        .sum()
                        / (n_parents * n_children)
                    ).item()
                    > 0
                )

    def test_sparseflip_vs_batch_sparseflip(self):
        N, H, n_parents, n_children = 1, 5, 4, 2
        sparsity, p_bf = 1 / H, 0.5
        parents = generate_unique_states(n_states=n_parents, H=H)
        parents = parents.unsqueeze(0).expand((N, -1, -1))
        seed = np.random.randint(10000)
        reset_rng_state(seed)
        children_batch = eem.batch_sparseflip(parents, n_children, sparsity, p_bf)
        reset_rng_state(seed)
        children = eem.sparseflip(parents[0], n_children, sparsity, p_bf)
        self.assertTrue(to.equal(children_batch[0], children))

    def test_cross(self):

        H, n_parents = 5, 4

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)
            children = eem.cross(parents)  # is (n_parents*n_children, H)

            self.assertEqual(children.shape[0], n_parents * (n_parents - 1))
            # The sum of the two crossover children must be equal, element by element,
            # to the sum of the parents.
            # The check assumes that children are ordered as if parents were crossed
            # two by two following the order of `combinations`.
            child_idx = 0
            for p1, p2 in combinations(range(n_parents), 2):
                sum_parents = parents[p1] + parents[p2]
                sum_children = children[child_idx] + children[child_idx + 1]
                self.assertTrue(to.equal(sum_parents, sum_children))
                child_idx += 2

    def test_batch_cross(self):

        N, H, n_parents = 3, 5, 4

        for x in range(self.n_runs):

            # parents have shape (N, n_parents, H)
            parents = generate_unique_states(n_states=n_parents, H=H)
            parents = parents.unsqueeze(0).expand((N, -1, -1))
            # children have shape (N, n_parents*n_children, H)
            children = eem.batch_cross(parents)

            self.assertEqual(children.shape, (N, n_parents * (n_parents - 1), H))
            # The sum of the two crossover children must be equal, element by element,
            # to the sum of the parents.
            # The check assumes that children are ordered as if parents were crossed
            # two by two following the order of `combinations`.
            for n in range(N):
                child_idx = 0
                for p1, p2 in combinations(range(n_parents), 2):
                    sum_parents = parents[n, p1] + parents[n, p2]
                    sum_children = children[n, child_idx] + children[n, child_idx + 1]
                    self.assertTrue(to.equal(sum_parents, sum_children))
                    child_idx += 2

    def test_cross_vs_batch_cross(self):
        N, H, n_parents = 1, 5, 4
        parents = generate_unique_states(n_states=n_parents, H=H)
        parents = parents.unsqueeze(0).expand((N, -1, -1))
        seed = np.random.randint(10000)
        reset_rng_state(seed)
        children_batch = eem.batch_cross(parents)
        reset_rng_state(seed)
        children = eem.cross(parents[0])
        self.assertTrue(to.equal(children_batch[0], children))

    def test_cross_randflip(self):

        H, n_parents = 5, 4

        for x in range(self.n_runs):

            seed = np.random.randint(10000)

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)

            reset_rng_state(seed)
            children_wth_flip = eem.cross(parents)

            reset_rng_state(seed)
            children_w_flip = eem.cross_randflip(parents, n_children=1)  # (n_parents*n_children, H)

            self.assertEqual(children_w_flip.shape[0], n_parents * (n_parents - 1))
            flips_per_child = (children_wth_flip != children_w_flip).sum(dim=1)
            self.assertTrue(to.all(flips_per_child == 1))

    def test_cross_sparseflip(self):

        H, n_parents, n_children_, sparsity, p_bf = 5, 4, 1, 1.0 / 5, 0.5
        seed = 7

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)

            to.manual_seed(seed)
            to.cuda.manual_seed_all(seed)
            children_wth_flip = eem.cross(parents)

            to.manual_seed(seed)
            to.cuda.manual_seed_all(seed)
            children_w_flip = eem.cross_sparseflip(
                parents, n_children_, sparsity, p_bf
            )  # is (n_parents*n_children, H)

            self.assertEqual(children_w_flip.shape[0], n_parents * (n_parents - 1) * n_children_)
            self.assertTrue(
                (
                    (children_wth_flip == children_w_flip).sum()
                    / (n_parents * (n_parents - 1) * n_children_)
                ).item()
                > 0
            )  # TODO Find better test
            self.assertTrue(
                (
                    (children_wth_flip == children_w_flip).sum()
                    / (n_parents * (n_parents - 1) * n_children_)
                ).item()
                <= H
            )  # TODO Find better test

    def test_batch_fitparents(self):

        batch_size, n_candidates, H, n_parents = 2, 3, 3, 2

        for x in range(self.n_runs):

            candidates = generate_unique_states(n_states=n_candidates, H=H).repeat(
                batch_size, 1, 1
            )  # is (batch_size, n_candidates, H)
            # is (batch_size, n_candidates)
            lpj = DummyModel().log_joint(None, candidates)

            # is (batch_size, n_parents, H)
            parents = eem.batch_fitparents(candidates, n_parents, lpj)

            self.assertTrue((list(parents.shape) == [batch_size, n_parents, H]))
            # TODO Find more tests

    def test_randparents(self):
        batch_size, n_candidates, H, n_parents = 2, 4, 3, 2

        candidates = generate_unique_states(n_states=n_candidates, H=H).repeat(batch_size, 1, 1)

        # check that parents have the expected shape
        parents = eem.batch_randparents(candidates, n_parents, lpj=None)
        self.assertTrue(parents.shape == (batch_size, n_parents, H))

        # check that the parents we selected for each batch were in candidates in the first place
        parents = eem.batch_randparents(candidates, n_candidates, lpj=None)
        for batch in range(batch_size):
            for p in parents[batch]:
                self.assertTrue(any(to.equal(p, c) for c in candidates[batch]))

    def test_update(self):

        eem_conf = {
            "precision": self.precision,
            "N": 2,
            "S": 3,
            "H": 4,
            "parent_selection": "batch_fitparents",
            "mutation": "randflip",
            "n_parents": 2,
            "n_children": 1,
            "n_generations": 1,
        }

        device = tvem.get_device()

        for x in range(self.n_runs):

            idx = to.arange(eem_conf["N"], device=device)
            data_dummy = to.empty((1,), device=device)

            var_states = eem.EEMVariationalStates(**eem_conf)

            # is (batch_size, n_candidates)
            var_states.lpj[:] = DummyModel().log_joint(data=data_dummy, states=var_states.K[idx])

            old_sum_lpj_over_n = var_states.lpj[:].sum(dim=1)  # is (N,)

            nsubs = var_states.update(idx=idx, batch=data_dummy, model=DummyModel())

            new_sum_lpj_over_n = var_states.lpj[:].sum(dim=1)  # is (N,)
            no_datapoints_without_lpj_increase = (new_sum_lpj_over_n == old_sum_lpj_over_n).sum()
            no_datapoints_with_lpj_decrease = (
                (new_sum_lpj_over_n - old_sum_lpj_over_n) < (-1e-10)
            ).sum()

            self.assertTrue(nsubs >= 0)
            self.assertEqual(no_datapoints_with_lpj_decrease, 0)
            if no_datapoints_without_lpj_increase > 0:
                self.assertTrue(nsubs < (eem_conf["N"] * eem_conf["S"]))
            # TODO Find more tests
