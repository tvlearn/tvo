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


def lpj_dummy(data: Tensor, states: Tensor) -> Tensor:
    """Dummy log-pseudo-joint. """
    N, S, H = states.shape

    s_ids = to.empty((H,), dtype=to.int64, device=states.device)
    for h in range(H):
        s_ids[h] = 2**h

    return to.mul(states.to(dtype=to.int64),
                  s_ids[None, None, :].expand(N, S, -1)).sum(dim=2).to(dtype=to.float64)


@pytest.mark.gpu
class TestEEM(unittest.TestCase):
    """Define unittests for tvem.variational.TVEMVariationalStates module.

    Can be executed individually with:
        ```
            python -m unittest test/eem_test.py
        ```
    """

    def setUp(self):
        self.dtype_f = to.float64
        self.n_runs = 30

    def test_randflip(self):

        H, n_parents, n_children = 5, 4, 2

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)
            # is (n_parents*n_children, H)
            children = eem.randflip(parents, n_children)

            self.assertEqual(children.shape[0], n_parents*n_children)
            self.assertEqual(((parents.repeat(1, n_children).view(-1, H) ==
                               children).sum(dim=1).sum()/(n_parents*n_children)).item(), H-1)

    def test_sparseflip(self):

        H, n_parents, n_children, sparsity, p_bf = 5, 4, 2, 1./5, 0.5

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)
            children = eem.sparseflip(parents, n_children, sparsity, p_bf)

            self.assertEqual(children.shape[0], n_parents*n_children)
            self.assertTrue(((parents.repeat(1, n_children).view(-1, H) == children).sum(
                dim=1).sum()/(n_parents*n_children)).item() > 0)  # TODO Find better test

    def test_cross(self):

        H, n_parents = 5, 4

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)
            children = eem.cross(parents)  # is (n_parents*n_children, H)

            self.assertEqual(children.shape[0], n_parents*(n_parents-1))
            # TODO Find more tests

    def test_cross_randflip(self):

        H, n_parents, n_children_ = 5, 4, 1
        seed = 7

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)

            to.manual_seed(seed)
            to.cuda.manual_seed_all(seed)
            children_wth_flip = eem.cross(parents)

            to.manual_seed(seed)
            to.cuda.manual_seed_all(seed)
            children_w_flip = eem.cross_randflip(
                parents, n_children_)  # is (n_parents*n_children, H)

            self.assertEqual(
                children_w_flip.shape[0], n_parents*(n_parents-1)*n_children_)
            self.assertEqual(((children_wth_flip == children_w_flip).sum(
                dim=1).sum()/(n_parents*(n_parents-1)*n_children_)).item(), H-1)

    def test_cross_sparseflip(self):

        H, n_parents, n_children_, sparsity, p_bf = 5, 4, 1, 1./5, 0.5
        seed = 7

        for x in range(self.n_runs):

            parents = generate_unique_states(n_states=n_parents, H=H)  # is (n_parents, H)

            to.manual_seed(seed)
            to.cuda.manual_seed_all(seed)
            children_wth_flip = eem.cross(parents)

            to.manual_seed(seed)
            to.cuda.manual_seed_all(seed)
            children_w_flip = eem.cross_sparseflip(
                parents, n_children_, sparsity, p_bf)  # is (n_parents*n_children, H)

            self.assertEqual(
                children_w_flip.shape[0], n_parents*(n_parents-1)*n_children_)
            self.assertTrue(((children_wth_flip == children_w_flip).sum(dim=1).sum(
            )/(n_parents*(n_parents-1)*n_children_)).item() > 0)  # TODO Find better test
            self.assertTrue(((children_wth_flip == children_w_flip).sum(dim=1).sum(
            )/(n_parents*(n_parents-1)*n_children_)).item() <= H)  # TODO Find better test

    def test_batch_fitparents(self):

        batch_size, n_candidates, H, n_parents = 2, 3, 3, 2

        for x in range(self.n_runs):

            candidates = generate_unique_states(n_states=n_candidates, H=H)\
                         .repeat(batch_size, 1, 1)  # is (batch_size, n_candidates, H)
            # is (batch_size, n_candidates)
            lpj = lpj_dummy(None, candidates)

            # is (batch_size, n_parents, H)
            parents = eem.batch_fitparents(candidates, n_parents, lpj)

            self.assertTrue(
                (list(parents.shape) == [batch_size, n_parents, H]))
            # TODO Find more tests

    def test_update(self):

        eem_conf = {
            'dtype': self.dtype_f,
            'N': 2,
            'S': 3,
            'H': 4,
            'parent_selection': 'batch_fitparents',
            'mutation': 'randflip',
            'n_parents': 2,
            'n_children': 1,
            'n_generations': 1}

        device = tvem.get_device()

        for x in range(self.n_runs):

            idx = to.arange(eem_conf['N'], device=device)
            data_dummy = to.empty((1,), device=device)

            var_states = eem.EEMVariationalStates(conf=eem_conf)

            # is (batch_size, n_candidates)
            var_states.lpj[:] = lpj_dummy(data=data_dummy, states=var_states.K[idx])

            old_sum_lpj_over_n = var_states.lpj[:].sum(dim=1)  # is (N,)

            nsubs = var_states.update(
                idx=idx, batch=data_dummy, lpj_fn=lpj_dummy)

            new_sum_lpj_over_n = var_states.lpj[:].sum(dim=1)  # is (N,)
            no_datapoints_without_lpj_increase = (
                new_sum_lpj_over_n == old_sum_lpj_over_n).sum()
            no_datapoints_with_lpj_decrease = (
                (new_sum_lpj_over_n-old_sum_lpj_over_n) < (-1e-10)).sum()

            self.assertTrue(nsubs >= 0)
            self.assertEqual(no_datapoints_with_lpj_decrease, 0)
            if no_datapoints_without_lpj_increase > 0:
                self.assertTrue(nsubs < (eem_conf['N']*eem_conf['S']))
            # TODO Find more tests
