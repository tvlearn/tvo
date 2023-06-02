# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0


import unittest

import torch as to

from tvo.variational._utils import (
    generate_unique_states,
    update_states_for_batch,
    set_redundant_lpj_to_low,
    _unique_ind,
)
import tvo
import pytest


@pytest.mark.gpu
class TestTVO(unittest.TestCase):
    """Define unittests for tvo.variational.TVOVariationalStates module.

    Can be executed individually with:
        ```
            python -m unittest test/TVOVariationalStates_test.py
        ```
    """

    def setUp(self):
        self.precision = to.float64

    def test_unique_ind(self):
        states = to.tensor([[0, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=to.uint8)

        states_unique_ind = _unique_ind(states)

        self.assertEqual(states_unique_ind.numel(), 3)
        self.assertEqual((states_unique_ind.sort()[0] == to.tensor([0, 2, 4])).sum().item(), 3)

    def test_generate_unique_states(self):
        n_states, H = 5, 8

        device = tvo.get_device()
        states = generate_unique_states(n_states=n_states, H=H, device=device)
        _, reverse_ind = to.unique(states, return_inverse=True, dim=0)

        self.assertEqual(states.shape[0], n_states)
        self.assertTrue(to.equal(reverse_ind.sort()[0], to.arange(n_states, device=device)))

    def test_update_states_for_batch(self):
        precision = self.precision

        idx = to.arange(2)
        new_states = to.ones((idx.numel(), 2, 1), dtype=to.uint8)
        all_states = to.zeros((4, 3, 1), dtype=to.uint8)  # is (N, S, H)
        new_lpj = to.ones((2, 2), dtype=precision)
        all_lpj = to.zeros((4, 3), dtype=precision)

        n_subs = update_states_for_batch(new_states, new_lpj, idx, all_states, all_lpj)

        self.assertEqual(
            (
                all_states
                == to.tensor(
                    [[[0], [1], [1]], [[0], [1], [1]], [[0], [0], [0]], [[0], [0], [0]]],
                    dtype=to.uint8,
                )
            ).sum(),
            all_states.numel(),
        )

        self.assertEqual(n_subs, 4)

    def test_set_redundant_lpj_to_low(self):
        precision = self.precision

        device = tvo.get_device()
        old_states = to.tensor(
            [[[0, 1, 1], [1, 0, 0]], [[0, 1, 1], [1, 0, 1]], [[0, 1, 1], [1, 1, 1]]],
            dtype=to.uint8,
            device=device,
        )
        new_states = to.tensor(
            [[[0, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 0, 1]], [[0, 0, 0], [0, 0, 0]]],
            dtype=to.uint8,
            device=device,
        )

        N, S, H = old_states.shape

        new_lpj = to.ones((N, S), dtype=precision, device=device)

        set_redundant_lpj_to_low(new_states, new_lpj, old_states)

        self.assertEqual(to.isclose(new_lpj, to.ones_like(new_lpj)).sum().item(), 4)
        # new_lpj[1, 1] repeats an old state
        expected_low = to.tensor(-1e20, device=device, dtype=precision)
        self.assertTrue(to.allclose(new_lpj[1, 1], expected_low))
        # new_lpj[2, :] are the same state: one of the two must be discarded
        # i.e. the sum of the two lpj will be ~expected_low)
        self.assertTrue(to.allclose(new_lpj[2].sum(), expected_low))
