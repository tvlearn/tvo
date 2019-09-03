# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0


import unittest

import torch as to

from tvem.variational._utils import (
    unique_ind,
    generate_unique_states,
    update_states_for_batch,
    set_redundant_lpj_to_low,
)
import tvem
import pytest


@pytest.mark.gpu
class TestTVEM(unittest.TestCase):
    """Define unittests for tvem.variational.TVEMVariationalStates module.

    Can be executed individually with:
        ```
            python -m unittest test/TVEMVariationalStates_test.py
        ```
    """

    def setUp(self):
        self.precision = to.float64

    def test_unique_ind(self):

        states = to.tensor([[0, 1, 1], [0, 1, 1], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=to.uint8)

        states_unique_ind = unique_ind(states)

        self.assertEqual(states_unique_ind.numel(), 3)
        self.assertEqual((states_unique_ind.sort()[0] == to.tensor([0, 2, 4])).sum().item(), 3)

    def test_generate_unique_states(self):

        n_states, H = 5, 8

        device = tvem.get_device()
        states = generate_unique_states(n_states=n_states, H=H, device=device)
        states_unique_ind = unique_ind(states, dim=0)

        self.assertEqual(states.shape[0], n_states)
        self.assertEqual(
            (states_unique_ind.sort()[0] == to.arange(n_states, device=device)).sum().item(),
            n_states,
        )

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

        device = tvem.get_device()
        old_states = to.tensor(
            [[[0, 1, 1], [1, 0, 0]], [[0, 1, 1], [1, 0, 1]]], dtype=to.uint8, device=device
        )
        new_states = to.tensor(
            [[[0, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 0, 1]]], dtype=to.uint8, device=device
        )

        N, S, H = old_states.shape

        new_lpj = to.ones((N, S), dtype=precision, device=device)

        set_redundant_lpj_to_low(new_states, new_lpj, old_states)

        self.assertEqual((new_lpj == 1.0).sum().item(), 3)
        self.assertEqual(new_lpj[1, 1].item(), -1e100)
