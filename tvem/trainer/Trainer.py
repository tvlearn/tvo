# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.models import TVEMModel
from tvem.variational import TVEMVariationalStates
from tvem.util.data import TVEMDataLoader


class Trainer:
    def __init__(self, model: TVEMModel):
        """Train and test a given TVEMModel.

        :param model: an object of a concrete type inheriting from TVEMModel
        """
        self.model = model

    def train(self, epochs: int, train_data: TVEMDataLoader, train_states: TVEMVariationalStates,
              val_data: TVEMDataLoader = None, val_states: TVEMVariationalStates = None):
        """Train model on given dataset for the given number of epochs.

        :param epochs: number of epochs to train for
        :param train_data: the contained dataset should have shape (N,D)
        :param train_states: TVEMVariationalStates with shape (N,S,H)
        :param val_data: the contained dataset should have shape (M,D) (optional)
        :param val_states: TVEMVariationalStates with shape (M,Z,H) (optional)

        Training steps on the validation dataset only perform E-steps,
        i.e. model parameters are not updated but val_states are.
        """
        assert (val_data is not None) == (val_states is not None),\
            'Please provide both validation dataset and variational states, or neither'

        # Setup #
        model = self.model
        train_N = train_data.dataset.tensors[0].shape[0]
        lpj_fn = model.log_pseudo_joint

        if val_data is not None:
            val_N = val_data.dataset.tensors[0].shape[0]

        for e in range(epochs):
            print(f'\nepoch {e}')

            # Training #
            model.init_epoch()
            train_F = 0.
            for idx, batch in train_data:
                # TODO count avg number of subs
                train_states.update(idx, batch, lpj_fn, sort_by_lpj=model.sorted_by_lpj)
                batch_F = model.update_param_batch(idx, batch, train_states)
                if batch_F is None:
                    batch_F = model.free_energy(idx, batch, train_states)
                train_F += batch_F
            model.update_param_epoch()
            print(f'\ttrain F/N: {train_F/train_N:.5f}')

            # Validation #
            if val_data is not None and val_states is not None:  # checking both to make mypy happy
                val_F = 0.
                for idx, batch in val_data:
                    val_states.update(idx, batch, lpj_fn)
                    val_F += model.free_energy(idx, batch, val_states)
                print(f'\tval F/N: {val_F/val_N:.5f}')

    def test(self, epochs: int, test_data: TVEMDataLoader, test_states: TVEMVariationalStates):
        """Test model (does not run M-step).

        :param epochs: number of epochs to run testing for: test_states are improved at each\
            iteration, therefore test results improve as the number of testing epochs increase.
        :param test_data: the contained dataset should have shape (N,D)
        :param test_states: TVEMVariationalStates with shape (N,S,H)
        """
        model = self.model
        test_N = test_data.dataset.tensors[0].shape[0]
        lpj_fn = model.log_pseudo_joint

        for e in range(epochs):
            print(f'\nepoch {e}')
            test_F = 0.
            for idx, batch in test_data:
                test_states.update(idx, batch, lpj_fn)
                test_F += model.free_energy(idx, batch, test_states)
            print(f'\ttest F/N: {test_F/test_N:.5f}')
