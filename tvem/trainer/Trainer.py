# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from tvem.models import TVEMModel  # type: ignore
from tvem.variational import TVEMVariationalStates  # type: ignore


def _make_dataloader(data: Tensor, batch_size: int = 25):
    """Create a pytorch DataLoader that returns datapoint indeces together with batches.

    data -- should have shape (N,D)
    batch_size -- should be an exact divisor of N

    Example usage of the DataLoader created here:
        for idx, batch in data_loader:
            process(idx, batch)
    """
    N = data.shape[0]
    assert N % batch_size == 0, 'batch_size should be an exact divisor of data.shape[0]'
    return DataLoader(TensorDataset(to.arange(N), data), batch_size=batch_size, shuffle=True)


class Trainer:
    """Train and test a given TVEMModel."""

    def __init__(self, model: TVEMModel):
        """Construct a Trainer."""
        self.model = model

    def train(self, epochs: int,
              train_data: Tensor, train_states: TVEMVariationalStates,
              val_data: Tensor = None, val_states: TVEMVariationalStates = None):
        """Train model on given dataset for the given number of epochs.

        epochs -- number of epochs to train for
        train_data -- should have shape (N,D)
        train_states -- TVEMVariationalStates with shape (N,S,H)
        val_data -- should have shape (M,D) (optional)
        val_states -- TVEMVariationalStates with shape (M,Z,H) (optional)

        Training steps on the validation dataset only perform E-steps,
        i.e. model parameters are not updated but val_states are.
        """
        assert (val_data is not None) == (val_states is not None),\
            'Please provide both validation dataset and variational states, or neither'

        # Setup #
        model = self.model
        train_N = train_data.shape[0]
        train_dataset = _make_dataloader(train_data)
        lpj_fn = model.get_lpj_func()

        if val_data is not None:
            val_N = val_data.shape[0]
            val_dataset = _make_dataloader(val_data)

        for e in range(epochs):
            print(f'\nepoch {e}')

            # Training #
            model.init_epoch()
            train_F = 0
            for idx, batch in train_dataset:
                n_subs, mstep_factors = train_states.update(idx, batch, lpj_fn)
                batch_F = model.update_param_batch(idx, batch, train_states, mstep_factors)
                if batch_F is None:
                    batch_F = model.free_energy(idx, batch, train_states)
                train_F += batch_F
            model.update_param_epoch()
            print(f'\ttrain F/N: {train_F/train_N:.5f}')

            # Validation #
            if val_states is not None:
                val_F = 0
                for idx, batch in val_dataset:
                    val_states.update(idx, batch, lpj_fn)
                    val_F += model.free_energy(idx, batch, val_states)
                print(f'\tval F/N: {val_F/val_N:.5f}')

    def test(self,  epochs: int, test_data: Tensor, test_states: TVEMVariationalStates):
        """Test model (does not run M-step).

        epochs -- number of epochs to run testing for: test_states are improved at each
            iteration, therefore test results improve as the number of testing epochs increase.
        test_data -- should have shape (N,D)
        test_states -- TVEMVariationalStates with shape (N,S,H)
        """
        model = self.model
        test_N = test_data.shape[0]
        test_dataset = _make_dataloader(test_data)
        lpj_fn = model.get_lpj_func()

        for e in range(epochs):
            print(f'\nepoch {e}')
            test_F = 0
            for idx, batch in test_dataset:
                test_states.update(idx, batch, lpj_fn)
                test_F += model.free_energy(idx, batch, test_states)
            print(f'\ttest F/N: {test_F/test_N:.5f}')
