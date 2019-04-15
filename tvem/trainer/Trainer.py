# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.models import TVEMModel
from tvem.variational import TVEMVariationalStates
from tvem.util.data import TVEMDataLoader
from tvem.util.parallel import all_reduce
from typing import Dict, Any
import torch as to


class Trainer:
    def __init__(self, model: TVEMModel,
                 train_data: TVEMDataLoader = None, train_states: TVEMVariationalStates = None,
                 test_data: TVEMDataLoader = None, test_states: TVEMVariationalStates = None):
        """Train and/or test a given TVEMModel.

        :param model: an object of a concrete type inheriting from TVEMModel
        :param train_data: the contained dataset should have shape (N,D)
        :param train_states: TVEMVariationalStates with shape (N,S,H)
        :param test_data: validation or test dataset. The contained dataset should have shape (M,D)
        :param test_states: TVEMVariationalStates with shape (M,Z,H)

        Both train_data and train_states must be provided, or neither.
        The same holds for test_data and test_states.
        At least one of these two pairs of arguments must be present.

        Training steps on test_data only perform E-steps, i.e. model parameters are
        not updated but test_states are. Therefore test_data can also be used for validation.
        """
        for data, states in ((train_data, train_states), (test_data, test_states)):
            assert (data is not None) == (states is not None),\
                'Please provide both dataset and variational states, or neither'
        self.can_train = train_data is not None and train_states is not None
        self.can_test = test_data is not None and test_states is not None
        if not self.can_train and not self.can_test:  # pragma: no cover
            raise RuntimeError(
                'Please provide at least one pair of dataset and variational states')

        self.model = model
        self.train_data = train_data
        self.train_states = train_states
        self.test_data = test_data
        self.test_states = test_states
        if train_data is not None:
            self.N_train = to.tensor(train_data.dataset.tensors[0].shape[0])
            all_reduce(self.N_train)
            self.N_train = self.N_train.item()
        if test_data is not None:
            self.N_test = to.tensor(test_data.dataset.tensors[0].shape[0])
            all_reduce(self.N_test)
            self.N_test = self.N_test.item()

    @staticmethod
    def _do_e_step(data, states, model, N):
        F = to.tensor(0.)
        subs = to.tensor(0)
        for idx, batch in data:
            model.init_batch()
            subs += states.update(idx, batch, model.log_pseudo_joint)
            F += model.free_energy(idx, batch, states)
        all_reduce(F)
        all_reduce(subs)
        return F.item() / N, subs.item() / N

    def e_step(self) -> Dict[str, Any]:
        """Run one epoch of E-steps on training and/or test data, depending on what is available.

        Only E-steps are executed.
        :returns: a dictionary containing 'train_F', 'train_subs', 'test_F', 'test_subs'
                  (keys might be missing depending on what is available)
        """
        ret = {}
        model = self.model
        train_data, train_states = self.train_data, self.train_states
        test_data, test_states = self.test_data, self.test_states

        # Training #
        if self.can_train:
            assert train_data is not None and train_states is not None  # to make mypy happy
            ret['train_F'], ret['train_subs'] = self._do_e_step(train_data, train_states, model,
                                                                self.N_train)

        # Validation/Testing #
        if self.can_test:
            assert test_data is not None and test_states is not None  # to make mypy happy
            ret['test_F'], ret['test_subs'] = self._do_e_step(test_data, test_states, model,
                                                              self.N_test)

        return ret

    def em_step(self) -> Dict[str, Any]:
        """Run one training and/or test epoch, depending on what data is available.

        Both E-step and M-step are executed.
        :returns: a dictionary containing 'train_F', 'train_subs', 'test_F', 'test_subs'
                  (keys might be missing depending on what is available)
        """
        model = self.model
        train_data, train_states = self.train_data, self.train_states
        test_data, test_states = self.test_data, self.test_states
        lpj_fn = self.model.log_pseudo_joint

        ret_dict = {}

        # Training #
        if self.can_train:
            assert train_data is not None and train_states is not None  # to make mypy happy
            F = to.tensor(0.)
            subs = to.tensor(0)
            model.init_epoch()
            for idx, batch in train_data:
                model.init_batch()
                subs += train_states.update(idx, batch,
                                            lpj_fn, sort_by_lpj=model.sorted_by_lpj)
                batch_F = model.update_param_batch(idx, batch, train_states)
                if batch_F is None:
                    batch_F = model.free_energy(idx, batch, train_states)
                F += batch_F
            model.update_param_epoch()
            all_reduce(F)
            all_reduce(subs)
            ret_dict['train_F'] = F.item() / self.N_train
            ret_dict['train_subs'] = subs.item() / self.N_train

        # Validation/Testing #
        if self.can_test:
            assert test_data is not None and test_states is not None  # to make mypy happy
            model.init_epoch()
            res = self._do_e_step(test_data, test_states, model, self.N_test)
            ret_dict['test_F'], ret_dict['test_subs'] = res

        return ret_dict
