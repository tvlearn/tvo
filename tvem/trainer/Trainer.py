# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.models import TVEMModel
from tvem.variational import TVEMVariationalStates
from tvem.utils.data import TVEMDataLoader
from tvem.utils.parallel import all_reduce
from typing import Dict, Any, Sequence
import torch as to


class Trainer:
    def __init__(
        self,
        model: TVEMModel,
        train_data: TVEMDataLoader = None,
        train_states: TVEMVariationalStates = None,
        test_data: TVEMDataLoader = None,
        test_states: TVEMVariationalStates = None,
        rollback_if_F_decreases: Sequence[str] = [],
    ):
        """Train and/or test a given TVEMModel.

        :param model: an object of a concrete type inheriting from TVEMModel
        :param train_data: the contained dataset should have shape (N,D)
        :param train_states: TVEMVariationalStates with shape (N,S,H)
        :param test_data: validation or test dataset. The contained dataset should have shape (M,D)
        :param test_states: TVEMVariationalStates with shape (M,Z,H)
        :param rollback_if_F_decreases: see ExpConfig docs

        Both train_data and train_states must be provided, or neither.
        The same holds for test_data and test_states.
        At least one of these two pairs of arguments must be present.

        Training steps on test_data only perform E-steps, i.e. model parameters are
        not updated but test_states are. Therefore test_data can also be used for validation.
        """
        for data, states in ((train_data, train_states), (test_data, test_states)):
            assert (data is not None) == (
                states is not None
            ), "Please provide both dataset and variational states, or neither"
        self.can_train = train_data is not None and train_states is not None
        self.can_test = test_data is not None and test_states is not None
        if not self.can_train and not self.can_test:  # pragma: no cover
            raise RuntimeError("Please provide at least one pair of dataset and variational states")

        self.model = model
        self.train_data = train_data
        self.train_states = train_states
        self.test_data = test_data
        self.test_states = test_states
        if train_data is not None:
            self.N_train = to.tensor(len(train_data.dataset))
            all_reduce(self.N_train)
            self.N_train = self.N_train.item()
        if test_data is not None:
            self.N_test = to.tensor(len(test_data.dataset))
            all_reduce(self.N_test)
            self.N_test = self.N_test.item()
        self._to_rollback = rollback_if_F_decreases

    @staticmethod
    def _do_e_step(data: TVEMDataLoader, states: TVEMVariationalStates, model: TVEMModel, N: int):
        F = to.tensor(0.0)
        subs = to.tensor(0)

        model.init_epoch()
        for idx, batch in data:
            model.init_batch()
            subs += states.update(idx, batch, model)
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
            ret["train_F"], ret["train_subs"] = self._do_e_step(
                train_data, train_states, model, self.N_train
            )

        # Validation/Testing #
        if self.can_test:
            assert test_data is not None and test_states is not None  # to make mypy happy
            ret["test_F"], ret["test_subs"] = self._do_e_step(
                test_data, test_states, model, self.N_test
            )

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

        ret_dict = {}

        # Training #
        if self.can_train:
            assert train_data is not None and train_states is not None  # to make mypy happy
            F = to.tensor(0.0)
            subs = to.tensor(0)
            model.init_epoch()
            for idx, batch in train_data:
                model.init_batch()
                subs += train_states.update(idx, batch, model)
                batch_F = model.update_param_batch(idx, batch, train_states)
                if batch_F is None:
                    batch_F = model.free_energy(idx, batch, train_states)
                F += batch_F
            if len(self._to_rollback) > 0:
                self._update_parameters_with_rollback()
            else:
                model.update_param_epoch()
            all_reduce(F)
            all_reduce(subs)
            ret_dict["train_F"] = F.item() / self.N_train
            ret_dict["train_subs"] = subs.item() / self.N_train

        # Validation/Testing #
        if self.can_test:
            assert test_data is not None and test_states is not None  # to make mypy happy
            res = self._do_e_step(test_data, test_states, model, self.N_test)
            ret_dict["test_F"], ret_dict["test_subs"] = res

        return ret_dict

    def _update_parameters_with_rollback(self) -> None:
        m = self.model
        lpj_fn = m.log_joint if m.log_pseudo_joint is NotImplemented else m.log_pseudo_joint

        assert self.train_data is not None and self.train_states is not None  # to make mypy happy
        all_data = self.train_data.dataset.tensors[1]
        states = self.train_states

        old_params = {p: m.theta[p].clone() for p in self._to_rollback}
        old_F = m.free_energy(idx=to.arange(all_data.shape[0]), batch=all_data, states=states)
        all_reduce(old_F)
        old_lpj = states.lpj.clone()
        m.update_param_epoch()
        states.lpj[:] = lpj_fn(all_data, states.K)
        new_F = m.free_energy(idx=to.arange(all_data.shape[0]), batch=all_data, states=states)
        all_reduce(new_F)
        if new_F < old_F:
            for p in self._to_rollback:
                m.theta[p][:] = old_params[p]
            states.lpj[:] = old_lpj
