# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import tvem
from tvem.utils.model_protocols import Trainable, Optimized, Reconstructor
from tvem.variational import TVEMVariationalStates
from tvem.utils.data import TVEMDataLoader
from tvem.utils.parallel import all_reduce
from typing import Dict, Any, Sequence, Union, Callable
import torch as to


class Trainer:
    def __init__(
        self,
        model: Trainable,
        train_data: Union[TVEMDataLoader, to.Tensor] = None,
        train_states: TVEMVariationalStates = None,
        test_data: Union[TVEMDataLoader, to.Tensor] = None,
        test_states: TVEMVariationalStates = None,
        rollback_if_F_decreases: Sequence[str] = [],
        will_reconstruct: bool = False,
        eval_F_at_epoch_end: bool = False,
        data_transform: Callable[[to.Tensor], to.Tensor] = None,
    ):
        """Train and/or test a given model.

        :param model: an object of a concrete type satisfying the Trainable protocol
        :param train_data: the contained dataset should have shape (N,D)
        :param train_states: TVEMVariationalStates with shape (N,S,H)
        :param test_data: validation or test dataset. The contained dataset should have shape (M,D)
        :param test_states: TVEMVariationalStates with shape (M,Z,H)
        :param rollback_if_F_decreases: see ExpConfig docs
        :param will_reconstruct: True if data will be reconstructed by the Trainer
        :param eval_F_at_epoch_end: By default, the trainer evaluates the model free energy batch
                                    by batch, accumulating the values over the course of the epoch.
                                    If this option is set to `True`, the free energy will be
                                    evaluated at the end of an epoch instead.
        :param data_transform: A transformation to be applied to datapoints before they are passed
                               to the model for training/evaluation.

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
        train_data = TVEMDataLoader(train_data) if isinstance(train_data, to.Tensor) else train_data
        test_data = TVEMDataLoader(test_data) if isinstance(test_data, to.Tensor) else test_data
        self.can_train = train_data is not None and train_states is not None
        self.can_test = test_data is not None and test_states is not None
        if not self.can_train and not self.can_test:  # pragma: no cover
            raise RuntimeError("Please provide at least one pair of dataset and variational states")

        _d, _s = (train_data, train_states) if self.can_train else (test_data, test_states)
        assert _d is not None and _s is not None
        if isinstance(model, Optimized):
            model.init_storage(_s.config["S"], _s.config["S_new"], _d.batch_size)

        self.model = model
        self.train_data = train_data
        self.train_states = train_states
        self.test_data = test_data
        self.test_states = test_states
        self.will_reconstruct = will_reconstruct
        self.eval_F_at_epoch_end = eval_F_at_epoch_end
        if train_data is not None:
            self.N_train = to.tensor(len(train_data.dataset))
            all_reduce(self.N_train)
            self.N_train = self.N_train.item()
            if self.will_reconstruct:
                self.train_reconstruction = train_data.dataset.tensors[1].clone()
        if test_data is not None:
            self.N_test = to.tensor(len(test_data.dataset))
            all_reduce(self.N_test)
            self.N_test = self.N_test.item()
            if self.will_reconstruct:
                self.test_reconstruction = test_data.dataset.tensors[1].clone()
        self._to_rollback = rollback_if_F_decreases
        self.data_transform = data_transform if data_transform is not None else lambda x: x

    @staticmethod
    def _do_e_step(
        data: TVEMDataLoader,
        states: TVEMVariationalStates,
        model: Trainable,
        N: int,
        data_transform,
        reconstruction: to.Tensor = None,
    ):
        if reconstruction is not None and not isinstance(model, Reconstructor):
            raise NotImplementedError(
                f"reconstruction not implemented for model {type(model).__name__}"
            )
        F = to.tensor(0.0)
        subs = to.tensor(0)
        if isinstance(model, Optimized):
            model.init_epoch()
        for idx, batch in data:
            batch = data_transform(batch)
            if isinstance(model, Optimized):
                model.init_batch()
            subs += states.update(idx, batch, model)
            F += model.free_energy(idx, batch, states)
            if reconstruction is not None:
                # full data estimation
                reconstruction[idx] = model.data_estimator(idx, batch, states)  # type: ignore
        all_reduce(F)
        all_reduce(subs)
        return F.item() / N, subs.item() / N, reconstruction

    def e_step(self, compute_reconstruction: bool = False) -> Dict[str, Any]:
        """Run one epoch of E-steps on training and/or test data, depending on what is available.

        Only E-steps are executed.

        :returns: a dictionary containing 'train_F', 'train_subs', 'test_F', 'test_subs'
                  (keys might be missing depending on what is available)
        """
        ret = {}
        model = self.model
        train_data, train_states = self.train_data, self.train_states
        test_data, test_states = self.test_data, self.test_states
        train_reconstruction = (
            self.train_reconstruction
            if (compute_reconstruction and hasattr(self, "train_reconstruction"))
            else None
        )
        test_reconstruction = (
            self.test_reconstruction
            if (compute_reconstruction and hasattr(self, "test_reconstruction"))
            else None
        )

        # Training #
        if self.can_train:
            assert train_data is not None and train_states is not None  # to make mypy happy
            ret["train_F"], ret["train_subs"], train_rec = self._do_e_step(
                train_data,
                train_states,
                model,
                self.N_train,
                self.data_transform,
                train_reconstruction,
            )
            if train_rec is not None:
                ret["train_rec"] = train_rec

        # Validation/Testing #
        if self.can_test:
            assert test_data is not None and test_states is not None  # to make mypy happy
            ret["test_F"], ret["test_subs"], test_rec = self._do_e_step(
                test_data, test_states, model, self.N_test, self.data_transform, test_reconstruction
            )
            if test_rec is not None:
                ret["test_rec"] = test_rec

        return ret

    def em_step(self, compute_reconstruction: bool = False) -> Dict[str, Any]:
        """Run one training and/or test epoch, depending on what data is available.

        Both E-step and M-step are executed. Eventually reconstructions are computed intermediately.

        :returns: a dictionary containing 'train_F', 'train_subs', 'test_F', 'test_subs'
                  (keys might be missing depending on what is available). The free energy values
                  are calculated per batch, so if the model updates its parameters in
                  `update_param_epoch`, the free energies reported at epoch X are calculated
                  using the weights of epoch X-1.
        """
        # NOTE:
        # For models that update the parameters in update_param_epoch, the free energy reported at
        # each epoch is the one after the E-step and before the M-step (K sets of epoch X and
        # \Theta of epoch X-1 yield free energy of epoch X).
        # For models that update the parameters in update_param_batch, the free energy reported
        # at each epoch does not correspond to a fixed set of parameters: each batch had a
        # different set of parameters and the reported free energy is more of an average of the
        # free energies yielded by all the sets of parameters spanned during an epoch.

        ret_dict = {}

        # Training #
        if self.can_train:
            F, subs, reco = self._train_epoch(compute_reconstruction)
            all_reduce(F)
            ret_dict["train_F"] = F.item() / self.N_train
            all_reduce(subs)
            ret_dict["train_subs"] = subs.item() / self.N_train
            if reco is not None:
                ret_dict["train_rec"] = reco

        # Validation/Testing #
        if self.can_test:
            test_data, test_states, test_reconstruction = (
                self.test_data,
                self.test_states,
                self.test_reconstruction
                if (compute_reconstruction and hasattr(self, "test_reconstruction"))
                else None,
            )
            model = self.model

            assert test_data is not None and test_states is not None  # to make mypy happy
            res = self._do_e_step(
                test_data, test_states, model, self.N_test, self.data_transform, test_reconstruction
            )
            ret_dict["test_F"], ret_dict["test_subs"], test_rec = res
            if test_reconstruction is not None:
                ret_dict["test_rec"] = test_reconstruction

        return ret_dict

    def _train_epoch(self, compute_reconstruction: bool):
        model = self.model
        train_data, train_states, train_reconstruction = (
            self.train_data,
            self.train_states,
            self.train_reconstruction
            if (compute_reconstruction and hasattr(self, "train_reconstruction"))
            else None,
        )

        assert train_data is not None and train_states is not None  # to make mypy happy
        F = to.tensor(0.0, device=tvem.get_device())
        subs = to.tensor(0)
        if isinstance(model, Optimized):
            model.init_epoch()
        for idx, batch in train_data:
            batch = self.data_transform(batch)
            if isinstance(model, Optimized):
                model.init_batch()
            with to.no_grad():
                subs += train_states.update(idx, batch, model)
                if train_reconstruction is not None:
                    assert isinstance(model, Reconstructor)
                    train_reconstruction[idx] = model.data_estimator(
                        idx, batch, train_states
                    )  # full data estimation
            if to.isnan(batch).any():
                missing_data_mask = to.isnan(batch)
                batch[missing_data_mask] = train_reconstruction[idx][missing_data_mask]
                train_reconstruction[idx] = batch
            batch_F = model.update_param_batch(idx, batch, train_states)
            if not self.eval_F_at_epoch_end:
                if batch_F is None:
                    batch_F = model.free_energy(idx, batch, train_states)
                F += batch_F
        self._update_parameters_with_rollback()
        return F, subs, train_reconstruction

    def eval_free_energies(self) -> Dict[str, Any]:
        """Return a dictionary with the same contents as e_step/em_step, without training the model.

        :returns: a dictionary containing 'train_F', 'train_subs', 'test_F', 'test_subs'
                  (keys might be missing depending on what is available)
        """
        m = self.model
        train_data, train_states = self.train_data, self.train_states
        test_data, test_states = self.test_data, self.test_states
        lpj_fn = m.log_pseudo_joint if isinstance(m, Optimized) else m.log_joint
        ret = {}

        if self.can_train:
            assert train_data is not None and train_states is not None  # to make mypy happy
            F = to.tensor(0.0)
            if isinstance(m, Optimized):
                m.init_epoch()
            for idx, batch in train_data:
                batch = self.data_transform(batch)
                if isinstance(m, Optimized):
                    m.init_batch()
                train_states.lpj[idx] = lpj_fn(batch, train_states.K[idx])
                F += m.free_energy(idx, batch, train_states)
            all_reduce(F)
            ret["train_F"] = F.item() / self.N_train
            ret["train_subs"] = 0

        if self.can_test:
            assert test_data is not None and test_states is not None  # to make mypy happy
            F = to.tensor(0.0)
            if isinstance(m, Optimized):
                m.init_epoch()
            for idx, batch in test_data:
                batch = self.data_transform(batch)
                if isinstance(m, Optimized):
                    m.init_batch()
                test_states.lpj[idx] = lpj_fn(batch, test_states.K[idx])
                F += m.free_energy(idx, batch, test_states)
            all_reduce(F)
            ret["test_F"] = F.item() / self.N_test
            ret["test_subs"] = 0

        return ret

    def _update_parameters_with_rollback(self) -> None:
        """Update model parameters calling `update_param_epoch`, roll back if F decreases."""

        if len(self._to_rollback) == 0:
            # nothing to rollback, fall back to simple parameter update
            self.model.update_param_epoch()
            return

        m = self.model
        lpj_fn = m.log_pseudo_joint if isinstance(m, Optimized) else m.log_joint

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

