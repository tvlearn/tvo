# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import tvem
from tvem.utils.model_protocols import Trainable, Optimized, Reconstructor
from tvem.variational import TVEMVariationalStates
from tvem.utils.data import TVEMDataLoader
from tvem.utils.parallel import all_reduce
from typing import Dict, Any, Sequence, Union, Callable
from tvem.trainer.Trainer import Trainer
import torch as to


class REM1_Trainer(Trainer):
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
        data_transform: Callable[[to.Tensor], to.Tensor] = None
    ):
        """Train  and/or test a model with REM1 according to Sahini 1999
        :params: See Trainer Class
        :param beta: torch tensor with temperatures of REM-steps
        :param beta_steps: number of steps for the temperatues
        """
        super().__init__(model, 
            train_data,
            train_states,
            test_data,
            test_states,
            rollback_if_F_decreases,
            will_reconstruct,
            eval_F_at_epoch_end,
            data_transform
            )
        
    def em_step(self, compute_reconstruction: bool = False, beta: float = 1.0) -> Dict[str, Any]:
        """Run a training or test epoch of REM1 with annealing-sheme
        
        :param compute_reconstruction: 
        :param beta: tensor of temperatures between 0 and 1 discribing the temperatures  for annealing
        :param beta_steps: tensor of same size as beta containing the number of epochs per temperature
        
        :returns: a dictionary containing 'train_F', 'train_subs', 'test_F', 'test_subs'
                  (keys might be missing depending on what is available). The free energy values
                  are calculated per batch, so if the model updates its parameters in
                  `update_param_epoch`, the free energies reported at epoch X are calculated
                  using the weights of epoch X-1.
        """
        ret_dict = {}
        # Training #
        if self.can_train:
            F_beta, F, subs, reco = self._train_epoch(compute_reconstruction, beta)
            all_reduce(F)
            ret_dict["train_F"] = F.item() / self.N_train
            ret_dict["train_F_beta"] = F_beta.item() / self.N_train
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

    def _train_epoch(self, compute_reconstruction: bool, beta: float = 1.0):
        model = self.model
        train_data, train_states, train_reconstruction = (
            self.train_data,
            self.train_states,
            self.train_reconstruction
            if (compute_reconstruction and hasattr(self, "train_reconstruction"))
            else None,
        )

        assert train_data is not None and train_states is not None  # to make mypy happy
        F_beta = to.tensor(0.0, device=tvem.get_device())
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
            batch_F_beta = model.update_param_batch(idx, batch, train_states, beta)
            if not self.eval_F_at_epoch_end:
                if batch_F_beta is None:
                    batch_F_beta = model.annealed_free_energy(idx, batch, train_states, beta)
                    batch_F = model.free_energy(idx, batch, train_states)
                F_beta += batch_F_beta
                F += batch_F
        self._update_parameters_with_rollback()
        return F_beta, F, subs, train_reconstruction

    def eval_free_energies(self, beta) -> Dict[str, Any]:
        """Return a dictionary with the same contents as e_step/em_step, without training the model.

        :returns: a dictionary containing 'train_F_beta, train_F', 'train_subs', 'test_F', 'test_subs'
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
            F_beta = to.tensor(0.0)
            if isinstance(m, Optimized):
                m.init_epoch()
            for idx, batch in train_data:
                batch = self.data_transform(batch)
                if isinstance(m, Optimized):
                    m.init_batch()
                train_states.lpj[idx] = lpj_fn(batch, train_states.K[idx])
                F += m.free_energy(idx, batch, train_states)
                F_beta += m.annealed_free_energy(idx, batch, train_states, beta)
            all_reduce(F)
            all_reduce(F_beta)
            ret["train_F"] = F.item() / self.N_train
            ret["train_F_beta"] = F_beta.item() / self.N_train
            ret["train_subs"] = 0

        if self.can_test:
            assert test_data is not None and test_states is not None  # to make mypy happy
            F = to.tensor(0.0)
            F_beta = to.tensor(0.0)
            if isinstance(m, Optimized):
                m.init_epoch()
            for idx, batch in test_data:
                batch = self.data_transform(batch)
                if isinstance(m, Optimized):
                    m.init_batch()
                test_states.lpj[idx] = lpj_fn(batch, test_states.K[idx])
                F += m.free_energy(idx, batch, test_states)
                F_beta += m.annealed_free_energy(idx, batch, train_states, beta)
            all_reduce(F)
            all_reduce(F_beta)
            ret["test_F"] = F.item() / self.N_test
            ret["test_F_beta"] = F_beta.item() / self.N_test
            ret["test_subs"] = 0

        return ret
