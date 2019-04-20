# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC, abstractmethod
from tvem.variational import EEMVariationalStates
from tvem.util.data import TVEMDataLoader
from tvem.models import TVEMModel
from tvem.trainer import Trainer
from tvem.util.parallel import pprint, init_processes, scatter2processes
from tvem.util import get
from tvem.exp._EStepConfig import EEMConfig, EStepConfig
import tvem

import math
import h5py
from typing import Tuple, Dict, Any
import torch as to
import torch.distributed as dist
import numpy as np


def _make_var_states(conf: EStepConfig, N: int, H: int, dtype: to.dtype) -> EEMVariationalStates:
    if isinstance(conf, EEMConfig):
        return _make_EEM_var_states(conf, N, H, dtype)
    else:
        raise RuntimeError('Unimplemented')


def _make_EEM_var_states(conf: EEMConfig, N: int, H: int, dtype: to.dtype):
    selection = {'fitness': 'batch_fitparents', 'uniform': 'randparents'}[conf.parent_selection]
    mutation = {'sparsity': 'sparseflip', 'uniform': 'randflip'}[conf.mutation]
    if conf.crossover:
        mutation = 'cross_' + mutation
    eem_conf = {'parent_selection': selection, 'mutation': mutation,
                'n_parents': conf.n_parents, 'n_children': conf.n_children,
                'n_generations': conf.n_generations, 'S': conf.n_states,
                'N': N, 'H': H, 'dtype': dtype}
    return EEMVariationalStates(eem_conf)


class Experiment(ABC):
    """Abstract base class for all experiments."""
    @abstractmethod
    def run(self, epochs: int):
        pass  # pragma: no cover


class _TrainingAndOrValidation(Experiment):
    def __init__(self, conf: Dict[str, Any], estep_conf: EStepConfig, model: TVEMModel,
                 train_dataset: to.Tensor = None, test_dataset: to.Tensor = None):
        """Helper class to avoid code repetition between Training and Testing.

        It performs training and/or validation/testings depending on what input is provided.
        """
        required_keys = ('precision', 'batch_size', 'shuffle', 'warmup_Esteps')
        for k in required_keys:
            assert k in conf
        assert conf['precision'] in (to.float32, to.float64)
        dtype = conf['precision']
        H = sum(model.shape[1:])
        self.model = model
        self.warmup_Esteps = conf['warmup_Esteps']

        self.train_data = None
        self.train_states = None
        if train_dataset is not None:
            if train_dataset.dtype is not to.uint8:
                train_dataset = train_dataset.to(dtype=dtype)
            train_dataset = train_dataset.to(device=tvem.get_device())
            self.train_data = TVEMDataLoader(train_dataset, batch_size=conf['batch_size'],
                                             shuffle=conf['shuffle'])
            N = train_dataset.shape[0]
            self.train_states = _make_var_states(estep_conf, N, H, dtype)
            assert self.train_states.precision is self.model.precision
            if train_dataset.dtype is not to.uint8:
                assert self.model.precision is self.train_data.precision
        self.test_data = None
        self.test_states = None
        if test_dataset is not None:
            if test_dataset.dtype is not to.uint8:
                test_dataset = test_dataset.to(dtype=dtype)
            test_dataset = test_dataset.to(device=tvem.get_device())
            self.test_data = TVEMDataLoader(test_dataset, batch_size=conf['batch_size'],
                                            shuffle=conf['shuffle'])
            N = test_dataset.shape[0]
            self.test_states = _make_var_states(estep_conf, N, H, dtype)
            assert self.test_states.precision is self.model.precision
            if test_dataset.dtype is not to.uint8:
                assert self.model.precision is self.test_data.precision

    def run(self, epochs: int):
        """Run training and/or testing.

        :param epochs: Number of epochs to train for
        """
        trainer = Trainer(self.model, self.train_data, self.train_states,
                          self.test_data, self.test_states)
        for e in range(self.warmup_Esteps):
            trainer.e_step()
        for e in range(epochs):
            pprint(f'epoch {e}')
            d = trainer.em_step()  # E- and M-step on training set, E-step on validation/test set
            if self.train_data is not None:
                F, subs = get(d, 'train_F', 'train_subs')
                assert not (math.isnan(F) or math.isinf(F)), 'training free energy is nan!'
                pprint(f'\ttrain F/N: {F:<10.5f} avg subs: {subs:<6.2f}')
            if self.test_data is not None:
                F, subs = get(d, 'test_F', 'test_subs')
                test_or_valid = 'valid.' if self.train_data is not None else 'test'
                assert not (math.isnan(F) or math.isinf(F)), f'{test_or_valid} free energy is nan!'
                pprint(f'\t{test_or_valid} F/N: {F:<10.5f} avg subs: {subs:<6.2f}')


def _get_h5_dataset_to_processes(fname: str, possible_keys: Tuple[str, ...]) -> to.Tensor:
    """Return dataset with the first of `possible_keys` that is found in hdf5 file `fname`."""
    rank = dist.get_rank() if dist.is_initialized() else 0

    f = h5py.File(fname, 'r')
    for dataset in possible_keys:
        if dataset in f.keys():
            break
    else:  # pragma: no cover
        raise RuntimeError(f'File "{fname}" does not contain any of keys {possible_keys}')
    if rank == 0:
        data = to.tensor(f[dataset], device=tvem.get_device())
        dtype = data.dtype
    else:
        data = None
        # convert h5py dtype to torch dtype passing through numpy
        dtype = to.from_numpy(np.empty(0, dtype=f[dataset].dtype)).dtype
    return scatter2processes(data, dtype=dtype, device=tvem.get_device())


class Training(_TrainingAndOrValidation):
    def __init__(self, conf: Dict[str, Any], estep_conf: EStepConfig, model: TVEMModel,
                 train_data_file: str, val_data_file: str = None):
        """Train model on given dataset for the given number of epochs.

        :param conf: Experiment configuration.
                     Optional keys (if not present, the corresponding default will be used):
                     - precision: Must be one of torch.float32 or torch.float64. It's the floating
                                  point precision that will be used throughout the experiment for
                                  all quantities. Defaults to torch.float64 (double precision).
                     - batch_size: Batch size for the data loaders.
                     - shuffle (optional): Whether data should be reshuffled at every epoch.
                       Defaults to true. See also torch's `DataLoader docs`_
                     - warmup_Esteps (optional): Number of warm-up E-steps to perform.
                       Defaults to 0.
        :param estep_conf: Instance of a class inheriting from EStepConfig.
        :param model: TVEMModel to train
        :param train_data_file: Path to an HDF5 file containing the training dataset.
                                Datasets with name "train_data" and "data" will be
                                searched in the file, in this order.
        :param n_train_states: Number of TVEM variational states to use for training.
        :param val_data_file: Path to an HDF5 file containing the training dataset.
                              Datasets with name "val_data" and "data" will be searched in the file,
                              in this order.
        :param n_val_states: Number of TVEM variational states to use for validation.

        On the validation dataset, Training only performs E-steps without updating
        the model parameters.

        .. _DataLoader docs: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """
        if tvem.get_run_policy() == 'mpi':
            init_processes()
        train_dataset = _get_h5_dataset_to_processes(train_data_file, ('train_data', 'data'))
        val_dataset = None
        if val_data_file is not None:
            val_dataset = _get_h5_dataset_to_processes(val_data_file, ('val_data', 'data'))

        if 'shuffle' not in conf:
            conf['shuffle'] = True
        if 'warmup_Esteps' not in conf:
            conf['warmup_Esteps'] = 0
        super().__init__(conf, estep_conf, model, train_dataset, val_dataset)


class Testing(_TrainingAndOrValidation):
    def __init__(self, conf: Dict[str, Any], estep_conf: EStepConfig, model: TVEMModel,
                 data_file: str):
        """Test given model on given dataset for the given number of epochs.

        :param conf: Experiment configuration.
                     Optional keys (if not present, the corresponding default will be used):
                     - precision: Must be one of torch.float32 or torch.float64. It's the floating
                                  point precision that will be used throughout the experiment for
                                  all quantities. Defaults to torch.float64 (double precision).
                     - batch_size: Batch size for the data loaders.
                     - warmup_Esteps (optional): Number of warm-up E-steps to perform.
                       Defaults to 0.
        :param estep_conf: Instance of a class inheriting from EStepConfig.
        :param model: TVEMModel to test
        :param data_file: Path to an HDF5 file containing the training dataset. Datasets with name
                          "test_data" and "data" will be searched in the file, in this order.
        :param n_states: Number of TVEM variational states to use for testing.

        Only E-steps are run. Model parameters are not updated.
        """
        if tvem.get_run_policy() == 'mpi':
            init_processes()
        dataset = _get_h5_dataset_to_processes(data_file, ('test_data', 'data'))
        conf['shuffle'] = False
        if 'warmup_Esteps' not in conf:
            conf['warmup_Esteps'] = 0
        super().__init__(conf, estep_conf, model, None, dataset)
