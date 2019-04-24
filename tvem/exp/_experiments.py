# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC, abstractmethod
from tvem.variational import EEMVariationalStates
from tvem.util.data import TVEMDataLoader, H5Logger
from tvem.models import TVEMModel
from tvem.trainer import Trainer
from tvem.util.parallel import pprint, init_processes, scatter_to_processes
from tvem.util import get
from tvem.exp._EStepConfig import EEMConfig, EStepConfig
import tvem

import math
import h5py
from typing import Tuple, Dict, Iterable
import torch as to
import torch.distributed as dist


def _make_var_states(conf: EStepConfig, N: int, H: int, dtype: to.dtype) -> EEMVariationalStates:
    if isinstance(conf, EEMConfig):
        return _make_EEM_var_states(conf, N, H, dtype)
    else:  # pragma: no cover
        raise NotImplementedError()


def _make_EEM_var_states(conf: EEMConfig, N: int, H: int, dtype: to.dtype):
    selection = {"fitness": "batch_fitparents", "uniform": "randparents"}[conf.parent_selection]
    mutation = {"sparsity": "sparseflip", "uniform": "randflip"}[conf.mutation]
    if conf.crossover:
        mutation = "cross_" + mutation
    eem_conf = {
        "parent_selection": selection,
        "mutation": mutation,
        "n_parents": conf.n_parents,
        "n_children": conf.n_children,
        "n_generations": conf.n_generations,
        "S": conf.n_states,
        "N": N,
        "H": H,
        "dtype": dtype,
    }
    return EEMVariationalStates(eem_conf)


class ExpConfig:
    def __init__(
        self,
        batch_size: int = 1,
        precision: to.dtype = to.float64,
        shuffle: bool = True,
        drop_last: bool = False,
        warmup_Esteps: int = 0,
        output: str = "tvem_exp.h5",
        log_blacklist: Iterable[str] = [],
    ):
        """Configuration object for Experiment classes.

        :param batch_size: Batch size for the data loaders.
        :param precision: Must be one of torch.float32 or torch.float64. It's the floating
                          point precision that will be used throughout the experiment for
                          all quantities.
        :param shuffle: Whether data should be reshuffled at every epoch.
                        See also torch's `DataLoader docs`_.
        :param drop_last: set to True to drop the last incomplete batch, if the dataset size is not
                          divisible by the batch size. See also torch's `DataLoader docs`_.
        :param warmup_Esteps: Number of warm-up E-steps to perform.
        :param output: Name or path of output HDF5 file. It is overwritten if it already exists.
        :param log_blacklist: By default, experiments log all available quantities. These are:
                              - "{train,valid,test}_F": one or more of training/validation/test
                                free energy, depending on the experiment
                              - "{train,valid,test}_subs": average variational state substitutions
                                per datapoint (which ones are available depends on the experiment)
                              - "{train,valid,test}_states": latest snapshot of variational states
                                                             per datapoint
                              - "{train,valid,test}_lpj": latest snapshot of log-pseudo-joints
                                                          per datapoint
                              - "theta": a group containing logs of whatever model.theta contains
                              If one of these names appears in `log_blacklist`, the corresponing
                              quantity will not be logged.
        """
        assert precision in (to.float32, to.float64), "Precision must be one of torch.float{32,64}"
        self.batch_size = batch_size
        self.precision = precision
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.warmup_Esteps = warmup_Esteps
        self.output = output
        self.log_blacklist = log_blacklist


class Experiment(ABC):
    """Abstract base class for all experiments."""

    @abstractmethod
    def run(self, epochs: int):
        pass  # pragma: no cover


class _TrainingAndOrValidation(Experiment):
    def __init__(
        self,
        conf: ExpConfig,
        estep_conf: EStepConfig,
        model: TVEMModel,
        train_dataset: to.Tensor = None,
        test_dataset: to.Tensor = None,
    ):
        """Helper class to avoid code repetition between Training and Testing.

        It performs training and/or validation/testings depending on what input is provided.
        """
        dtype = conf.precision
        H = sum(model.shape[1:])
        self.model = model
        self.warmup_Esteps = conf.warmup_Esteps
        self.out_fname = conf.output
        self.log_blacklist = conf.log_blacklist

        self.train_data = None
        self.train_states = None
        if train_dataset is not None:
            if train_dataset.dtype is not to.uint8:
                train_dataset = train_dataset.to(dtype=dtype)
            train_dataset = train_dataset.to(device=tvem.get_device())
            self.train_data = TVEMDataLoader(
                train_dataset,
                batch_size=conf.batch_size,
                shuffle=conf.shuffle,
                drop_last=conf.drop_last,
            )
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
            self.test_data = TVEMDataLoader(
                test_dataset,
                batch_size=conf.batch_size,
                shuffle=conf.shuffle,
                drop_last=conf.drop_last,
            )
            N = test_dataset.shape[0]
            self.test_states = _make_var_states(estep_conf, N, H, dtype)
            assert self.test_states.precision is self.model.precision
            if test_dataset.dtype is not to.uint8:
                assert self.model.precision is self.test_data.precision

    def run(self, epochs: int):
        """Run training and/or testing.

        :param epochs: Number of epochs to train for
        """
        trainer = Trainer(
            self.model, self.train_data, self.train_states, self.test_data, self.test_states
        )
        logger = H5Logger(self.out_fname, blacklist=self.log_blacklist)
        logger.set(warmup_Esteps=to.tensor(self.warmup_Esteps))

        # warm-up E-steps
        if self.warmup_Esteps > 0:
            pprint("Warm-up E-steps")
        for e in range(self.warmup_Esteps):
            d = trainer.e_step()
            self._log_epoch(logger, d)

        # log initial model parameters
        logger.append(theta=self.model.theta)

        # EM steps
        for e in range(epochs):
            pprint(f"epoch {e}")
            d = trainer.em_step()  # E- and M-step on training set, E-step on validation/test set
            self._log_epoch(logger, d)

    def _log_epoch(self, logger: H5Logger, epoch_results: Dict[str, float]):
        """Log F, subs, model.theta, states.K and states.lpj to file.

        :param logger: the logger for this run
        :param epoch_results: dictionary returned by Trainer.e_step or Trainer.em_step
        """
        for data_kind in "train", "test":
            if data_kind + "_F" not in epoch_results:
                continue

            # log_kind is one of "train", "valid" or "test"
            # (while data_kind is one of "train" or "test")
            log_kind = "valid" if data_kind == "test" and self.train_data is not None else data_kind

            # log F and subs to stdout and file
            F, subs = get(epoch_results, f"{data_kind}_F", f"{data_kind}_subs")
            assert not (math.isnan(F) or math.isinf(F)), f"{log_kind} free energy is invalid!"
            pprint(f"\t{log_kind} F/N: {F:<10.5f} avg subs: {subs:<6.2f}")
            F_and_subs_dict = {f"{log_kind}_F": to.tensor(F), f"{log_kind}_subs": to.tensor(subs)}
            logger.append(**F_and_subs_dict)

            # log latest states and lpj to file (FIXME must gather K and lpj from all processes)
            states = getattr(self, f"{data_kind}_states")
            states_and_lpj_dict = {f"{log_kind}_states": states.K, f"{log_kind}_lpj": states.lpj}
            logger.set(**states_and_lpj_dict)

        logger.append(theta=self.model.theta)
        logger.write()


def _get_h5_dataset_to_processes(fname: str, possible_keys: Tuple[str, ...]) -> to.Tensor:
    """Return dataset with the first of `possible_keys` that is found in hdf5 file `fname`."""
    rank = dist.get_rank() if dist.is_initialized() else 0

    f = h5py.File(fname, "r")
    for dataset in possible_keys:
        if dataset in f.keys():
            break
    else:  # pragma: no cover
        raise ValueError(f'File "{fname}" does not contain any of keys {possible_keys}')
    if rank == 0:
        data = to.tensor(f[dataset], device=tvem.get_device())
    else:
        data = None
    return scatter_to_processes(data)


class Training(_TrainingAndOrValidation):
    def __init__(
        self,
        conf: ExpConfig,
        estep_conf: EStepConfig,
        model: TVEMModel,
        train_data_file: str,
        val_data_file: str = None,
    ):
        """Train model on given dataset for the given number of epochs.

        :param conf: Experiment configuration.
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
        if tvem.get_run_policy() == "mpi":
            init_processes()
        train_dataset = _get_h5_dataset_to_processes(train_data_file, ("train_data", "data"))
        val_dataset = None
        if val_data_file is not None:
            val_dataset = _get_h5_dataset_to_processes(val_data_file, ("val_data", "data"))

        super().__init__(conf, estep_conf, model, train_dataset, val_dataset)


class Testing(_TrainingAndOrValidation):
    def __init__(self, conf: ExpConfig, estep_conf: EStepConfig, model: TVEMModel, data_file: str):
        """Test given model on given dataset for the given number of epochs.

        :param conf: Experiment configuration.
        :param estep_conf: Instance of a class inheriting from EStepConfig.
        :param model: TVEMModel to test
        :param data_file: Path to an HDF5 file containing the training dataset. Datasets with name
                          "test_data" and "data" will be searched in the file, in this order.
        :param n_states: Number of TVEM variational states to use for testing.

        Only E-steps are run. Model parameters are not updated.
        """
        if tvem.get_run_policy() == "mpi":
            init_processes()
        dataset = _get_h5_dataset_to_processes(data_file, ("test_data", "data"))
        super().__init__(conf, estep_conf, model, None, dataset)
