# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC, abstractmethod
from tvem.utils.data import TVEMDataLoader, H5Logger
from tvem.models import TVEMModel
from tvem.trainer import Trainer
from tvem.utils.parallel import pprint, init_processes, gather_from_processes
from tvem.exp._utils import _make_var_states, _get_h5_dataset_to_processes
from tvem.utils import get
from tvem.exp._EStepConfig import EStepConfig
from tvem.exp._ExpConfig import ExpConfig
import tvem

import math
from typing import Dict, Any
import torch as to
import torch.distributed as dist
import time
from pathlib import Path
import os
from munch import Munch


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
        H = sum(model.shape[1:])
        self.model = model
        self._conf = Munch(vars(conf))
        self._conf.model = type(model).__name__
        self._estep_conf = Munch(vars(estep_conf))

        self.train_data = None
        self.train_states = None
        if train_dataset is not None:
            if train_dataset.dtype is not to.uint8:
                train_dataset = train_dataset.to(dtype=conf.precision)
            train_dataset = train_dataset.to(device=tvem.get_device())
            self.train_data = TVEMDataLoader(
                train_dataset,
                batch_size=conf.batch_size,
                shuffle=conf.shuffle,
                drop_last=conf.drop_last,
            )
            N = train_dataset.shape[0]
            self.train_states = _make_var_states(estep_conf, N, H, conf.precision)
            self._estep_conf.estep_type = type(self.train_states).__name__
            assert self.train_states.precision is self.model.precision
            if train_dataset.dtype is not to.uint8:
                assert self.model.precision is self.train_data.precision

        self.test_data = None
        self.test_states = None
        if test_dataset is not None:
            if test_dataset.dtype is not to.uint8:
                test_dataset = test_dataset.to(dtype=conf.precision)
            test_dataset = test_dataset.to(device=tvem.get_device())
            self.test_data = TVEMDataLoader(
                test_dataset,
                batch_size=conf.batch_size,
                shuffle=conf.shuffle,
                drop_last=conf.drop_last,
            )
            N = test_dataset.shape[0]
            self.test_states = _make_var_states(estep_conf, N, H, conf.precision)
            assert self.test_states.precision is self.model.precision
            if test_dataset.dtype is not to.uint8:
                assert self.model.precision is self.test_data.precision

    @property
    def conf(self) -> Dict[str, Any]:
        return dict(self._conf)

    @property
    def estep_conf(self) -> Dict[str, Any]:
        return dict(self._estep_conf)

    def run(self, epochs: int):
        """Run training and/or testing.

        :param epochs: Number of epochs to train for
        """
        trainer = Trainer(
            self.model, self.train_data, self.train_states, self.test_data, self.test_states
        )
        logger = H5Logger(self._conf.output, blacklist=self._conf.log_blacklist)

        self._log_confs(logger)

        # warm-up E-steps
        if self._conf.warmup_Esteps > 0:
            pprint("Warm-up E-steps")
        for e in range(self._conf.warmup_Esteps):
            d = trainer.e_step()
            self._log_epoch(logger, d)

        # log initial model parameters
        logger.append(theta=self.model.theta)

        # EM steps
        for e in range(epochs):
            pprint(f"epoch {e}")
            start_t = time.time()
            d = trainer.em_step()  # E- and M-step on training set, E-step on validation/test set
            end_t = time.time()
            self._log_epoch(logger, d, epoch_runtime=end_t - start_t)

        # remove leftover ".old" logfiles produced by the logger
        rank = dist.get_rank() if dist.is_initialized() else 0
        leftover_logfile = self._conf.output + ".old"
        if rank == 0 and Path(leftover_logfile).is_file():
            os.remove(leftover_logfile)

    def _log_confs(self, logger: H5Logger):
        """Dump experiment+estep configuration to screen and save it to output file."""
        pprint("\nExperiment configuration:")
        for k, v in self.conf.items():
            pprint(f"\t{k:<20}: {v}")
        logger.set(exp_config=self.conf)
        pprint("E-step configuration:")
        for k, v in self.estep_conf.items():
            pprint(f"\t{k:<20}: {v}")
        pprint()
        logger.set(estep_config=self.estep_conf)

    def _log_epoch(
        self, logger: H5Logger, epoch_results: Dict[str, float], epoch_runtime: float = None
    ):
        """Log F, subs, model.theta, states.K and states.lpj to file.

        :param logger: the logger for this run
        :param epoch_results: dictionary returned by Trainer.e_step or Trainer.em_step
        :param epoch_runtime: wall-clock duration of the epoch
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

            # log latest states and lpj to file
            states = getattr(self, f"{data_kind}_states")
            states_and_lpj_dict = {
                f"{log_kind}_states": gather_from_processes(states.K),
                f"{log_kind}_lpj": gather_from_processes(states.lpj),
            }
            logger.set(**states_and_lpj_dict)

        if epoch_runtime is not None:
            pprint(f"\ttotal epoch runtime: {epoch_runtime:<5.2f} s")

        logger.append(theta=self.model.theta)
        logger.write()


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
        :param val_data_file: Path to an HDF5 file containing the training dataset.
                              Datasets with name "val_data" and "data" will be searched in the file,
                              in this order.

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

        setattr(conf, "train_dataset", train_data_file)
        setattr(conf, "val_dataset", val_data_file)
        super().__init__(conf, estep_conf, model, train_dataset, val_dataset)


class Testing(_TrainingAndOrValidation):
    def __init__(self, conf: ExpConfig, estep_conf: EStepConfig, model: TVEMModel, data_file: str):
        """Test given model on given dataset for the given number of epochs.

        :param conf: Experiment configuration.
        :param estep_conf: Instance of a class inheriting from EStepConfig.
        :param model: TVEMModel to test
        :param data_file: Path to an HDF5 file containing the training dataset. Datasets with name
                          "test_data" and "data" will be searched in the file, in this order.

        Only E-steps are run. Model parameters are not updated.
        """
        if tvem.get_run_policy() == "mpi":
            init_processes()
        dataset = _get_h5_dataset_to_processes(data_file, ("test_data", "data"))

        setattr(conf, "test_dataset", data_file)
        super().__init__(conf, estep_conf, model, None, dataset)
