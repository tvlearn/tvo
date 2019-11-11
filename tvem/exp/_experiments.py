# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC, abstractmethod
from tvem.utils.data import TVEMDataLoader
from tvem.models import TVEMModel
from tvem.utils.parallel import pprint, init_processes, gather_from_processes
from tvem.exp._utils import make_var_states, get_h5_dataset_to_processes
from tvem.utils import get, H5Logger
from tvem.trainer import Trainer
from tvem.exp._EStepConfig import EStepConfig
from tvem.exp._ExpConfig import ExpConfig
from tvem.exp._EpochLog import EpochLog
from tvem.variational import TVEMVariationalStates
import tvem

import math
from typing import Dict, Any, Generator
import torch as to
import torch.distributed as dist
import time
from pathlib import Path
import os
from munch import Munch


class Experiment(ABC):
    """Abstract base class for all experiments."""

    @abstractmethod
    def run(self, epochs: int) -> Generator[EpochLog, None, None]:
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
        self._conf = Munch(conf.as_dict())
        self._conf.model = type(model).__name__
        self._estep_conf = Munch(estep_conf.as_dict())
        model.init_storage(
            self._estep_conf.n_states, self._estep_conf.n_new_states, self._conf.batch_size
        )

        self.train_data = None
        self.train_states = None
        if train_dataset is not None:
            self.train_data = self._make_dataloader(train_dataset, conf)
            # might differ between processes: last process might have smaller N and less states
            # (but TVEMDataLoader+ShufflingSampler make sure the number of batches is the same)
            N = train_dataset.shape[0]
            self.train_states = self._make_states(N, H, conf.precision, estep_conf)

        self.test_data = None
        self.test_states = None
        if test_dataset is not None:
            self.test_data = self._make_dataloader(test_dataset, conf)
            N = test_dataset.shape[0]
            self.test_states = self._make_states(N, H, conf.precision, estep_conf)

    def _make_dataloader(self, dataset: to.Tensor, conf: ExpConfig) -> TVEMDataLoader:
        if dataset.dtype is not to.uint8:
            dataset = dataset.to(dtype=conf.precision)
            assert dataset.dtype is self.model.precision
        dataset = dataset.to(device=tvem.get_device())
        return TVEMDataLoader(
            dataset, batch_size=conf.batch_size, shuffle=conf.shuffle, drop_last=conf.drop_last
        )

    def _make_states(
        self, N: int, H: int, precision: to.dtype, estep_conf: EStepConfig
    ) -> TVEMVariationalStates:
        states = make_var_states(estep_conf, N, H, precision)
        assert states.precision is self.model.precision
        return states

    @property
    def conf(self) -> Dict[str, Any]:
        return dict(self._conf)

    @property
    def estep_conf(self) -> Dict[str, Any]:
        return dict(self._estep_conf)

    def run(self, epochs: int) -> Generator[EpochLog, None, None]:
        """Run training and/or testing.

        :param epochs: Number of epochs to train for
        """
        will_reconstruct = (
            self._conf.reco_epochs is not None or self._conf.warmup_reco_epochs is not None
        )
        trainer = Trainer(
            self.model,
            self.train_data,
            self.train_states,
            self.test_data,
            self.test_states,
            rollback_if_F_decreases=self._conf.rollback_if_F_decreases,
            will_reconstruct=will_reconstruct,
        )
        logger = H5Logger(self._conf.output, blacklist=self._conf.log_blacklist)

        self._log_confs(logger)

        # warm-up E-steps
        if self._conf.warmup_Esteps > 0:
            pprint("Warm-up E-steps")
        for e in range(self._conf.warmup_Esteps):
            compute_reconstruction = (
                self._conf.warmup_reco_epochs is not None and e in self._conf.warmup_reco_epochs
            )
            d = trainer.e_step(compute_reconstruction)
            self._log_epoch(logger, d)

        # log initial free energies (after warm-up E-steps if any)
        if self._conf.warmup_Esteps == 0:
            d = trainer.eval_free_energies()
        self._log_epoch(logger, d)
        yield EpochLog(epoch=0, results=d)

        # EM steps
        for e in range(epochs):
            start_t = time.time()
            compute_reconstruction = (
                self._conf.reco_epochs is not None and e in self._conf.reco_epochs
            )
            d = trainer.em_step(compute_reconstruction)
            epoch_runtime = time.time() - start_t
            self._log_epoch(logger, d)
            yield EpochLog(e + 1, d, epoch_runtime)

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

    def _log_epoch(self, logger: H5Logger, epoch_results: Dict[str, float]):
        """Log F, subs, model.theta, states.K and states.lpj to file, return printable log.

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
            F_and_subs_dict = {f"{log_kind}_F": to.tensor(F), f"{log_kind}_subs": to.tensor(subs)}
            logger.append(**F_and_subs_dict)

            # log latest states and lpj to file
            states = getattr(self, f"{data_kind}_states")
            states_and_lpj_dict = {
                f"{log_kind}_states": gather_from_processes(states.K),
                f"{log_kind}_lpj": gather_from_processes(states.lpj),
            }
            logger.set(**states_and_lpj_dict)

            # log data reconstructions
            reco_dict = {}
            if (
                f"{log_kind}_reconstruction" not in self._conf.log_blacklist
                and f"{log_kind}_rec" in epoch_results
            ):
                reco_dict[f"{log_kind}_reconstruction"] = gather_from_processes(
                    epoch_results[f"{log_kind}_rec"]
                )
                logger.set(**reco_dict)

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
        train_dataset = get_h5_dataset_to_processes(train_data_file, ("train_data", "data"))
        val_dataset = None
        if val_data_file is not None:
            val_dataset = get_h5_dataset_to_processes(val_data_file, ("val_data", "data"))

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
        dataset = get_h5_dataset_to_processes(data_file, ("test_data", "data"))

        setattr(conf, "test_dataset", data_file)
        super().__init__(conf, estep_conf, model, None, dataset)
