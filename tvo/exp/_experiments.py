# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC, abstractmethod
from tvo.utils.data import TVODataLoader
from tvo.utils.model_protocols import Trainable
from tvo.utils.parallel import (
    pprint,
    init_processes,
    gather_from_processes,
    get_h5_dataset_to_processes,
)
from tvo.exp._utils import make_var_states
from tvo.utils import get, H5Logger
from tvo.trainer import Trainer
from tvo.exp._EStepConfig import EStepConfig
from tvo.exp._ExpConfig import ExpConfig
from tvo.exp._EpochLog import EpochLog
from tvo.variational import TVOVariationalStates
import tvo

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
        model: Trainable,
        train_dataset: to.Tensor = None,
        test_dataset: to.Tensor = None,
        train_states: to.Tensor = None,
    ):
        """Helper class to avoid code repetition between Training and Testing.

        It performs training and/or validation/testings depending on what input is provided.
        """
        H = sum(model.shape[1:])
        self.model = model
        assert isinstance(model, Trainable)
        self._conf = Munch(conf.as_dict())
        self._conf.model = type(model).__name__
        self._conf.device = tvo.get_device().type
        self._estep_conf = Munch(estep_conf.as_dict())
        self.train_data = None
        self.train_states = None
        self._precision = model.precision
        if train_dataset is not None:
            self.train_data = self._make_dataloader(train_dataset, conf)
            # might differ between processes: last process might have smaller N and less states
            # (but TVODataLoader+ShufflingSampler make sure the number of batches is the same)
            N = train_dataset.shape[0]
            if hasattr(self._conf, "train_states") and self._conf.train_states is not None:
                self.train_states = self._conf.train_states
            else:
                self.train_states = self._make_states(
                    N, H, self._precision, estep_conf
                )  # init Variational States

        self.test_data = None
        self.test_states = None
        if test_dataset is not None:
            self.test_data = self._make_dataloader(test_dataset, conf)
            N = test_dataset.shape[0]
            self.test_states = self._make_states(N, H, self._precision, estep_conf)

        will_reconstruct = (
            self._conf.reco_epochs is not None or self._conf.warmup_reco_epochs is not None
        )
        self.trainer = Trainer(
            self.model,
            self.train_data,
            self.train_states,
            self.test_data,
            self.test_states,
            rollback_if_F_decreases=self._conf.rollback_if_F_decreases,
            will_reconstruct=will_reconstruct,
            eval_F_at_epoch_end=self._conf.eval_F_at_epoch_end,
            data_transform=self._conf.data_transform,
        )
        self.logger = H5Logger(self._conf.output, blacklist=self._conf.log_blacklist)

    def _make_dataloader(self, dataset: to.Tensor, conf: ExpConfig) -> TVODataLoader:
        if dataset.dtype is not to.uint8:
            dataset = dataset.to(dtype=self._precision)
        dataset = dataset.to(device=tvo.get_device())
        return TVODataLoader(
            dataset,
            batch_size=conf.batch_size,
            shuffle=conf.shuffle,
            drop_last=conf.drop_last,
        )

    def _make_states(
        self, N: int, H: int, precision: to.dtype, estep_conf: EStepConfig
    ) -> TVOVariationalStates:
        states = make_var_states(estep_conf, N, H, precision)
        return states

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._conf)

    @property
    def estep_config(self) -> Dict[str, Any]:
        return dict(self._estep_conf)

    def run(self, epochs: int) -> Generator[EpochLog, None, None]:
        """Run training and/or testing.

        :param epochs: Number of epochs to train for
        """
        trainer = self.trainer
        logger = self.logger

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

        # put trainer into undefined state after the experiment is finished
        # self.trainer = None  # type: ignore

    def _log_confs(self, logger: H5Logger):
        """Dump experiment+estep configuration to screen and save it to output file."""
        titles = ["Experiment", "E-step"]
        confs = [self.config, self.estep_config]
        logger.set(exp_config=self.config)
        logger.set(estep_config=self.estep_config)

        model_conf = self.model.config  # could raise
        logger.set(model_config=model_conf)
        confs.append(model_conf)
        titles.append("Model")

        for title, conf in zip(titles, confs):
            pprint(f"\n{title} configuration:")
            for k, v in conf.items():
                pprint(f"\t{k:<20}: {v}")

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
            F_and_subs_dict = {
                f"{log_kind}_F": to.tensor(F),
                f"{log_kind}_subs": to.tensor(subs),
            }
            logger.append(**F_and_subs_dict)

            # log latest states and lpj to file
            states = getattr(self, f"{data_kind}_states")
            if f"{log_kind}_states" not in self._conf.log_blacklist:
                K = gather_from_processes(states.K)
                logger.set(**{f"{log_kind}_states": K})
            else:
                K = None
            if f"{log_kind}_lpj" not in self._conf.log_blacklist:
                logger.set(**{f"{log_kind}_lpj": gather_from_processes(states.lpj)})

            if self._conf.keep_best_states:
                best_F_name = f"best_{log_kind}_F"
                best_F = getattr(self, f"_{best_F_name}", None)
                if best_F is None or F > best_F:
                    rank = dist.get_rank() if dist.is_initialized() else 0
                    if K is None:
                        K = gather_from_processes(states.K)
                    if rank == 0:
                        assert isinstance(K, to.Tensor)  # to make mypy happy
                        best_states_dict = {
                            best_F_name: to.tensor(F),
                            f"best_{log_kind}_states": K.cpu().clone(),
                        }
                        logger.set(**best_states_dict)
                    setattr(self, f"_{best_F_name}", F)

            # log data reconstructions
            reco_dict = {}
            if (
                f"{log_kind}_reconstruction" not in self._conf.log_blacklist
                and f"{data_kind}_rec" in epoch_results
            ):
                reco_dict[f"{log_kind}_reconstruction"] = gather_from_processes(
                    epoch_results[f"{data_kind}_rec"]
                )
                logger.set(**reco_dict)

        log_theta_fn = logger.set if self._conf.log_only_latest_theta else logger.append
        log_theta_fn(theta=self.model.theta)
        logger.write()


class Training(_TrainingAndOrValidation):
    def __init__(
        self,
        conf: ExpConfig,
        estep_conf: EStepConfig,
        model: Trainable,
        train_data_file: str,
        val_data_file: str = None,
    ):
        """Train model on given dataset for the given number of epochs.

        :param conf: Experiment configuration.
        :param estep_conf: Instance of a class inheriting from EStepConfig.
        :param model: model to train
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
        if tvo.get_run_policy() == "mpi":
            init_processes()
        train_dataset = get_h5_dataset_to_processes(train_data_file, ("train_data", "data"))
        val_dataset = None
        if val_data_file is not None:
            val_dataset = get_h5_dataset_to_processes(val_data_file, ("val_data", "data"))

        setattr(conf, "train_dataset", train_data_file)
        setattr(conf, "val_dataset", val_data_file)
        super().__init__(conf, estep_conf, model, train_dataset, val_dataset)


class Testing(_TrainingAndOrValidation):
    def __init__(self, conf: ExpConfig, estep_conf: EStepConfig, model: Trainable, data_file: str):
        """Test given model on given dataset for the given number of epochs.

        :param conf: Experiment configuration.
        :param estep_conf: Instance of a class inheriting from EStepConfig.
        :param model: model to test
        :param data_file: Path to an HDF5 file containing the training dataset. Datasets with name
                          "test_data" and "data" will be searched in the file, in this order.

        Only E-steps are run. Model parameters are not updated.
        """
        if tvo.get_run_policy() == "mpi":
            init_processes()
        dataset = get_h5_dataset_to_processes(data_file, ("test_data", "data"))

        setattr(conf, "test_dataset", data_file)
        super().__init__(conf, estep_conf, model, None, dataset)
