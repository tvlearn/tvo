# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import torch as to

from typing import Iterable, Sequence, Dict, Any


class ExpConfig:
    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        warmup_Esteps: int = 0,
        output: str = None,
        log_blacklist: Iterable[str] = [],
        rollback_if_F_decreases: Sequence[str] = [],
        warmup_reco_epochs: Iterable[int] = None,
        reco_epochs: Iterable[int] = None,
    ):
        """Configuration object for Experiment classes.

        :param batch_size: Batch size for the data loaders.
        :param shuffle: Whether data should be reshuffled at every epoch.
                        See also torch's `DataLoader docs`_.
        :param drop_last: set to True to drop the last incomplete batch, if the dataset size is not
                          divisible by the batch size. See also torch's `DataLoader docs`_.
        :param warmup_Esteps: Number of warm-up E-steps to perform.
        :param output: Name or path of output HDF5 file. The default filename is "tvem_exp_<PID>.h5"
                       where PID is the process ID. It is overwritten if it already exists.
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
        :param rollback_if_F_decreases: names of model parameters (corresponding to those in
                                        TVEMModel.theta) that should be rolled back (i.e. not
                                        updated) if the free energy value before and after
                                        `model.update_param_epoch` decreases for a given epoch.
                                        This is only useful for models that perform the actual
                                        update of those parameters in `update_param_epoch` and not
                                        in `update_param_batch`.
                                        BSC and NoisyOR are such models. This feature is useful,
                                        for example, to prevent NoisyOR's M-step equation from
                                        oscillating away from the fixed point (i.e. the optimum).
        :param warmup_reco_epochs: List of warmup_Estep indices at which to compute data
                                   reconstructions.
        :param reco_epochs: List of epoch indices at which to compute data reconstructions.
        """
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.warmup_Esteps = warmup_Esteps
        self.output = output if output is not None else f"tvem_exp_{os.getpid()}.h5"
        self.log_blacklist = log_blacklist
        self.rollback_if_F_decreases = rollback_if_F_decreases
        self.warmup_reco_epochs = warmup_reco_epochs
        self.reco_epochs = reco_epochs

    def as_dict(self) -> Dict[str, Any]:
        return vars(self)
