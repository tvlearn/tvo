# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from typing import Iterable


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
