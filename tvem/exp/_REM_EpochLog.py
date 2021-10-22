# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.utils.parallel import pprint
from tvem.utils import get


class REM_EpochLog:
    """Experiment epoch log."""

    def __init__(self, epoch, results, runtime=None):
        self.epoch = epoch
        self.runtime = runtime

        self._results = results

    def print(self):
        """Print epoch log.

        In MPI runs, this method is no-op for all processes but the one with rank 0.
        """
        if self.epoch == 0:
            pprint("Start")
        else:
            pprint(f"Epoch {self.epoch}")
        for data_kind in "train", "test":
            if (data_kind + "_F" not in self._results) and (data_kind + "_F_beta" not in self._results):
                pprint("free energy AND annealed free energy not supplied")
                continue
            # log_kind is one of "train", "valid" or "test"
            # (while data_kind is one of "train" or "test")
            log_kind = "valid" if data_kind == "test" and "train_F" in self._results else data_kind
            F_beta, F, subs = get(self._results, f"{data_kind}_F_beta", f"{data_kind}_F", f"{data_kind}_subs")
            pprint(f"\t{log_kind} F_beta/N: {F_beta:<10.5f} F/N: {F:<10.5f} avg subs: {subs:<6.2f}")
        if self.runtime is not None:
            pprint(f"\ttotal epoch runtime: {self.runtime:<5.2f} s")
