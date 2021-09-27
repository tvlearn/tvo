
# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os

from typing import Iterable, Sequence, Dict, Any, Callable
from tvem.exp._ExpConfig import ExpConfig
import torch as to
class REMExpConfig(ExpConfig):
    def __init__(
        self,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = False,
        warmup_Esteps: int = 0,
        output: str = None,
        log_blacklist: Iterable[str] = [],
        log_only_latest_theta: bool = False,
        rollback_if_F_decreases: Sequence[str] = [],
        warmup_reco_epochs: Iterable[int] = None,
        reco_epochs: Iterable[int] = None,
        keep_best_states: bool = False,
        eval_F_at_epoch_end: bool = False,
        data_transform: Callable[[to.Tensor], to.Tensor] = None,
        beta: to.tensor = to.tensor([1.0]),
        beta_steps: to.Tensor = to.tensor([100]),
        beta_warmup: float = 1.0
    ):
        assert to.all(beta <= 1) and to.all(beta >= 0), "temperature has to be between 0 and 1"
        assert to.numel(beta) == to.numel(beta_steps), "beta and beta_steps must have the same number of elements"
        if not (warmup_reco_epochs==0):
            assert beta_warmup <= 1 and beta_warmup >= 0, 'beta_warmup has to be between  0 and 1'


        self.beta = beta
        self.beta_steps = beta_steps
        self.beta_warmup = beta_warmup
        super().__init__(
            batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, warmup_Esteps=warmup_Esteps,
            output=output, log_blacklist=log_blacklist, log_only_latest_theta=log_only_latest_theta, 
            rollback_if_F_decreases=rollback_if_F_decreases, warmup_reco_epochs=warmup_reco_epochs,
            keep_best_states=keep_best_states, eval_F_at_epoch_end=eval_F_at_epoch_end, 
            data_transform=data_transform
            )

