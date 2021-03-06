# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch import Tensor

from abc import ABC, abstractmethod
from typing import Dict, Any

from tvo.variational._utils import generate_unique_states
from tvo.utils import get
import tvo
from tvo.utils.model_protocols import Trainable
from tvo.utils.parallel import get_h5_dataset_to_processes


class TVOVariationalStates(ABC):
    def __init__(self, conf: Dict[str, Any], K_init: Tensor = None):
        """Abstract base class for TVO realizations.

        :param conf: dictionary with hyper-parameters. Required keys: N, H, S, dtype, device
        :param K_init: if specified and if `conf` does specify `K_init_file`, self.K will be
                       initialized with this Tensor of shape (N,S,H)
        """
        required_keys = ("N", "H", "S", "S_new", "precision")
        for c in required_keys:
            assert c in conf and conf[c] is not None
        self.config = conf

        N, H, S, _, precision = get(conf, *required_keys)

        _K_init = (
            get_h5_dataset_to_processes(conf["K_init_file"], ("initial_states", "states"))
            if "K_init_file" in conf and conf["K_init_file"] is not None
            else K_init
        )

        if _K_init is not None:
            assert _K_init.shape == (N, S, H)
            self.K = _K_init.clone().to(dtype=to.uint8)
        else:
            self.K = generate_unique_states(S, H).repeat(N, 1, 1)  # (N, S, H)
        self.lpj = to.empty((N, S), dtype=precision, device=tvo.get_device())
        self.precision = precision

    @abstractmethod
    def update(self, idx: Tensor, batch: Tensor, model: Trainable) -> int:
        """Generate new variational states, update K and lpj with best samples and their lpj.

        :param idx: data point indices of batch w.r.t. K
        :param batch: batch of data points
        :param model: the model being used
        :returns: average number of variational state substitutions per datapoint performed
        """
        pass  # pragma: no cover
