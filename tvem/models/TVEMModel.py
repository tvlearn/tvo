# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from abc import ABC, abstractmethod
from torch import Tensor
from tvem.variational import TVEMVariationalStates  # type: ignore
from typing import Dict, Optional


class TVEMModel(ABC):
    """Abstract base class for probabilistic generative models to be trained with TVEM."""

    @abstractmethod
    def log_pseudo_joint(self, data: Tensor, states: Tensor) -> Tensor:
        """Evaluate log-pseudo-joint probabilities for this model.

        :param data: shape is (N,D)
        :param states: shape is (N,S,H)
        :returns: log-pseudo-joints for data and states - shape is (N,S)

        Log-pseudo-joint probabilities are the log-joint probabilities of the model
        for the specified set of datapoints and variational states where, potentially,
        some factors that do not depend on the variational states have been elided.
        """
        pass

    def init_epoch(self):
        """This method is called once at the beginning of each training epoch.

        Concrete models can optionally override this method if it's convenient.
        By default, it does nothing.
        """
        pass

    @abstractmethod
    def update_param_batch(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates,
                           mstep_factors: Dict[str, Tensor] = None) -> Optional[float]:
        """Execute batch-wise M-step or batch-wise section of an M-step computation.

        :param idx: indeces of the datapoints that compose the batch within the dataset
        :param batch: batch of datapoints, Tensor with shape (N,D)
        :param states: all variational states for this dataset
        :param mstep_factors: optional dictionary containing the Tensors that were evaluated\
            by the lpj_fn function returned by get_lpj_func during this batch's E-step.

        If the model allows it, as an optimization this method can return this batch's free energy
        evaluated _before_ the model parameter update. If the batch's free energy is returned here,
        Trainers will skip a direct per-batch call to the free_energy method.
        """
        pass

    def update_param_epoch(self) -> None:
        """Execute epoch-wise M-step or epoch-wise section of an M-step computation.

        This method is called at the end of each training epoch.
        Implementing this method is optional: models can leave the body empty (just a `pass`)
        or even not implement it at all.
        """
        pass

    @abstractmethod
    def free_energy(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> float:
        """Evaluate free energy for the given batch of datapoints.

        :param idx: indeces of the datapoints in batch within the full dataset
        :param batch: batch of datapoints, Tensor with shape (N,D)
        :param states: all TVEMVariationalStates states for this dataset
        """
        pass

    @abstractmethod
    def generate_data(self, N: int) -> Tensor:
        """Generate N random datapoints from this model."""
        pass

    @abstractmethod
    def generate_from_hidden(self, hidden_state: Tensor) -> Tensor:
        """Generate N random datapoints from this model.

        :param hidden_state: Tensor with shape (N,H) where H is the number of units in the
            first latent layer.

        The returned Tensor has shape (N,D) where D is the number of observables for this model.
        """
        pass

    def get_mstep_factors(self) -> Dict[str, Tensor]:
        """Get quantities required to perform an M-step."""
        pass
