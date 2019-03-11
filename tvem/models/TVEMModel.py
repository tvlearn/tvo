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
    def get_lpj_func(self):
        """Return a function that evaluates log-joints (or log-pseudo-joints) for this model.

        The function returned should have signature
            lpj_fn(data, states) -> lpj, mstep_factors
        where:
            - data -- Tensor with shape (N,D)
            - states -- Tensor with shape (N,S,H)
            - lpj -- Tensor with shape (N,S)
            - mstep_factors -- optional dictionary containing other Tensors with shapes (N,S,...).
                This dictionary is passed to update_param_batch after lpj_fn has been called on
                a given batch. The dictionary can be empty and mstep_factors can just be None.
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

        idx -- indeces of the datapoints that compose the batch within the dataset
        batch -- batch of datapoints, Tensor with shape (N,D)
        states -- all variational states for this dataset
        mstep_factors -- optional dictionary containing the Tensors that were evaluated
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

        idx -- indeces of the datapoints in batch within the full dataset
        batch -- batch of datapoints, Tensor with shape (N,D)
        states -- all TVEMVariationalStates states for this dataset
        """
        pass

    @abstractmethod
    def generate_data(self, N: int) -> Tensor:
        """Generate N random datapoints from this model."""
        pass

    @abstractmethod
    def generate_from_hidden(self, hidden_state: Tensor) -> Tensor:
        """Generate N random datapoints from this model.

        hidden_state -- Tensor with shape (N,H) where H is the number of units in the
            first latent layer.

        The returned Tensor has shape (N,D) where D is the number of observables for this model.
        """
        pass
