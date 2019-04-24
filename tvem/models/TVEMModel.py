# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from abc import ABC, abstractmethod
from torch import Tensor
import torch.distributed as dist
from tvem.variational import TVEMVariationalStates  # type: ignore
from typing import Dict, Optional, Tuple
import tvem
from tvem.util.parallel import broadcast


class TVEMModel(ABC):
    def __init__(self, theta: Dict[str, Tensor]):
        """Abstract base class for probabilistic generative models to be trained with TVEM."""
        # assert that all parameters have compatible precision
        last_dtype = None
        for p in theta.values():
            this_dtype = p.dtype
            if this_dtype is to.uint8:
                continue
            if last_dtype is not None:
                assert last_dtype is this_dtype,\
                    'All floating point model parameters must have the same precision'
            last_dtype = this_dtype

        self.theta = theta
        if last_dtype is not None:
            self.precision = last_dtype

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
        pass  # pragma: no cover

    def init_epoch(self) -> None:
        """This method is called once at the beginning of each training epoch.

        Concrete models can optionally override this method if it's convenient.
        By default, it does nothing.
        """
        pass  # pragma: no cover

    @abstractmethod
    def update_param_batch(self, idx: Tensor, batch: Tensor,
                           states: TVEMVariationalStates) -> Optional[float]:
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
        pass  # pragma: no cover

    def update_param_epoch(self) -> None:
        """Execute epoch-wise M-step or epoch-wise section of an M-step computation.

        This method is called at the end of each training epoch.
        Implementing this method is optional: models can leave the body empty (just a `pass`)
        or even not implement it at all.
        """
        pass  # pragma: no cover

    @abstractmethod
    def free_energy(self, idx: Tensor, batch: Tensor, states: TVEMVariationalStates) -> float:
        """Evaluate free energy for the given batch of datapoints.

        :param idx: indeces of the datapoints in batch within the full dataset
        :param batch: batch of datapoints, Tensor with shape (N,D)
        :param states: all TVEMVariationalStates states for this dataset
        """
        pass  # pragma: no cover

    def generate_data(self, N: int) -> Dict[str, Tensor]:
        """Sample N datapoints from this model.

        :param N: number of data points to be generated.
        :returns: dictionary with keys
                  - `data`: a tensor with shape (N, D) where D is the number of
                    observables for this model.
                  - `hidden_state`: a tensor with shape (N, H) where H is the number of
                    hidden variables for this model
        """
        theta = self.theta

        pies = theta['pies']

        S = to.rand((N, pies.numel()), dtype=pies.dtype, device=pies.device) <= pies

        return {'data': self.generate_from_hidden(S), 'hidden_state': S}

    @abstractmethod
    def generate_from_hidden(self, hidden_state: Tensor) -> Dict[str, Tensor]:
        """Generate N random datapoints from this model.

        :param hidden_state: Tensor with shape (N,H) where H is the number of units in the
            first latent layer.

        Data points are stored in the returned Dictionary in key 'Y' and have shape (N,D) where
        D is the number of observables for this model.
        """
        pass  # pragma: no cover

    @property
    def sorted_by_lpj(self) -> Dict[str, Tensor]:
        """A dictionary of Tensors that are to be kept ordered in sync with log-pseudo-joints.

        The Trainer will take care that the tensors in this dictionary are sorted the same way
        log-pseudo-joints are during an E-step.
        Tensors must have shapes (batch_size, S, ...) where S is the number of variational
        states per datapoint used during training.
        By default the dictionary is empy. Concrete models can override this property if need be.
        """
        return {}

    def init_batch(self):
        """Model-specific initializations per batch."""
        pass  # pragma: no cover

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, ...]:
        """The model shape.

        :returns: the model shape, observable layer followed by the hidden layers: (D, H1, H2, ...)
        """
        pass  # pragma: no cover


def check_theta(theta_new: Dict[str, Tensor], policy: Dict[str, Tensor]):
    """Perform sanity check of values in theta dict according to policy.

    :param theta_new: Dictionary containing new model parameters
    :param policy: Policy dictionary bounds and replacements for invalid values
    """
    rank = dist.get_rank() if dist.is_initialized() else 0

    for key, val in policy.items():
        new_value = theta_new[key]
        if rank == 0:
            replacement, low_bound, up_bound = val

            if to.isnan(new_value).any():
                new_value[to.isnan(new_value)] = replacement[to.isnan(new_value)]
                print("Sanity check: Replaced nan entries of %s." % key)

            if to.isinf(new_value).any():
                new_value[to.isinf(new_value)] = replacement[to.isinf(new_value)]
                print("Sanity check: Replaced infinite entries of %s." % key)

            if (new_value < low_bound).any():
                to.max(input=low_bound, other=new_value, out=new_value)
                print("Sanity check: Reset lower bound of %s" % key)

            if (new_value >= up_bound).any():
                to.min(input=up_bound, other=new_value, out=new_value)
                print("Sanity check: Reset upper bound of %s" % key)

        broadcast(new_value)
        theta_new[key] = new_value


def init_W_data_mean(data_mean: Tensor, data_var: Tensor, H: int, dtype: to.dtype = to.float64,
                     device: to.device = None) -> Tensor:
    """Initialize weights W based on noisy mean of the data points.

    param data_mean: Mean of all data points. Length equals data dimensionality D.
    param data_var: Variance of all data points in each dimension d=1,...D.
    param H: Number of basis functions to be generated.
    param dtype: dtype of output Tensor. Defaults to torch.float64.
    param device: torch.device of output Tensor. Defaults to tvem.get_device().
    returns: Weight matrix W with shape (D,H).
    """

    if device is None:
        device = tvem.get_device()
    return data_mean.to(dtype=dtype, device=device).repeat((H, 1)).t() +\
        to.mean(to.sqrt(data_var.to(dtype=dtype, device=device))) * \
        to.randn((data_mean.size(), H), dtype=dtype, device=device)


def init_sigma_default(data_var: Tensor, dtype: to.dtype = to.float64,
                       device: to.device = None) -> Tensor:
    """Initialize scalar sigma parameter based on variance of the data points.

    param data_var: Variance of all data points in each dimension d=1,...D of the data.
    param dtype: dtype of output Tensor. Defaults to torch.float64.
    param device: torch.device of output Tensor. Defaults to tvem.get_device().
    returns: Scalar sigma parameter.

    Returns the mean of the variance in each dimension d=1,...,D.
    """

    if device is None:
        device = tvem.get_device()
    return to.mean(to.sqrt(data_var.to(dtype=dtype, device=device)))


def init_pies_default(H: int, crowdedness: float = 2., dtype: to.dtype = to.float64,
                      device: to.device = None):
    """Initialize pi parameter based on given crowdedness.

    param H: Length of pi vector.
    param crowdedness: Average crowdedness corresponding to sum of elements in vector pi.
    param dtype: dtype of output Tensor. Defaults to torch.float64.
    param device: torch.device of output Tensor. Defaults to tvem.get_device().
    returns: Vector pi.
    """

    if device is None:
        device = tvem.get_device()
    return to.full((H,), fill_value=crowdedness/H, dtype=dtype, device=device)
