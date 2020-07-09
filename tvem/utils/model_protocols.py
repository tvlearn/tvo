from typing_extensions import Protocol, runtime_checkable
from typing import Tuple, Dict, Any, Optional, Union, TYPE_CHECKING
import torch as to
from abc import abstractmethod

if TYPE_CHECKING:
    from tvem.variational.TVEMVariationalStates import TVEMVariationalStates


@runtime_checkable
class Trainable(Protocol):
    """The most basic model.

    Requires implementation of log_joint, update_parameter_batch, update_parameter_epoch.
    Provides default implementation of free_energy.
    """

    @abstractmethod
    def log_joint(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        """Evaluate log-joint probabilities for this model.

        :param data: shape is (N,D)
        :param states: shape is (N,S,H)
        :returns: log-joints for data and states - shape is (N,S)
        """
        ...

    def update_param_batch(
        self, idx: to.Tensor, batch: to.Tensor, states: "TVEMVariationalStates"
    ) -> Optional[float]:
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
        ...

    def update_param_epoch(self) -> None:
        """Execute epoch-wise M-step or epoch-wise section of an M-step computation.

        This method is called at the end of each training epoch.
        Implementing this method is optional: models can leave the body empty (just a `pass`)
        or even not implement it at all.
        """
        ...

    def free_energy(
        self, idx: to.Tensor, batch: to.Tensor, states: "TVEMVariationalStates"
    ) -> float:
        """Evaluate free energy for the given batch of datapoints.

        :param idx: indeces of the datapoints in batch within the full dataset
        :param batch: batch of datapoints, Tensor with shape (N,D)
        :param states: all TVEMVariationalStates states for this dataset

        .. note::
        This default implementation of free_energy is only appropriate for Trainable models
        that are not Optimized.
        """
        log_joints = states.lpj[idx]  # these are actual log-joints in Trainable models
        return to.logsumexp(log_joints, dim=1).sum(dim=0).item()

    @property
    def shape(self) -> Tuple[int, ...]:
        """The model shape.

        :returns: the model shape, observable layer followed by the hidden layers: (D, H1, H2, ...)

        The default implementation returns self._shape.
        """
        return getattr(self, "_shape")

    @property
    def config(self) -> Dict[str, Any]:
        """Model configuration.

        The default implementation returns self._config.
        """
        return getattr(self, "_config")

    @property
    def theta(self) -> Dict[str, to.Tensor]:
        """Dictionary of model parameters.

        The default implementation returns self._theta.
        """
        return getattr(self, "_theta")

    @property
    def precision(self) -> to.dtype:
        """The floating point precision the model works at (either to.float32 or to.float64).

        The default implementation returns self._precision.
        """
        return getattr(self, "_precision")


@runtime_checkable
class Optimized(Trainable, Protocol):
    """Additionally implements log_pseudo_joint, init_storage, init_batch, init_epoch."""

    @abstractmethod
    def log_joint(self, data: to.Tensor, states: to.Tensor, lpj: to.Tensor = None) -> to.Tensor:
        """Evaluate log-joint probabilities for this model.

        :param data: shape is (N,D)
        :param states: shape is (N,S,H)
        :param lpj: shape is (N,S). When lpj is not None it must contain pre-evaluated
                    log-pseudo joints for the given data and statss. The implementation can take
                    advantage of the extra argument to save computation.
        :returns: log-joints for data and states - shape is (N,S)
        """
        ...

    @abstractmethod
    def log_pseudo_joint(self, data: to.Tensor, states: to.Tensor) -> to.Tensor:
        """Evaluate log-pseudo-joint probabilities for this model.

        :param data: shape is (N,D)
        :param states: shape is (N,S,H)
        :returns: log-pseudo-joints for data and states - shape is (N,S)

        Log-pseudo-joint probabilities are the log-joint probabilities of the model
        for the specified set of datapoints and variational states where, potentially,
        some factors that do not depend on the variational states have been elided.

        Implementation of this method is an optional performance optimization.
        """
        ...

    def free_energy(
        self, idx: to.Tensor, batch: to.Tensor, states: "TVEMVariationalStates"
    ) -> float:
        """Evaluate free energy for the given batch of datapoints.

        :param idx: indeces of the datapoints in batch within the full dataset
        :param batch: batch of datapoints, Tensor with shape (N,D)
        :param states: all TVEMVariationalStates states for this dataset

        .. note::
        This default implementation of free_energy is only appropriate for Optimized models.
        """
        log_joints = self.log_joint(batch, states.K[idx], states.lpj[idx])
        return to.logsumexp(log_joints, dim=1).sum(dim=0).item()

    def init_storage(self, S: int, Snew: int, batch_size: int) -> None:
        """This method is called once by an experiment when initializing a model

        :param n_states: Number of variational states per datapoint to keep in memory
        :param n_new_states: Number of new states per datapoint sampled in variational E-step
        :param batch_size: Batch size used by the data loader

        Concrete models can optionally override this method if it's convenient.
        By default, it does nothing.
        """
        pass

    def init_epoch(self) -> None:
        """This method is called once at the beginning of each training epoch.

        Concrete models can optionally override this method if it's convenient.
        By default, it does nothing.
        """
        pass

    def init_batch(self) -> None:
        """Model-specific initializations per batch."""
        pass

    @property
    def sorted_by_lpj(self) -> Dict[str, to.Tensor]:
        """Optional dictionary of Tensors that must be kept ordered in sync with log-pseudo-joints.

        The Trainer will take care that the tensors in this dictionary are sorted the same way
        log-pseudo-joints are during an E-step.
        Tensors must have shapes (batch_size, S, ...) where S is the number of variational
        states per datapoint used during training.
        By default the dictionary is empy. Concrete models can override this property if need be.
        """
        return {}


@runtime_checkable
class Sampler(Protocol):
    """Implements generate_data (hidden_state is an optional parameter)."""

    @abstractmethod
    def generate_data(
        self, N: int, hidden_state: to.Tensor = None
    ) -> Union[to.Tensor, Tuple[to.Tensor, to.Tensor]]:
        """Sample N datapoints from this model.

        :param N: number of data points to be generated.
        :param hidden_state: optional Tensor with shape (N,H) where H is the number of units in the
                             first latent layer.
        :returns: if hidden_state was not provided, a tuple (data, hidden_state) where data is
                  a Tensor with shape (N, D) where D is the number of observables for this model
                  and hidden_state is a the corresponding tensor of hidden variables with shape
                  (N, H) where H is the number of hidden variables for this model.
        """
        ...


@runtime_checkable
class Reconstructor(Protocol):
    """Implements data_estimator."""

    @abstractmethod
    def data_estimator(self, idx: to.Tensor, states: "TVEMVariationalStates") -> to.Tensor:
        """Estimator used for data reconstruction. Data reconstruction can only be supported
        by a model if it implements this method. The estimator to be implemented is defined
        as follows:""" r"""
        :math:`\\langle \langle y_d \rangle_{p(y_d|\vec{s},\Theta)} \rangle_{q(\vec{s}|\mathcal{K},\Theta)}`  # noqa
        """
        ...
