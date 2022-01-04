import tvem
import numpy as np
import torch as to
from torch import Tensor


def init_W_data_mean(
    data: Tensor,
    H: int,
    std_factor: float = 0.25,
    dtype: to.dtype = to.float64,
    device: to.device = None,
) -> Tensor:
    """Initialize weights W based on noisy mean of the data points.

    :param data_mean: Mean of all data points. Length equals data dimensionality D.
    :param data_var: Variance of all data points in each dimension d=1,...D.
    :param H: Number of basis functions to be generated.
    :param std_factor: Scalar to control amount of standard deviation of additive noise
    :param dtype: dtype of output Tensor. Defaults to torch.float64.
    :param device: torch.device of output Tensor. Defaults to tvem.get_device().
    :returns: Weight matrix W with shape (D,H).
    """
    device_ = tvem.get_device() if device is None else device
    data_nanmean = to.from_numpy(np.nanmean(data.detach().cpu().numpy(), axis=0)).to(
        dtype=dtype, device=device_
    )
    var = init_sigma_default(data, dtype, device_)
    return data_nanmean.repeat((H, 1)).t() + std_factor * to.sqrt(var) * to.randn(
        (len(data_nanmean), H), dtype=dtype, device=device_
    )


def init_sigma_default(
    data: Tensor, dtype: to.dtype = to.float64, device: to.device = None
) -> Tensor:
    """Initialize scalar sigma parameter based on variance of the data points.

    :param data_var: Variance of all data points in each dimension d=1,...D of the data.
    :param dtype: dtype of output Tensor. Defaults to torch.float64.
    :param device: torch.device of output Tensor. Defaults to tvem.get_device().
    :returns: Scalar sigma parameter.

    Returns the mean of the variance in each dimension d=1,...,D.
    """
    _device = tvem.get_device() if device is None else device
    cov = to.from_numpy(np.cov(data.detach().cpu().numpy().T)).to(device=_device, dtype=dtype)
    return to.mean(to.diag(cov)) + to.tensor([0.001], device=_device, dtype=dtype)


def init_pies_default(
    H: int, crowdedness: float = 2.0, dtype: to.dtype = to.float64, device: to.device = None
):
    """Initialize pi parameter based on given crowdedness.

    :param H: Length of pi vector.
    :param crowdedness: Average crowdedness corresponding to sum of elements in vector pi.
    :param dtype: dtype of output Tensor. Defaults to torch.float64.
    :param device: torch.device of output Tensor. Defaults to tvem.get_device().
    :returns: Vector pi.
    """

    if device is None:
        device = tvem.get_device()
    return to.full((H,), fill_value=crowdedness / H, dtype=dtype, device=device)
