import tvem
import torch as to
from torch import Tensor
from scipy.stats import special_ortho_group


def init_W_data_mean(
    data_mean: Tensor,
    data_var: Tensor,
    H: int,
    std_factor: float = 0.01,
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

    if device is None:
        device = tvem.get_device()
    return data_mean.to(dtype=dtype, device=device).repeat((H, 1)).t() + std_factor * to.mean(
        to.sqrt(data_var.to(dtype=dtype, device=device))
    ) * to.randn((len(data_mean), H), dtype=dtype, device=device)


def init_sigma_default(
    data_var: Tensor, dtype: to.dtype = to.float64, device: to.device = None
) -> Tensor:
    """Initialize scalar sigma parameter based on variance of the data points.

    :param data_var: Variance of all data points in each dimension d=1,...D of the data.
    :param dtype: dtype of output Tensor. Defaults to torch.float64.
    :param device: torch.device of output Tensor. Defaults to tvem.get_device().
    :returns: Scalar sigma parameter.

    Returns the mean of the variance in each dimension d=1,...,D.
    """

    if device is None:
        device = tvem.get_device()
    return to.mean(to.sqrt(data_var.to(dtype=dtype, device=device)), dim=0, keepdim=True)

def init_Sigma_default(
    data_cov: Tensor, dtype: to.dtype = to.float64, device: to.device = None
) -> Tensor:
    """Initialize Sigma parameter based on variance of the data points.

    :param data_cov: Variance of all data points in each dimension d=1,...D of the data.
    :param dtype: dtype of output Tensor. Defaults to torch.float64.
    :param device: torch.device of output Tensor. Defaults to tvem.get_device().
    :returns: Sigma parameter.

    Returns the covariance of the whole data with some added noise maintaining a covariance matrix
    """

    if device is None:
        device = tvem.get_device()
    
    # creation of a semmetric positive semidefinite matrix
    a = [to.diag(to.rand(D, dtype=precision)) for k in range(H)]

    b = to.zeros(H, D, D, dtype=precision)
    for h, cluster in enumerate(a):
        V = to.tensor(special_ortho_group.rvs(D).T, dtype=precision)
        b[h,:,:] = V @ cluster @ V.T
    # b = b.permute(1,2,0) # make D, D, H
    res = data_cov + b 
    
    return res.permute(1,2,0)

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
