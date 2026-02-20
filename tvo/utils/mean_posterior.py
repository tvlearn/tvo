import torch as to
from tvo import get_device
def lpj2pjc(lpj: to.Tensor):
    """Shift log-pseudo-joint and convert log- to actual probability

    :param lpj: log-pseudo-joint tensor
    :returns: probability tensor
    """
    up_lpg_bound = 0.0
    shft = up_lpg_bound - lpj.max(dim=1, keepdim=True)[0]
    tmp = to.exp(lpj + shft)
    return tmp.div_(tmp.sum(dim=1, keepdim=True))


def _mean_post_einsum(g: to.Tensor, lpj: to.Tensor) -> to.Tensor:
    """Compute expectation value of g(s) w.r.t truncated variational distribution q(s).

    :param g: Values of g(s) with shape (N,S,...).
    :param lpj: Log-pseudo-joint with shape (N,S).
    :returns: tensor with shape (N,...).
    """
    return to.einsum("ns...,ns->n...", (g, lpj2pjc(lpj)))


def _mean_post_mul(g: to.Tensor, lpj: to.Tensor) -> to.Tensor:
    """Compute expectation value of g(s) w.r.t truncated variational distribution q(s).

    :param g: Values of g(s) with shape (N,S,...).
    :param lpj: Log-pseudo-joint with shape (N,S).
    :returns: tensor with shape (N,...).
    """
    # reshape lpj from (N,S) to (N,S,1,...), to match dimensionality of g
    lpj = lpj.view(*lpj.shape, *(1 for _ in range(g.ndimension() - 2)))
    return lpj2pjc(lpj).mul(g).sum(dim=1)


def mean_posterior(g: to.Tensor, lpj: to.Tensor) -> to.Tensor:
    """Compute expectation value of g(s) w.r.t truncated variational distribution q(s).

    :param g: Values of g(s) with shape (N,S,...).
    :param lpj: Log-pseudo-joint with shape (N,S).
    :returns: tensor with shape (N,...).
    """
    if get_device().type == "cpu":
        means = _mean_post_einsum(g, lpj)
    else:
        means = _mean_post_mul(g, lpj)

    assert means.shape == (g.shape[0], *g.shape[2:])
    assert not to.isnan(means).any() and not to.isinf(means).any()
    return means
