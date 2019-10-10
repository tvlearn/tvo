import torch as to
import tvem


def generate_bars(
    H: int,
    bar_amp: float = 1.0,
    neg_amp: bool = False,
    bg_amp: float = 0.0,
    add_unit: float = None,
    precision: to.dtype = to.float64,
):
    """ Generate a ground-truth dictionary W suitable for a std. bars test

    Creates H bases vectors with horizontal and vertival bars on a R*R pixel grid,
    (wth R = H // 2).

    :param H: Number of latent variables
    :param bar_amp: Amplitude of each bar
    :param neg_amp: Set probability of amplitudes taking negative values to 50 percent
    :param bg_amp: Background amplitude
    :param add_unit: If not None an additional unit with amplitude add_unit will be inserted
    :param precision: torch.dtype of the returned tensor
    :returns: tensor containing the bars dictionary
    """
    R = H // 2
    D = R ** 2

    W = bg_amp * to.ones((R, R, H), dtype=precision, device=tvem.get_device())
    for i in range(R):
        W[i, :, i] = bar_amp
        W[:, i, R + i] = bar_amp

    if neg_amp:
        sign = 1 - 2 * to.randint(high=2, size=(H), device=tvem.get_device())
        W = sign[None, None, :] * W

    if add_unit is not None:
        add_unit = add_unit * to.ones((D, 1), device=tvem.get_device())
        W = to.cat((W, add_unit), dim=1)
        H += 1

    return W.view((D, H))
