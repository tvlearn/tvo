# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import h5py
import torch as to
from torch import Tensor
from typing import Tuple, Optional, Dict

from tvo.utils.model_protocols import Sampler, Trainable
from tvo.variational import FullEM


def get_bars_gfs(
    no_bars: int,
    bar_amp: float,
    neg_amp: bool = False,
    bg_amp: float = 0.0,
    precision: to.dtype = to.float32,
) -> Tensor:
    """
    Create horizontal and vertical bars.
    :param no_bars: Number of bars (must be multiple of two)
    :param bar_amp: Bar amplitude(s)
    :param neg_amp: Randomly set amplitudes to positive and negative values
    :param bg_amp: Background amplitude
    :param precision: torch.dtype of output tensor
    :return: Dictionary of bar gfs stacked as column vectors, is (no_pixels, no_bars)
    """
    assert no_bars % 2 == 0, "no_gen_fields must be a multiple of two"
    R = no_bars // 2
    D = R ** 2
    bg_amp = 0.0

    W = bg_amp * to.ones((R, R, no_bars), dtype=precision)
    for i in range(R):
        W[i, :, i] = bar_amp
        W[:, i, R + i] = bar_amp

    neg_amp = False  # Whether to set probability of amplitudes taking negative values to 50 percent
    if neg_amp:
        sign = 1 - 2 * to.randint(high=2, size=(no_bars,))
        W = sign[None, None, :] * W

    return W.view((D, no_bars))


def _compute_log_likelihood(model: Trainable, data: Tensor) -> float:
    no_data_points = data.shape[0]
    return (
        to.logsumexp(
            model.log_joint(
                data=data,
                states=FullEM(
                    N=no_data_points, H=model.shape[1], precision=model.config["precision"]
                ).K,
            ),
            dim=1,
        )
        .sum(dim=0)
        .item()
        / no_data_points
    )


def _store_as_h5(to_store_dict: Dict[str, to.Tensor], output_name: str):
    os.makedirs(os.path.split(output_name)[0], exist_ok=True)
    with h5py.File(output_name, "w") as f:
        for key, val in to_store_dict.items():
            f.create_dataset(key, data=val if isinstance(val, float) else val.detach().cpu())
    print(f"Wrote {output_name}")


def generate_data_and_write_to_h5(
    model: Sampler,
    h5_file: str,
    no_data_points: int,
    compute_ll: bool,
) -> Tuple[Tensor, Dict[str, Tensor], Optional[float]]:
    """
    Generate dataset using bar-like generative fields (GFs). An H5 file will be created
    containing the data, the generative parameters and their respective log-likelihood.
    :param model: The generative model to draw from. Must be an instance of Sampler.
    :param h5_file: Name of the created H5 file (if directory does not exist, it will be created)
    :param no_data_points: Number of data points to be sampled
    :param compute_ll: Whether to compute the log-likelihood of the generative parameters given
                       the data
    :return: Generated dataset, is (no_data_points, no_pixels)
    """
    assert isinstance(model, Sampler)

    data = model.generate_data(no_data_points)[0]

    ll_gen = None
    if compute_ll:
        assert isinstance(model, Trainable)  # to make mypy happy
        ll_gen = _compute_log_likelihood(model, data)

    to_store_dict = {
        **{f"{k}_gen": model.theta[k] for k in model.theta},  # type: ignore
        "data": data,
        "LL_gen": ll_gen,
    }

    _store_as_h5(to_store_dict, h5_file)

    return data, model.theta, ll_gen  # type: ignore
