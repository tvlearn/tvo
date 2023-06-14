# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from torch import Tensor
from typing import Union
import torch.distributed as dist
from tvo.utils.parallel import broadcast
from typing import Dict, List


def fix_theta(theta: Dict[str, Tensor], policy: Dict[str, List]):
    """Perform sanity check of values in theta dict according to policy.

    :param theta: Dictionary containing model parameters
    :param policy: Policy dictionary. Must have same keys as theta.

    Each key in policy contains a list with three floats/tensors referred to as replacement,
    low_bound and up_bound. If tensors are provided these must have the same shape as the
    corresponding tensors in the theta dictionary. For each key
    - infinite values in tensors from the theta dictionary are replaced with values from the
    corresponding entry in replacement and
    - the values in tensors from the theta dictionary are clamped to the corresponding values
    of low_bound and up_bound.
    """
    assert set(theta.keys()) == set(policy.keys()), "theta and policy must have same keys"

    rank = dist.get_rank() if dist.is_initialized() else 0

    for key, val in policy.items():
        new_val = theta[key]
        if rank == 0:
            replacement, low_bound, up_bound = val

            fix_infinite(new_val, replacement, key)
            fix_bounds(new_val, low_bound, up_bound, key)

        broadcast(new_val)
        theta[key] = new_val


def fix_infinite(values: Tensor, replacement: Union[float, Tensor], name: str = None):
    """Fill infinite entries in values with  replacement
    :param values: Input tensor
    :param replacement: Scalar or tensor with replacements for infinite values
    :param name: Name of input tensor (optional).
    """
    mask_infinite = to.isnan(values) | to.isinf(values)
    if mask_infinite.any():
        if isinstance(replacement, float):
            values[mask_infinite] = replacement
        elif isinstance(replacement, Tensor):
            values[mask_infinite] = replacement[mask_infinite]
        if name is not None:
            print("Sanity check: Replaced infinite entries of %s." % name)


def fix_bounds(
    values: Tensor,
    lower: Union[float, Tensor] = None,
    upper: Union[float, Tensor] = None,
    name: str = None,
):
    """Clamp entries in values to not exceed lower and upper bounds.
    :param values: Input tensor
    :param lower: Scalar or tensor with lower bounds for values
    :param upper: Scalar or tensor with upper bounds for values
    :param name: Name of input tensor (optional).
    """
    if (lower is not None) and (values < lower).any():
        if isinstance(lower, float):
            to.clamp(input=values, min=lower, out=values)
        elif isinstance(lower, Tensor):
            to.max(input=lower, other=values, out=values)
        if name is not None:
            print("Sanity check: Reset lower bound of %s" % name)

    if (upper is not None) and (values >= upper).any():
        if isinstance(upper, float):
            to.clamp(input=values, max=upper, out=values)
        elif isinstance(upper, Tensor):
            to.min(input=upper, other=values, out=values)
        if name is not None:
            print("Sanity check: Reset upper bound of %s" % name)
