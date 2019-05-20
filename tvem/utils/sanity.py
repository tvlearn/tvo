# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to

from torch import Tensor
from typing import Union


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
