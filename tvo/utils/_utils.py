# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from typing import Dict, Any


def get(d: Dict[Any, Any], *keys: Any):
    """Shorthand to retrieve valus at specified keys from dictionary.

    :param d: input dictionary
    :param keys: a list of keys for dictionary d

    Example usage::

        val1, val2 = get(my_dict, 'key1', 'key2')
    """

    return map(d.get, keys)


def get_lstsq(torch):
    '''
    Versioned least squares function depending on Pytorch version.
    Input: torch
    '''
    torch_major_version, torch_minor_version = to.__version__.split(".")[:2]
    if torch_major_version >= 2:
        def lstsq(a, b):
            return torch.linalg.lstsq(b, a)

    elif torch_minor_version >= 10:
        # pytorch 1.10 deprecates to.lstsq in favour of to.linalg.lstsq,
        # which takes arguments in reversed order
        def lstsq(a, b):
            return torch.linalg.lstsq(b, a)

    elif torch_minor_version >= 2:
        # pytorch 1.2 deprecates to.gels in favour of to.lstsq
        lstsq = torch.lstsq

    else:
        raise ValueError("Pytorch versions below 1.2 are unsupported")

    return lstsq