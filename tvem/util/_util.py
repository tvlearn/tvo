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
