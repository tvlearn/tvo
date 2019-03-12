# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from typing import Dict


def get(d: Dict, *keys):
    return map(d.get, keys)
