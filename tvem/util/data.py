# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch.utils.data import TensorDataset, DataLoader


class TVEMDataLoader(DataLoader):
    """General TVEM DataLoader class. Derived from torch.utils.data.DataLoader"""
    def __init__(self, data: TensorDataset, *args, **kwargs):
        N = data.tensors[0].shape[0]
        super().__init__(TensorDataset(to.arange(N), *data.tensors), *args, **kwargs)
