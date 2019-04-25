# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
import h5py
from typing import Dict, Iterable
from os import path, rename


class TVEMDataLoader(DataLoader):
    def __init__(self, data: to.Tensor, *args, **kwargs):
        """TVEM DataLoader class. Derived from torch.utils.data.DataLoader.

        TVEMDataLoader is constructed exactly the same way as pytorch's DataLoader,
        but it restricts possible input datasets to a single to.Tensor.
        All other arguments are forwarded to pytorch's DataLoader.
        When iterated over, TVEMDataLoader yields a tuple containing the indeces of
        the datapoints in each batch as well as the actual datapoints for each
        tensor in the input Tensor.

        :param data: Tensor containing the input dataset. Must have exactly two dimensions (N,D).
        :param args: forwarded to pytorch's DataLoader.
        :param kwargs: forwarded to pytorch's DataLoader.
        """
        N = data.shape[0]
        if data.dtype is not to.uint8:
            self.precision = data.dtype
        super().__init__(TensorDataset(to.arange(N), data), *args, **kwargs)


class H5Logger:
    def __init__(self, output: str, blacklist: Iterable[str] = []):
        """Utility class to iteratively write to HD5 files.

        :param output: Output filename or file path. Overwritten if it already exists.
        :param blacklist: Variables in `blacklist` are ignored and never get logged.

        If tvem.get_run_policy() is 'mpi', operations on H5Logger are no-op for all processes
        except for the process with rank 0.
        """
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._fname = output
        self._data: Dict[str, to.Tensor] = {}
        self._blacklist = blacklist

    def append(self, **kwargs):
        """Append input arguments to log. All arguments must be torch.Tensors."""
        if self._rank != 0:
            return

        data = self._data

        for k, v in kwargs.items():
            if k in self._blacklist:
                continue

            assert isinstance(v, to.Tensor), "all arguments must be torch.Tensors"

            if k not in data:
                data[k] = v.unsqueeze(0)  # extra dim will be used for concatenation
            else:
                assert data[k].shape[1:] == v.shape, f"variable {k} changed shape between appends"
                data[k] = to.cat((data[k], v.unsqueeze(0)))

    def set(self, **kwargs):
        """Set keyword arguments to desired value in output file.

        All arguments must be torch.Tensors.
        """
        if self._rank != 0:
            return

        for k, v in kwargs.items():
            if k in self._blacklist:
                continue

            assert isinstance(v, to.Tensor), "all arguments must be torch.Tensors"
            self._data[k] = v

    def write(self):
        """Write logged data to output file.

        If a file with this name already exists (e.g. because of a previous call to this method)
        the old file is renamed to `<fname>.old`.
        """
        if self._rank != 0:
            return

        fname = self._fname

        if path.exists(fname):
            rename(fname, fname + ".old")

        f = h5py.File(fname, "w")
        for k, v in self._data.items():
            f.create_dataset(k, data=v.cpu())
        f.close()
