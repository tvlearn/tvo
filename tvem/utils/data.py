# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch.utils.data import TensorDataset, DataLoader
import torch.distributed as dist
import h5py
from typing import Union, Dict, Iterable
from os import path, rename


class TVEMDataLoader(DataLoader):
    def __init__(self, *data: to.Tensor, **kwargs):
        """TVEM DataLoader class. Derived from torch.utils.data.DataLoader.

        TVEMDataLoader is constructed exactly the same way as pytorch's DataLoader,
        but it restricts possible input datasets to a single to.Tensor.
        All other arguments are forwarded to pytorch's DataLoader.
        When iterated over, TVEMDataLoader yields a tuple containing the indeces of
        the datapoints in each batch as well as the actual datapoints for each
        tensor in the input Tensor.

        :param data: Tensor containing the input dataset. Must have exactly two dimensions (N,D).
        :param kwargs: forwarded to pytorch's DataLoader.
        """
        N = data[0].shape[0]
        for d in data:
            assert d.shape[0] == N, "Dimension mismatch in data sets."
        if data[0].dtype is not to.uint8:
            self.precision = data[0].dtype
        super().__init__(TensorDataset(to.arange(N), *data), **kwargs)


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

    def append(self, **kwargs: Union[to.Tensor, Dict[str, to.Tensor]]):
        """Append arguments to log. Arguments can be torch.Tensors or dictionaries thereof.

        The output HDF5 file will contain one dataset for each of the tensors and one group
        for each of the dictionaries.
        """
        if self._rank != 0:
            return

        def append_to_dict(d: Dict[str, to.Tensor], k: str, t: to.Tensor):
            """Append tensor t to dict d at key k."""
            if k not in d:
                # the extra 0-sized dimension will be used for concatenation
                d[k] = to.empty((0, *t.shape))
            assert d[k].shape[1:] == t.shape, f"variable {k} changed shape between appends"
            d[k] = to.cat((d[k].to(t), t.unsqueeze(0)))

        data = self._data
        for k, v in kwargs.items():
            if k in self._blacklist:
                continue

            if isinstance(v, to.Tensor):
                append_to_dict(data, k, v)
            elif isinstance(v, dict):
                if k not in data:
                    data[k] = {}
                for name, tensor in v.items():
                    append_to_dict(data[k], name, tensor)
            else:  # pragma: no cover
                raise TypeError("Arguments must be torch.Tensors or dictionaries thereof.")

    def set(self, **kwargs: Union[to.Tensor, Dict[str, to.Tensor]]):
        """Set or reset arguments to desired value in log.

        Arguments can be torch.Tensors or dictionaries thereof.
        The output HDF5 file will contain one dataset for each of the tensors and one group
        for each of the dictionaries.
        """
        if self._rank != 0:
            return

        for k, v in kwargs.items():
            if k in self._blacklist:
                continue

            if not isinstance(v, to.Tensor) and not isinstance(v, dict):  # pragma: no cover
                raise TypeError("Arguments must be torch.Tensors or dictionaries thereof.")

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
            if isinstance(v, to.Tensor):
                f.create_dataset(k, data=v.cpu())
            else:  # dictionary
                g = f.create_group(k)
                for name, tensor in v.items():
                    g.create_dataset(name, data=tensor.cpu())
        f.close()
