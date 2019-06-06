# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import torch.distributed as dist
import h5py
from typing import Union, Dict, Iterable, Any
from os import path, rename
import numpy as np
import tvem
from tvem.utils.parallel import broadcast


class TVEMDataLoader(DataLoader):
    def __init__(self, *data: to.Tensor, **kwargs):
        """TVEM DataLoader class. Derived from torch.utils.data.DataLoader.

        :param data: Tensor containing the input dataset. Must have exactly two dimensions (N,D).
        :param kwargs: forwarded to pytorch's DataLoader.

        TVEMDataLoader is constructed exactly the same way as pytorch's DataLoader,
        but it restricts datasets to TensorDataset constructed from the *data passed
        as parameter. All other arguments are forwarded to pytorch's DataLoader.

        When iterated over, TVEMDataLoader yields a tuple containing the indeces of
        the datapoints in each batch as well as the actual datapoints for each
        tensor in the input Tensor.

        TVEMDataLoader instances optionally expose the attribute `precision`, which is set to the
        dtype of the first dataset in *data if it is a floating point dtype.
        """
        N = data[0].shape[0]
        assert all(d.shape[0] == N for d in data), "Dimension mismatch in data sets."

        if data[0].dtype is not to.uint8:
            self.precision = data[0].dtype

        dataset = TensorDataset(to.arange(N), *data)

        if tvem.get_run_policy() == "mpi" and "sampler" not in kwargs:
            # Number of _desired_ datapoints per worker: the last worker might have less actual
            # datapoints, but we want it to sample as many as the other workers so that all
            # processes can loop over batches in sync.
            # NOTE: this means that the E-step will sometimes write over a certain K[idx] and
            # lpj[idx] twice over the course of an epoch, even in the same batch (although that
            # will happen rarely). This double writing is not a race condition: the last write wins.
            n_samples = to.tensor(N)
            broadcast(n_samples, src=0)
            kwargs["sampler"] = ShufflingSampler(dataset, int(n_samples))
            kwargs["shuffle"] = None

        super().__init__(dataset, **kwargs)


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

    def write(self) -> None:
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
            H5Logger._write_one(f, k, v)
        f.close()

    @staticmethod
    def _write_one(f: h5py.Group, key: str, value: Any) -> None:
        if isinstance(value, to.Tensor):
            f.create_dataset(key, data=value.cpu())
        elif isinstance(value, dict):
            g = f.create_group(key)
            for k, v in value.items():
                H5Logger._write_one(g, k, v)
        else:
            try:
                f.create_dataset(key, data=value)
            except TypeError:
                f.create_dataset(key, data=str(value))


class ShufflingSampler(Sampler):
    def __init__(self, dataset: Dataset, n_samples: int = None):
        """A torch sampler that shuffles datapoints.

        :param dataset: The torch dataset for this sampler.
        :param n_samples: Number of desired samples. Defaults to len(dataset). If larger than
                          len(dataset), some datapoints will be sampled multiple times.
        """
        self._ds_len = len(dataset)
        self.n_samples = n_samples if n_samples is not None else self._ds_len

    def __iter__(self):
        idxs = np.arange(self._ds_len)
        np.random.shuffle(idxs)

        if self.n_samples > self._ds_len:
            n_extra_samples = self.n_samples - self._ds_len
            replace = True if n_extra_samples > idxs.size else False
            extra_samples = np.random.choice(idxs, size=n_extra_samples, replace=replace)
            idxs = np.concatenate((idxs, extra_samples))
        else:
            idxs = idxs[: self.n_samples]

        return iter(idxs)

    def __len__(self):
        return self.n_samples
