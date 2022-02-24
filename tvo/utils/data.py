# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
import torch.distributed as dist
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import numpy as np
import tvo
from tvo.utils.parallel import broadcast


class TVODataLoader(DataLoader):
    def __init__(self, *data: to.Tensor, **kwargs):
        """TVO DataLoader class. Derived from torch.utils.data.DataLoader.

        :param data: Tensor containing the input dataset. Must have exactly two dimensions (N,D).
        :param kwargs: forwarded to pytorch's DataLoader.

        TVODataLoader is constructed exactly the same way as pytorch's DataLoader,
        but it restricts datasets to TensorDataset constructed from the *data passed
        as parameter. All other arguments are forwarded to pytorch's DataLoader.

        When iterated over, TVODataLoader yields a tuple containing the indeces of
        the datapoints in each batch as well as the actual datapoints for each
        tensor in the input Tensor.

        TVODataLoader instances optionally expose the attribute `precision`, which is set to the
        dtype of the first dataset in *data if it is a floating point dtype.
        """
        N = data[0].shape[0]
        assert all(d.shape[0] == N for d in data), "Dimension mismatch in data sets."

        if data[0].dtype is not to.uint8:
            self.precision = data[0].dtype

        dataset = TensorDataset(to.arange(N), *data)

        if tvo.get_run_policy() == "mpi" and "sampler" not in kwargs:
            # Number of _desired_ datapoints per worker: the last worker might have less actual
            # datapoints, but we want it to sample as many as the other workers so that all
            # processes can loop over batches in sync.
            # NOTE: this means that the E-step will sometimes write over a certain K[idx] and
            # lpj[idx] twice over the course of an epoch, even in the same batch (although that
            # will happen rarely). This double writing is not a race condition: the last write wins.
            n_samples = to.tensor(N)
            assert dist.is_initialized()
            comm_size = dist.get_world_size()
            # Ranks ..., (comm_size-2), (comm_size-1) are
            # assigned one data point more than ranks
            # 0, 1, ... if the dataset cannot be evenly
            # distributed across MPI processes. The split
            # point depends on the total number of data
            # points and number of MPI processes (see
            # scatter_to_processes, gather_from_processes)
            broadcast(n_samples, src=comm_size - 1)
            kwargs["sampler"] = ShufflingSampler(dataset, int(n_samples))
            kwargs["shuffle"] = None

        super().__init__(dataset, **kwargs)


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
            extra_samples = np.random.choice(
                idxs, size=n_extra_samples, replace=replace
            )
            idxs = np.concatenate((idxs, extra_samples))
        else:
            idxs = idxs[: self.n_samples]

        return iter(idxs)

    def __len__(self):
        return self.n_samples
