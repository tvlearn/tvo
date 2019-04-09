# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import platform

import torch as to
import torch.distributed as dist
from torch import Tensor
from typing import Iterable

import tvem


def pprint(obj: object = "", end: str = '\n'):
    """Print on root process of torch.distributed process group.

    param obj: Message to print
    param end: Suffix of message. Default is linebreak.
    """
    if tvem.get_run_policy() == 'mpi' and dist.get_rank() != 0:
        return

    print(obj, end=end)


def init_processes(multi_node: bool = False):
    """Initialize MPI process group using torch.distributed module.

    param multi_node: Deploy multiple computing nodes.

    Eventually updates the value of tvem.device.
    """
    if to.distributed.is_initialized():
        return

    dist.init_process_group('mpi')

    global_rank = dist.get_rank()
    comm_size = dist.get_world_size()

    if tvem.get_device().type == 'cuda':
        device_count = int(to.cuda.device_count())
        if multi_node:
            node_count = comm_size // device_count
        else:
            node_count = 1
        # 0..device_count (first_node), ..., 0..device_count (last_node)
        local_rank = (list(range(device_count))*node_count)[global_rank]
        device_str = 'cuda:%i' % local_rank
    else:
        device_str = 'cpu'

    tvem._set_device(device_str)

    pprint("Initializting %i processes." % comm_size)
    print("New process on %s. Global rank %d. Device %s. Total no processes %d." % (
        platform.node(), global_rank, device_str, comm_size))


def scatter2processes(*tensors: Tensor, src: int = 0, dtype: to.dtype = None,
                      device: to.device = None) -> Iterable[Tensor]:
    """Split tensors into chunks and scatter within process group.

    :param data: Tensor to be scattered. Chunks are cut along dimension 0.
    :param src: Source rank to scatter from.
    :param dtype: dtype of resulting tensor. Defaults to the dtype of the corresponding
                  input tensor if not specified.
    :param device: device of resulting tensor. Defaults to the device of the corresponding
                   input tensor if not specified.
    :returns: Tensor scattered to local rank.

    Tensor data is assumed to be None on all but the root processes.
    """
    my_tensors = []

    if tvem.get_run_policy() == 'seq':
        for data in tensors:
            this_dtype = data.dtype if dtype is None else dtype
            this_device = data.device if device is None else device
            my_tensors.append(data.to(dtype=this_dtype, device=this_device))
    return my_tensors[0] if len(my_tensors) == 1 else my_tensors

    comm_size, comm_rank = dist.get_world_size(), dist.get_rank()

    for data in tensors:
        this_dtype = data.dtype if dtype is None else dtype
        this_device = data.device if device is None else device
        ndim = to.empty((1,), dtype=to.int64)
        if comm_rank == src:
            ndim[:] = data.dim()
        dist.broadcast(ndim, src)
        shape = to.empty((ndim.item(),), dtype=to.int64)
        if comm_rank == src:
            shape[:] = to.tensor(data.shape)
        dist.broadcast(shape, src)
        total_length, other_length = shape[0], shape[1:]

        # no datapoints per commrank including dummy rows
        local_length_ = int(to.ceil(to.tensor([float(total_length) / comm_size])))
        # determine number of and eventually add dummy rows for scatter/gather compatibility
        empty_length = local_length_ * comm_size - total_length
        local_length = local_length_
        if comm_rank == comm_size-1:
            local_length -= empty_length  # no datapoints per commrank excluding dummy rows actual
            # last commrank eventually runs on smaller chunk

        dist.barrier()

        # split into chunks and scatter
        chunks = []  # type: ignore
        if comm_rank == 0:
            chunks = list(to.chunk(to.cat((data.to(dtype=this_dtype, device=this_device),
                                           to.zeros((empty_length, other_length),
                                                    dtype=this_dtype, device=this_device)),
                                          dim=0),
                                   comm_size, dim=0))

        my_data = to.zeros((local_length_, other_length), dtype=this_dtype, device=this_device)

        dist.scatter(my_data, src=src, scatter_list=chunks)

        # remove dummy rows again before actual computation starts
        if empty_length != 0:
            if comm_rank == comm_size-1:
                my_data = my_data[:local_length, :]

        my_tensors.append(my_data)

    return my_tensors[0] if len(my_tensors) == 1 else my_tensors


def all_reduce(tensor: Tensor, op=dist.ReduceOp.SUM):
    """Equivalent to torch's all_reduce if tvem.get_run_policy() is 'mpi', no-op otherwise."""
    if tvem.get_run_policy() == 'mpi':
        dist.all_reduce(tensor, op)
