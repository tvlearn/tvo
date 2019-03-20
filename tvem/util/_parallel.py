# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import sys
import platform

from typing import Dict, Any

import torch as to
import torch.distributed as dist
from torch import Tensor


def pprint(obj: object = "", dist: to.distributed = dist, end: str = '\n'):
    """Print on root process of torch.distributed process group.

    param obj: Message to print
    param dist: torch.distributed module
    param end: Suffix of message. Default is linebreak.

    Adapted from https://github.com/jbornschein/mpi4py-examples.
    """
    if dist.get_rank() != 0:
        return

    if isinstance(obj, str):
        sys.stdout.write(obj + end)
    else:
        sys.stdout.write(repr(obj))
        sys.stdout.write(end)
        sys.stdout.flush()


def init_processes(cuda: bool = False, multi_node: bool = False) -> Dict[str, Any]:
    """Initialize MPI process group using torch.distributed module.

    param cuda: Deploy CUDA devices.
    param multi_node: Deploy multiple computing nodes.
    returns: Dictionary containing the process_group, and the device information.
    """

    comm = {
        'process_group': dist.init_process_group('mpi'),
        'device_count': int(to.cuda.device_count())
    }

    global_rank = dist.get_rank()
    comm_size = dist.get_world_size()

    if cuda:
        if multi_node:
            node_count = int(float(comm_size) / comm['device_count'])
        else:
            node_count = 1
        # 0..device_count (first_node), ..., 0..device_count (last_node)
        local_rank = (
            list(range(comm['device_count']))*node_count)[global_rank]
        comm['device_str'] = 'cuda:%i' % local_rank
    else:
        comm['device_str'] = 'cpu'

    comm['device'] = to.device(comm['device_str'])

    pprint("Initializting %i processes." % comm_size)
    print("New process on %s. Global rank %d. Device %s. Total no processes %d." % (
        platform.node(), global_rank, comm['device_str'], comm_size))

    return comm


def scatter2processes(data: Tensor, src: int = 0, dtype: to.dtype = to.float64,
                      device: to.device = to.device('cpu')):
    """Split tensor into chunks and scatter within process group.

    param data: Tensor to be scattered. Chunks are cut along dimension 0.
    param src: Source rank to scatter from.
    param dtype: dtype of resulting tensor.
    param device: device of resulting tensor.

    Tensor data is assumed to be None on all but the root processes.
    """

    comm_size, comm_rank = dist.get_world_size(), dist.get_rank()

    ndim = to.empty((1,), dtype=to.uint64)
    if comm_rank == src:
        ndim[:] = data.dim()
    dist.broadcast(ndim, src)
    shape = to.empty((ndim.item(),), dtype=to.int64)
    if comm_rank == src:
        shape[:] = to.tensor(data.shape)
    dist.broadcast(shape, src)
    total_length, other_length = shape[0], shape[1:]

    # determine number of and eventually add dummy rows for scatter/gather compatibility
    # no datapoints per
    local_length_ = int(to.ceil(to.tensor([float(total_length)/comm_size])))
    # commrank including
    # dummy rows
    empty_length = local_length_ * comm_size - total_length
    local_length = local_length_
    if comm_rank == comm_size-1:
        local_length -= empty_length  # no datapoints per commrank excluding dummy rows actual
        # last commrank eventually runs on smaller chunk

    dist.barrier()

    # split into chunks and scatter
    chunks = []  # type: ignore
    if comm_rank == 0:
        chunks = list(to.chunk(to.cat((data.to(dtype=dtype, device=device), to.zeros(
            (empty_length, other_length), dtype=dtype, device=device)), dim=0), comm_size, dim=0))

    my_data = to.zeros((local_length_, other_length),
                       dtype=dtype, device=device)

    dist.scatter(my_data, src=src, scatter_list=chunks)

    # remove dummy rows again before actual computation starts
    if empty_length != 0:
        if comm_rank == comm_size-1:
            my_data = my_data[:local_length, :]

    return my_data
