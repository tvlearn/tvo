# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import platform
import math

import torch
import torch.distributed as dist
from torch import Tensor
from typing import Iterable, Union, Dict

import tvem


def pprint(obj: object = "", end: str = "\n"):
    """Print on root process of torch.distributed process group.

    param obj: Message to print
    param end: Suffix of message. Default is linebreak.
    """
    if tvem.get_run_policy() == "mpi" and dist.get_rank() != 0:
        return

    print(obj, end=end)


def init_processes(multi_node: bool = False):
    """Initialize MPI process group using torch.distributed module.

    param multi_node: Deploy multiple computing nodes.

    Eventually updates the value of tvem.device.
    """
    if torch.distributed.is_initialized():
        return

    dist.init_process_group("mpi")

    global_rank = dist.get_rank()
    comm_size = dist.get_world_size()

    if tvem.get_device().type == "cuda":
        device_count = int(torch.cuda.device_count())
        if multi_node:
            node_count = comm_size // device_count
        else:
            node_count = 1
        # 0..device_count (first_node), ..., 0..device_count (last_node)
        local_rank = (list(range(device_count)) * node_count)[global_rank]
        device_str = "cuda:%i" % local_rank
    else:
        device_str = "cpu"

    tvem._set_device(torch.device(device_str))

    pprint("Initializting %i processes." % comm_size)
    print(
        "New process on %s. Global rank %d. Device %s. Total no processes %d."
        % (platform.node(), global_rank, device_str, comm_size)
    )


def bcast_dtype(data: Tensor, src: int = 0) -> torch.dtype:
    """Broadcast dtype of data on src rank.

    :param data: Tensor on src rank
    :param src: Source rank
    :returns: dtype on each rank
    """
    if tvem.get_run_policy() == "seq":
        return data.dtype
    else:
        comm_rank = dist.get_rank()

        dtypes = [
            torch.float32,
            torch.float64,
            torch.float16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ]

        ind_dtype = torch.empty((1,), dtype=torch.uint8)
        if comm_rank == src:
            dtype = data.dtype
            ind_dtype[:] = [*map(str, dtypes)].index(str(dtype))
        dist.broadcast(ind_dtype, 0)
        return dtypes[ind_dtype.item()]


def bcast_shape(data: Tensor, src: int) -> Tensor:
    """Broadcast shape of data on src rank.

    :param data: Tensor on src rank
    :param src: Source rank
    :returns: Tensor with shape on each rank
    """
    if tvem.get_run_policy() == "seq":
        return torch.tensor(data.shape)
    else:
        comm_rank = dist.get_rank()

        ndim = torch.empty((1,), dtype=torch.int64)
        if comm_rank == src:
            ndim[:] = data.dim()
        dist.broadcast(ndim, src)
        shape = torch.empty((ndim.item(),), dtype=torch.int64)
        if comm_rank == src:
            shape[:] = torch.tensor(data.shape)
        dist.broadcast(shape, src)
        return shape


def scatter_to_processes(*tensors: Tensor, src: int = 0) -> Iterable[Tensor]:
    """Split tensors into chunks and scatter within process group.

    :param tensors: Tensor to be scattered. Chunks are cut along dimension 0.
    :param src: Source rank to scatter from.
    :returns: Tensor scattered to local rank.

    Tensor data is assumed to be None on all but the root processes.
    """
    my_tensors = []

    if tvem.get_run_policy() == "seq":
        for data in tensors:
            my_tensors.append(data)

    elif tvem.get_run_policy() == "mpi":
        comm_size, comm_rank = dist.get_world_size(), dist.get_rank()
        for data in tensors:

            this_dtype = bcast_dtype(data, src)
            this_device = tvem.get_device()

            shape = bcast_shape(data, src)
            total_length = shape[0].item()
            other_length = tuple(shape[1:])

            # logic to ensure that input to `dist.scatter` is evenly divisible by comm_size
            assert (
                total_length / comm_size
            ) >= 1, "number of data points must be greater or equal to number of MPI processes"
            local_length_ceiled = math.ceil(total_length / comm_size)
            total_length_ceiled = local_length_ceiled * comm_size
            no_dummy = total_length_ceiled - total_length
            local_length = local_length_ceiled - 1 if comm_rank < no_dummy else local_length_ceiled

            # split into chunks and scatter
            chunks = []  # type: ignore
            if comm_rank == 0:
                to_cut_into_chunks = torch.zeros(
                    ((total_length_ceiled,) + other_length), dtype=this_dtype, device=this_device
                )
                local_start = 0
                for r in range(comm_size):
                    local_length_ = local_length_ceiled - 1 if r < no_dummy else local_length_ceiled
                    to_cut_into_chunks[
                        r * local_length_ceiled : r * local_length_ceiled + local_length_
                    ] = data[range(local_start, local_start + local_length_)]
                    local_start += local_length_
                chunks = list(torch.chunk(to_cut_into_chunks, comm_size, dim=0))

            my_data = torch.zeros(
                (local_length_ceiled,) + other_length, dtype=this_dtype, device=this_device
            )

            dist.scatter(my_data, src=src, scatter_list=chunks)

            my_data = my_data[:local_length]

            N = torch.tensor([local_length])
            all_reduce(N)
            assert N.item() == total_length

            my_tensors.append(my_data)

    return my_tensors[0] if len(my_tensors) == 1 else my_tensors


def gather_from_processes(*my_tensors: Tensor, dst: int = 0) -> Union[Tensor, Iterable[Tensor]]:
    """Gather tensors from process group.

    :param my_tensors: List of tensors to be gathered from local process on process dst.
                       For each element tensor.shape[1:] must be identical on
                       each process.
    :param dst: Rank of destination process to gather tensors.
    :returns: List of tensors gathered from process group.

    Only process with rank dst will contain gathered data.
    """
    tensors = []

    if tvem.get_run_policy() == "seq":
        for data in my_tensors:
            tensors.append(data)

    elif tvem.get_run_policy() == "mpi":
        comm_size, comm_rank = dist.get_world_size(), dist.get_rank()
        for my_data in my_tensors:

            local_length = my_data.shape[0]
            other_length = tuple(my_data.shape[1:])
            total_length = torch.tensor([local_length])
            all_reduce(total_length)
            total_length = total_length.item()
            local_length_ceiled = math.ceil(total_length / comm_size)
            no_dummy = local_length_ceiled * comm_size - total_length

            chunks = (
                [
                    torch.zeros(
                        (local_length_ceiled,) + other_length,
                        dtype=my_data.dtype,
                        device=my_data.device,
                    )
                    for r in range(comm_size)
                ]
                if comm_rank == 0
                else []
            )

            dist.gather(
                tensor=torch.cat(
                    (
                        my_data,
                        torch.zeros(
                            (1,) + other_length, dtype=my_data.dtype, device=my_data.device
                        ),
                    )
                )
                if comm_rank < no_dummy
                else my_data,
                gather_list=chunks,
                dst=dst,
            )

            if comm_rank == 0:
                for r in range(no_dummy):
                    chunks[r] = chunks[r][:-1]
                data = torch.cat(chunks)
                assert data.shape[0] == total_length
                tensors.append(data)

    return tensors[0] if len(tensors) == 1 else tensors


def all_reduce(tensor: Tensor, op=dist.ReduceOp.SUM):
    """Equivalent to torch's all_reduce if tvem.get_run_policy() is 'mpi', no-op otherwise."""
    if tvem.get_run_policy() == "mpi":
        dist.all_reduce(tensor, op)


def broadcast(tensor: Tensor, src: int = 0):
    """Equivalent to torch's broadcast if tvem.get_run_policy() is 'mpi', no-op otherwise."""
    if tvem.get_run_policy() == "mpi":
        dist.broadcast(tensor, src)


def barrier():
    """Equivalent to torch's dist.barrier if tvem.get_run_policy() is 'mpi', no-op otherwise."""
    if tvem.get_run_policy() == "mpi":
        dist.barrier()


def mpi_average_grads(theta: Dict[str, torch.Tensor]) -> None:
    """Average gradients across processes. See https://bit.ly/2FlJsxS.

    :param theta: dictionary with torch.tensors storing TVEM model parameters
    """
    if tvem.get_run_policy() != "mpi":
        return  # nothing to do

    n_procs = dist.get_world_size()
    parameters = [p for p in theta.values() if p.requires_grad]
    with torch.no_grad():
        for p in parameters:
            all_reduce(p.grad)
            p.grad /= n_procs
