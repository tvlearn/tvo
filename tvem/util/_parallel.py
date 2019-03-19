# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import sys
import platform

from typing import Dict, Any

import torch as to
import torch.distributed as dist


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
