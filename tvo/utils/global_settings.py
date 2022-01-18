# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import torch as to


def _choose_device() -> to.device:
    dev = to.device("cpu")
    if "TVO_GPU" in os.environ:
        gpu_n = int(os.environ["TVO_GPU"])
        dev = to.device(f"cuda:{gpu_n}")
    return dev


class _GlobalDevice:
    """A singleton object containing the global device settings for the framework.

    Set and get the corresponding to.device with `{set,get}_device()`.
    """

    _device: to.device = _choose_device()

    @classmethod
    def get_device(cls) -> to.device:
        return cls._device

    @classmethod
    def set_device(cls, dev: to.device):
        cls._device = dev


def get_device() -> to.device:
    """Get the torch.device that all objects in the package will use by default.

    The default ('cpu') can be changed by setting the `TVO_GPU` environment variable
    to the number of the desired CUDA device. For example, in bash, `export TVO_GPU=0`
    will make the framework default to device 'cuda:0'. Note that some computations might
    still be performed on CPU for performance reasons.
    """
    return _GlobalDevice.get_device()


def _set_device(dev: to.device):
    """Private method to change the TVO device settings. USE WITH CARE."""
    _GlobalDevice.set_device(dev)


def _choose_run_policy() -> str:
    policy = "seq"
    if (
        "TVO_MPI" in os.environ and os.environ["TVO_MPI"] != 0
    ) or "OMPI_COMM_WORLD_SIZE" in os.environ:
        policy = "mpi"
    return policy


class _GlobalPolicy:
    """A singleton object containing the global execution policy for the framework.

    Set and get the policy with `{set,get}_run_policy()`.
    """

    _policy: str = _choose_run_policy()

    @classmethod
    def get_policy(cls) -> str:
        return cls._policy


def get_run_policy() -> str:
    """Get the current parallelization policy.

    * `'seq'`: the framework will not perform any parallelization other than what torch tensors
      offer out of the box on the relevant device.
    * `'mpi'`: the framework will perform data parallelization for the algorithms that
      implement it.

    The policy is 'seq' unless the framework detects that the program is running within `mpirun`,
    in which case the policy is 'mpi'. The default can also be overridden by setting the
    `TVO_MPI` environment variable to a non-zero value, e.g. in bash with
    `export TVO_MPI=1`.
    """
    return _GlobalPolicy.get_policy()
