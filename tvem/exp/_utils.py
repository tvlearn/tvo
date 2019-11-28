# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import h5py
import torch as to
import torch.distributed as dist
from typing import Tuple, Union

import tvem
from tvem.variational import FullEM, EEMVariationalStates
from tvem.exp._EStepConfig import FullEMConfig, EEMConfig, EStepConfig
from tvem.utils.parallel import scatter_to_processes


def make_var_states(
    conf: EStepConfig, N: int, H: int, precision: to.dtype
) -> Union[EEMVariationalStates, FullEM]:
    if isinstance(conf, FullEMConfig):
        return FullEM(N, H, precision)
    elif isinstance(conf, EEMConfig):
        return _make_EEM_var_states(conf, N, H, precision)
    else:  # pragma: no cover
        raise NotImplementedError()


def _make_EEM_var_states(conf: EEMConfig, N: int, H: int, precision: to.dtype):
    selection = {"fitness": "batch_fitparents", "uniform": "randparents"}[conf.parent_selection]
    mutation = {"sparsity": "sparseflip", "uniform": "randflip"}[conf.mutation]
    eem_conf = {
        "parent_selection": selection,
        "mutation": mutation,
        "n_parents": conf.n_parents,
        "n_children": conf.n_children if not conf.crossover else None,
        "n_generations": conf.n_generations,
        "S": conf.n_states,
        "N": N,
        "H": H,
        "crossover": conf.crossover,
        "precision": precision,
        "bitflip_frequency": conf.bitflip_frequency,
    }
    return EEMVariationalStates(**eem_conf)


def get_h5_dataset_to_processes(fname: str, possible_keys: Tuple[str, ...]) -> to.Tensor:
    """Return dataset with the first of `possible_keys` that is found in hdf5 file `fname`."""
    rank = dist.get_rank() if dist.is_initialized() else 0

    f = h5py.File(fname, "r")
    for dataset in possible_keys:
        if dataset in f.keys():
            break
    else:  # pragma: no cover
        raise ValueError(f'File "{fname}" does not contain any of keys {possible_keys}')
    if rank == 0:
        data = to.tensor(f[dataset][...], device=tvem.get_device())
    else:
        data = None
    return scatter_to_processes(data)
