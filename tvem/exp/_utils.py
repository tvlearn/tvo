# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import torch as to
from typing import Union

from tvem.variational import (
    FullEM,
    EEMVariationalStates,
    FullEMSingleCauseModels,
    TVSVariationalStates,
    RandomSampledVarStates,
)
from tvem.exp._EStepConfig import (
    FullEMConfig,
    EEMConfig,
    EStepConfig,
    FullEMSingleCauseConfig,
    TVSConfig,
    RandomSamplingConfig,
)


def make_var_states(
    conf: EStepConfig, N: int, H: int, precision: to.dtype
) -> Union[
    EEMVariationalStates,
    FullEM,
    FullEMSingleCauseModels,
    TVSVariationalStates,
    RandomSampledVarStates,
]:

    if isinstance(conf, FullEMConfig):
        assert conf.n_states == 2 ** H, "FullEMConfig and model have different H"
        return FullEM(N, H, precision)
    elif isinstance(conf, FullEMSingleCauseConfig):
        assert conf.n_states == H, "FullEMSingleCauseConfig and model have different H"
        return FullEMSingleCauseModels(N, H, precision)
    elif isinstance(conf, EEMConfig):
        return _make_EEM_var_states(conf, N, H, precision)
    elif isinstance(conf, TVSConfig):
        return TVSVariationalStates(
            N,
            H,
            conf.n_states,
            precision,
            conf.n_prior_samples,
            conf.n_marginal_samples,
            conf.K_init_file,
        )
    elif isinstance(conf, RandomSamplingConfig):
        return RandomSampledVarStates(
            N, H, conf.n_states, precision, conf.n_samples, conf.sparsity, conf.K_init_file
        )
    else:  # pragma: no cover
        raise NotImplementedError()


def _make_EEM_var_states(conf: EEMConfig, N: int, H: int, precision: to.dtype):
    selection = {"fitness": "batch_fitparents", "uniform": "randparents"}[conf.parent_selection]
    mutation = {"sparsity": "sparseflip", "uniform": "randflip"}[conf.mutation]
    return EEMVariationalStates(
        N=N,
        H=H,
        S=conf.n_states,
        precision=precision,
        parent_selection=selection,
        mutation=mutation,
        n_parents=conf.n_parents,
        n_generations=conf.n_generations,
        n_children=conf.n_children if not conf.crossover else None,
        crossover=conf.crossover,
        bitflip_frequency=conf.bitflip_frequency,
        K_init_file=conf.K_init_file,
    )
