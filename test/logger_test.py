# -*- coding: utf-8 -*-
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

from tvem.util.data import H5Logger
import torch as to
import torch.distributed as dist
import pytest
import h5py
import os


@pytest.fixture(scope='function')
def file_and_logger():
    fname = 'logger_test_output.h5'

    yield fname, H5Logger(fname)

    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank == 0:
        os.remove(fname)
        os.remove(fname + '.old')


def check_contents(fname):
    rank = dist.get_rank() if dist.is_initialized() else 0
    if rank != 0:
        return

    f = h5py.File(fname, 'r')
    t = f['v']
    assert (to.tensor(t) == to.arange(6).reshape(2, 3)).all()
    f.close()


@pytest.mark.mpi
def test_append(file_and_logger):
    fname, logger = file_and_logger
    logger.append(v=to.arange(3))
    logger.write()
    logger.append(v=to.arange(3, 6))
    logger.write()
    check_contents(fname)


@pytest.mark.mpi
def test_set(file_and_logger):
    fname, logger = file_and_logger
    logger.set(v=to.arange(6).reshape(2, 3))
    logger.write()
    logger.set(v=to.arange(6).reshape(2, 3))
    logger.write()
    check_contents(fname)
