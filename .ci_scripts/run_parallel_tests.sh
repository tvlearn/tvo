#!/bin/bash
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# Run tests and only have one MPI process to print to stdout.
# Run as `mpirun -n N bash run_parallel_tests.sh` where N is the desired number of processes.

if [[ ! -v OMPI_COMM_WORLD_RANK ]]; then
   echo 'Usage: mpirun -n <N> bash $0' >&2
   exit 1
fi

if [[ $OMPI_COMM_WORLD_RANK -eq 0 ]]; then
   pytest --cov=tvem --cov-append -v -m parallel test
else
   pytest -m parallel test > /dev/null
fi
