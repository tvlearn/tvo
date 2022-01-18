#!/bin/bash
# Copyright (C) 2019 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

# Run tests and only have one MPI process to print to stdout.
# Run as `mpirun -n N bash run_mpi_tests.sh` where N is the desired number of processes.

if [[ ! -v OMPI_COMM_WORLD_RANK ]]; then
   echo "Usage: mpirun -n <N> bash $0 [coverage_rank]" >&2
   echo 'coverage_rank: optionally, specify for which rank to record coverage information (default is 0)' >&2
   exit 1
fi

COV_RANK=${1-0} # $1 if present, 0 otherwise
if [[ $OMPI_COMM_WORLD_RANK -eq $COV_RANK ]]; then
   pytest -x --cov=tvo --cov-append -v -m mpi test
else
   pytest -q -x -m mpi test
fi
