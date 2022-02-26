# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import time
import datetime
import numpy as np
import torch as to

import tvo
from tvo.exp import EVOConfig, ExpConfig, Training
from tvo.models import BSC
from tvo.utils.parallel import pprint, broadcast, barrier, bcast_shape
from tvo.utils.param_init import init_W_data_mean, init_sigma2_default

from params import get_args
from utils import init_processes, stdout_logger, prepare_training_dataset
from viz import Visualizer

DEVICE = tvo.get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


def feature_learning():

    # initialize MPI (if executed with env TVO_MPI=...), otherwise pass
    comm_rank = init_processes()[0]

    # get hyperparameters
    args = get_args()
    pprint("Argument list:")
    for k in sorted(vars(args), key=lambda s: s.lower()):
        pprint("{: <25} : {}".format(k, vars(args)[k]))

    # determine directories to save output
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime("%y-%m-%d-%H-%M-%S")
    output_directory = (
        f"./out/{timestamp}" if args.output_directory is None else args.output_directory
    )
    os.makedirs(output_directory, exist_ok=True)
    data_file, training_file = output_directory + "/data.h5", output_directory + "/training.h5"
    txt_file = output_directory + "/terminal.txt"
    if comm_rank == 0:
        sys.stdout = stdout_logger(txt_file)  # type: ignore
    pprint("Will write training output to {}.".format(training_file))
    pprint("Will write terminal output to {}".format(txt_file))

    # generate data set
    pprint("Preparing image patches dataset")
    data = (
        prepare_training_dataset(args.image_file, args.patch_size, args.no_patches, data_file).to(
            **dtype_device_kwargs
        )
        if comm_rank == 0
        else None
    )
    N, D = bcast_shape(data, 0)
    if comm_rank != 0:
        data = to.zeros(N, D, **dtype_device_kwargs)
    barrier()
    broadcast(data)

    # initialize model
    pprint("Initializing model")
    W_init = (
        init_W_data_mean(data=data, H=args.H, **dtype_device_kwargs).contiguous()
        if comm_rank == 0
        else to.zeros((D, args.H), **dtype_device_kwargs)
    )
    sigma2_init = (
        init_sigma2_default(data, PRECISION, DEVICE)
        if comm_rank == 0
        else to.zeros((1), **dtype_device_kwargs)
    )
    barrier()
    broadcast(W_init)
    broadcast(sigma2_init)
    pies_init = to.full((args.H,), 1.0 / args.H, **dtype_device_kwargs)
    model = BSC(
        H=args.H,
        D=D,
        W_init=W_init,
        precision=PRECISION,
        sigma2_init=sigma2_init,
        pies_init=pies_init,
    )

    pprint("Initializing experiment")

    # define hyperparameters of the variational optimization
    estep_conf = EVOConfig(
        n_states=args.Ksize,
        n_parents=args.no_parents,
        n_children=args.no_children,
        n_generations=args.no_generations,
        parent_selection=args.selection,
        crossover=args.crossover,
    )

    # define general hyperparameters of the experiment
    exp_config = ExpConfig(batch_size=32, output=training_file)
    exp = Training(conf=exp_config, estep_conf=estep_conf, model=model, train_data_file=data_file)

    # initialize visualizer
    pprint("Initializing visualizer")
    visualizer = (
        None
        if comm_rank != 0
        else Visualizer(  # type: ignore
            output_directory=output_directory,
            viz_every=args.viz_every if args.viz_every is not None else args.no_epochs,
            datapoints=data[np.random.permutation(data.shape[0])[:16]],
            patch_size=args.patch_size,
            sort_acc_to_desc_priors=True,
            ncol_gfs=16,
            gif_framerate=args.gif_framerate,
        )
    )
    barrier()

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # visualize epoch
        if comm_rank == 0:
            visualizer.process_epoch(
                epoch=epoch,
                F=summary._results["train_F"],
                pies=model.theta["pies"],
                gfs=model.theta["W"],
            )
        barrier()

    barrier()

    pprint("Finished")

    if comm_rank == 0:
        assert isinstance(visualizer, Visualizer)  # to make mypy happy
        visualizer.finalize()

    barrier()


if __name__ == "__main__":
    feature_learning()
