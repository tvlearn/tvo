# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import time
import datetime
import torch as to

import tvo
from tvo.exp import EVOConfig, ExpConfig, Training
from tvo.models import NoisyOR, BSC, SSSC
from tvo.utils.parallel import pprint, broadcast, barrier
from tvo.utils.param_init import init_W_data_mean, init_sigma2_default
from tvo.utils.model_protocols import Sampler

from params import get_args
from utils import init_processes, stdout_logger
from data import get_bars_gfs, generate_data_and_write_to_h5
from viz import Visualizer as _Visualizer, BSCVisualizer, SSSCVisualizer

DEVICE = tvo.get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


if __name__ == "__main__":

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
    D = int((args.H_gen / 2) ** 2)
    if comm_rank == 0:

        gfs = get_bars_gfs(no_bars=args.H_gen, bar_amp=args.bar_amp, precision=PRECISION)
        assert gfs.shape == (D, args.H_gen)
        pi_gen = args.pi_gen if args.pi_gen is not None else 2 / args.H_gen
        gen_model: Sampler
        if args.model == "nor":
            gen_model = NoisyOR(
                H=args.H_gen,
                D=D,
                W_init=gfs,
                pi_init=to.full((args.H_gen,), pi_gen, **dtype_device_kwargs),
                precision=PRECISION,
            )
        elif args.model == "bsc":
            gen_model = BSC(
                H=args.H_gen,
                D=D,
                W_init=gfs,
                sigma2_init=to.tensor([args.sigma2_gen], **dtype_device_kwargs),
                pies_init=to.full((args.H_gen,), pi_gen, **dtype_device_kwargs),
                precision=PRECISION,
            )
        elif args.model == "sssc":
            gen_model = SSSC(
                H=args.H_gen,
                D=D,
                W_init=gfs,
                sigma2_init=to.tensor([args.sigma2_gen], **dtype_device_kwargs),
                pies_init=to.full((args.H_gen,), pi_gen, **dtype_device_kwargs),
                mus_init=to.full((args.H_gen,), args.mu_gen, **dtype_device_kwargs),
                Psi_init=to.eye(args.H_gen, **dtype_device_kwargs) * args.Psi_gen,
                precision=PRECISION,
            )
        else:
            raise NotImplementedError("Generative model {} not supported".format(args.model))
        compute_ll = args.H_gen <= 10
        print(
            "Generating training dataset"
            + (
                "\nComputing log-likelihood of generative parameters given the data"
                if compute_ll
                else ""
            )
        )
        data, theta_gen, ll_gen = generate_data_and_write_to_h5(
            gen_model, data_file, args.no_data_points, compute_ll
        )

    # initialize model
    H = args.H if args.H is not None else args.H_gen
    W_init = (
        init_W_data_mean(data=data, H=H, dtype=PRECISION, device=DEVICE).contiguous()
        if comm_rank == 0
        else to.zeros((D, H), dtype=PRECISION, device=DEVICE)
    )
    sigma2_init = (
        init_sigma2_default(data, PRECISION, DEVICE)
        if comm_rank == 0
        else to.zeros((1), dtype=PRECISION, device=DEVICE)
    )  # obsolete for Noisy-OR
    barrier()
    broadcast(W_init)
    broadcast(sigma2_init)
    pies_init = to.full((H,), 1.0 / H, **dtype_device_kwargs)
    model_kwargs = {"H": H, "D": D, "W_init": W_init, "precision": PRECISION}
    model = {
        "nor": NoisyOR(pi_init=pies_init, **model_kwargs),
        "bsc": BSC(sigma2_init=sigma2_init, pies_init=pies_init, **model_kwargs),
        "sssc": SSSC(sigma2_init=sigma2_init, pies_init=pies_init, **model_kwargs),
    }[args.model]

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
    pprint("Initializing experiment")
    exp = Training(conf=exp_config, estep_conf=estep_conf, model=model, train_data_file=data_file)

    # initialize visualizer
    pprint("Initializing visualizer")
    Visualizer = (
        {"nor": _Visualizer, "bsc": BSCVisualizer, "sssc": SSSCVisualizer}[args.model]
        if comm_rank == 0
        else None
    )
    visualizer = (
        None
        if comm_rank != 0
        else Visualizer(  # type: ignore
            viz_every=args.viz_every if args.viz_every is not None else args.no_epochs,
            output_directory=output_directory,
            datapoints=data[:15],
            theta_gen=theta_gen,
            L_gen=ll_gen,
            gif_framerate=args.gif_framerate,
        )
    )
    barrier()

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # visualize epoch
        if comm_rank == 0:
            assert isinstance(visualizer, _Visualizer)  # to make mypy happy
            visualizer.process_epoch(
                epoch=epoch, F=summary._results["train_F"], theta=exp.trainer.model.theta
            )

    barrier()

    pprint("Finished")

    if comm_rank == 0:
        assert isinstance(visualizer, _Visualizer)  # to make mypy happy
        visualizer.finalize()

    barrier()
