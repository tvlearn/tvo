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
from tvo.models import BSC, PMCA
from tvo.utils.parallel import pprint, broadcast, barrier, bcast_shape, gather_from_processes
from tvo.utils.param_init import init_W_data_mean, init_sigma2_default
from tvo.utils.model_protocols import Reconstructor

from tvutil.prepost import (
    OverlappingPatches,
    MultiDimOverlappingPatches,
    mean_merger,
    median_merger,
)

from params import get_args
from utils import (
    init_processes,
    stdout_logger,
    get_image,
    store_as_h5,
    get_epochs_from_every,
    eval_fn,
)
from viz import Visualizer

DEVICE = tvo.get_device()
PRECISION = to.float32
dtype_device_kwargs = {"dtype": PRECISION, "device": DEVICE}


def gaussian_denoising_example():

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
    data_file, training_file = (
        output_directory + "/image_patches.h5",
        output_directory + "/training.h5",
    )
    txt_file = output_directory + "/terminal.txt"
    if comm_rank == 0:
        sys.stdout = stdout_logger(txt_file)  # type: ignore
    pprint("Will write training output to {}.".format(training_file))
    pprint("Will write terminal output to {}".format(txt_file))

    # generate noisy image and extract image patches
    patch_width = args.patch_width if args.patch_width is not None else args.patch_height
    if comm_rank == 0:
        clean = get_image(args.clean_image, args.rescale).to(**dtype_device_kwargs)
        isrgb = clean.dim() == 3 and clean.shape[2] == 3
        noisy = clean + args.noise_level * to.randn(clean.shape)
        print("Added white Gaussian noise with σ={}".format(args.noise_level))
        OVP = MultiDimOverlappingPatches if isrgb else OverlappingPatches
        ovp = OVP(noisy, args.patch_height, patch_width, patch_shift=1)
        train_data = ovp.get().t()
        store_as_h5({"data": train_data}, data_file)
    else:
        clean = None
    isrgb = len(bcast_shape(clean, 0)) == 3
    D = args.patch_height * patch_width * (3 if isrgb else 1)
    barrier()

    pprint("Initializing model")

    # initialize model
    W_init = (
        init_W_data_mean(data=train_data, H=args.H, dtype=PRECISION, device=DEVICE).contiguous()
        if comm_rank == 0
        else to.zeros((D, args.H), dtype=PRECISION, device=DEVICE)
    )
    sigma2_init = (
        init_sigma2_default(train_data, PRECISION, DEVICE)
        if comm_rank == 0
        else to.zeros((1), dtype=PRECISION, device=DEVICE)
    )
    barrier()
    broadcast(W_init)
    broadcast(sigma2_init)
    model = BSC(
        H=args.H,
        D=D,
        W_init=W_init,
        sigma2_init=sigma2_init,
        pies_init=to.full((args.H,), 2.0 / args.H, **dtype_device_kwargs),
        precision=PRECISION,
    )
    assert isinstance(model, Reconstructor)

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

    # setup the experiment
    merge_every = args.merge_every if args.merge_every is not None else args.viz_every
    reco_epochs = get_epochs_from_every(every=merge_every, total=args.no_epochs)
    exp_config = ExpConfig(
        batch_size=32,
        output=training_file,
        reco_epochs=reco_epochs,
        log_blacklist=["train_lpj", "train_states", "train_subs", "train_reconstruction"],
        log_only_latest_theta=True,
    )
    exp = Training(conf=exp_config, estep_conf=estep_conf, model=model, train_data_file=data_file)
    logger, trainer = exp.logger, exp.trainer
    # append the noisy image to the data logger
    if comm_rank == 0:
        logger.set_and_write(noisy_image=noisy)
    # define strategies to merge reconstructed patches
    merge_strategies = {"mean": mean_merger, "median": median_merger}

    # initialize visualizer
    pprint("Initializing visualizer")
    visualizer = (
        Visualizer(
            viz_every=args.viz_every,
            output_directory=output_directory,
            clean_image=clean,
            noisy_image=noisy,
            patch_size=(args.patch_height, patch_width),
            sort_gfs=True,
            ncol_gfs=4,
            gif_framerate=args.gif_framerate,
        )
        if comm_rank == 0
        else None
    )
    barrier()

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # merge reconstructed image patches and generate reconstructed image
        gather = epoch in (reco_epochs + 1)
        assert hasattr(trainer, "train_reconstruction")
        rec_patches = gather_from_processes(trainer.train_reconstruction) if gather else None
        merge = gather and comm_rank == 0
        imgs = {
            k: ovp.set_and_merge(rec_patches.t(), merge_method=v) if merge else None
            for k, v in merge_strategies.items()
        }
        barrier()

        # assess reconstruction quality in terms of PSNR
        psnrs = {k: eval_fn(clean, v) if merge else None for k, v in imgs.items()}

        to_log = (
            {
                **{f"reco_image_{k}": v for k, v in imgs.items()},
                **{f"psnr_{k}": v for k, v in psnrs.items()},
            }
            if merge
            else None
        )

        if to_log is not None:
            logger.append_and_write(**to_log)
        barrier()

        # visualize epoch
        if comm_rank == 0:
            visualizer.process_epoch(
                epoch=epoch,
                pies=model.theta["pies"],
                gfs=model.theta["W"],
                rec=imgs["mean"] if merge else None,
            )
        barrier()

    barrier()

    pprint("Finished")

    if comm_rank == 0:
        assert isinstance(visualizer, Visualizer)  # to make mypy happy
        visualizer.finalize()

    barrier()

########################################################


def poisson_denoising_example():

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
    data_file, training_file = (
        output_directory + "/image_patches.h5",
        output_directory + "/training.h5",
    )
    txt_file = output_directory + "/terminal.txt"
    if comm_rank == 0:
        sys.stdout = stdout_logger(txt_file)  # type: ignore
    pprint("Will write training output to {}.".format(training_file))
    pprint("Will write terminal output to {}".format(txt_file))

    # generate noisy image and extract image patches
    patch_width = args.patch_width if args.patch_width is not None else args.patch_height
    if comm_rank == 0:
        clean = get_image(args.clean_image, args.rescale).to(**dtype_device_kwargs)
        isrgb = clean.dim() == 3 and clean.shape[2] == 3

        I_orig = (clean / to.max(clean)) * args.noise_level   # rescaling the image to the desired peak value
        noisy = to.poisson(I_orig) #.astype(to.float64)
        print("Image with Poisson noise at peak={}".format(args.noise_level))
        OVP = MultiDimOverlappingPatches if isrgb else OverlappingPatches
        ovp = OVP(noisy, args.patch_height, patch_width, patch_shift=1)
        train_data = ovp.get().t()
        store_as_h5({"data": train_data}, data_file)
    else:
        clean = None
    isrgb = len(bcast_shape(clean, 0)) == 3
    D = args.patch_height * patch_width * (3 if isrgb else 1)
    barrier()

    pprint("Initializing model")

    # initialize model
    W_init = (
        init_W_data_mean(data=train_data, H=args.H, dtype=PRECISION, device=DEVICE).contiguous()
        if comm_rank == 0
        else to.zeros((D, args.H), dtype=PRECISION, device=DEVICE)
    )
    barrier()
    broadcast(W_init)
    model = PMCA(
        H=args.H,
        D=D,
        W_init=W_init,
        pies_init=to.full((args.H,), 2.0 / args.H, **dtype_device_kwargs),
        precision=PRECISION,
    )
    assert isinstance(model, Reconstructor)

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

    # setup the experiment
    merge_every = args.merge_every if args.merge_every is not None else args.viz_every
    reco_epochs = get_epochs_from_every(every=merge_every, total=args.no_epochs)
    exp_config = ExpConfig(
        batch_size=32,
        output=training_file,
        reco_epochs=reco_epochs,
        log_blacklist=["train_lpj", "train_states", "train_subs", "train_reconstruction"],
        log_only_latest_theta=True,
    )
    exp = Training(conf=exp_config, estep_conf=estep_conf, model=model, train_data_file=data_file)
    logger, trainer = exp.logger, exp.trainer
    # append the noisy image to the data logger
    if comm_rank == 0:
        logger.set_and_write(noisy_image=noisy)
    # define strategies to merge reconstructed patches
    merge_strategies = {"mean": mean_merger, "median": median_merger}

    # initialize visualizer
    pprint("Initializing visualizer")
    visualizer = (
        Visualizer(
            viz_every=args.viz_every,
            output_directory=output_directory,
            clean_image=clean,
            noisy_image=noisy,
            patch_size=(args.patch_height, patch_width),
            sort_gfs=True,
            ncol_gfs=4,
            gif_framerate=args.gif_framerate,
        )
        if comm_rank == 0
        else None
    )
    barrier()

    # run epochs
    for epoch, summary in enumerate(exp.run(args.no_epochs)):
        summary.print()

        # merge reconstructed image patches and generate reconstructed image
        gather = epoch in (reco_epochs + 1)
        assert hasattr(trainer, "train_reconstruction")
        rec_patches = gather_from_processes(trainer.train_reconstruction) if gather else None
        merge = gather and comm_rank == 0
        imgs = {
            k: ovp.set_and_merge(rec_patches.t(), merge_method=v) if merge else None
            for k, v in merge_strategies.items()
        }
        barrier()

        # assess reconstruction quality in terms of PSNR
        psnrs = {k: eval_fn(clean, v * (to.max(clean) / args.noise_level)) if merge else None for k, v in imgs.items()}

        to_log = (
            {
                **{f"reco_image_{k}": v for k, v in imgs.items()},
                **{f"psnr_{k}": v for k, v in psnrs.items()},
            }
            if merge
            else None
        )

        if to_log is not None:
            logger.append_and_write(**to_log)
        barrier()

        # visualize epoch
        if comm_rank == 0:
            visualizer.process_epoch(
                epoch=epoch,
                pies=model.theta["pies"],
                gfs=model.theta["W"],
                rec=imgs["mean"] if merge else None,
            )
        barrier()

    barrier()

    pprint("Finished")

    if comm_rank == 0:
        assert isinstance(visualizer, Visualizer)  # to make mypy happy
        visualizer.finalize()

    barrier()

if __name__ == "__main__":
    #gaussian_denoising_example()
    poisson_denoising_example()
