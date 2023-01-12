# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse


awgn_parser = argparse.ArgumentParser(add_help=False)
awgn_parser.add_argument(
    "--clean_image",
    type=str,
    help="Full path to clean image (png, jpg, ... file)",
    default="./img/house.png",
)

awgn_parser.add_argument(
    "--rescale",
    type=float,
    help="If specified, the size of the clean image will be rescaled by this factor "
    "(only for demonstration purposes to minimize computational effort)",
    default=0.5,
)

awgn_parser.add_argument(
    "--noise_level",
    type=int,
    help="Standard deviation of the additive white Gaussian noise",
    default=25,
)


patch_parser = argparse.ArgumentParser(add_help=False)
patch_parser.add_argument(
    "--patch_height",
    type=int,
    help="Patch height",
    default=5,
)

patch_parser.add_argument(
    "--patch_width",
    type=int,
    help="Patch width (defaults to patch_height if not specified)",
    default=None,
)


variational_parser = argparse.ArgumentParser(add_help=False)
variational_parser.add_argument(
    "--Ksize",
    type=int,
    help="Size of the K sets (i.e., S=|K|)",
    default=50,
)

variational_parser.add_argument(
    "--selection",
    type=str,
    help="Selection operator",
    choices=["fitness", "uniform"],
    default="fitness",
)

variational_parser.add_argument(
    "--crossover",
    action="store_true",
    help="Whether to apply crossover. Must be False if no_children is specified",
    default=False,
)

variational_parser.add_argument(
    "--no_parents",
    type=int,
    help="Number of parental states to select per generation",
    default=20,
)

variational_parser.add_argument(
    "--no_children",
    type=int,
    help="Number of children to evolve per generation",
    default=2,
)

variational_parser.add_argument(
    "--no_generations",
    type=int,
    help="Number of generations to evolve",
    default=1,
)


gf_parser = argparse.ArgumentParser(add_help=False)
gf_parser.add_argument(
    "-H",
    type=int,
    help="Number of generative fields to learn (dictionary size)",
    default=32,
)


decoding_net_parser = argparse.ArgumentParser(add_help=False)
decoding_net_parser.add_argument(
    "--inner_net_shape",
    nargs="+",
    type=int,
    help="Decoder shape (...,H1,H0) excluding number of observables. "
    "Full network shape will be (patch_height*patch_width*no_channels,...,H1,H0)",
    default=[25, 25],
)


experiment_parser = argparse.ArgumentParser(add_help=False)
experiment_parser.add_argument(
    "--no_epochs",
    type=int,
    help="Number of epochs to train",
    default=40,
)


experiment_parser.add_argument(
    "--merge_every",
    type=int,
    help="Generate reconstructed image by merging image patches every Xth epoch (will be set "
    "equal to viz_every if not specified)",
    default=None,
)


output_parser = argparse.ArgumentParser(add_help=False)
output_parser.add_argument(
    "--output_directory",
    type=str,
    help="Directory to write H5 training output and visualizations to (will be output/<TIMESTAMP> "
    "if not specified)",
    default=None,
)


viz_parser = argparse.ArgumentParser(add_help=False)
viz_parser.add_argument(
    "--viz_every",
    type=int,
    help="Create visualizations every Xth epoch.",
    default=1,
)

viz_parser.add_argument(
    "--gif_framerate",
    type=str,
    help="If specified, the training output will be additionally saved as animated gif. The "
    "framerate is given in frames per second. If not specified, no gif will be produced.",
    default=None,
)


def get_args():
    parser = argparse.ArgumentParser(prog="Gaussian Denoising")
    algo_parsers = parser.add_subparsers(help="Select model to train", dest="model", required=True)
    comm_parents = [
        awgn_parser,
        patch_parser,
        variational_parser,
        experiment_parser,
        output_parser,
        viz_parser,
    ]

    algo_parsers.add_parser(
        "bsc",
        help="Run experiment with BSC",
        parents=comm_parents
        + [
            gf_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    algo_parsers.add_parser(
        "sssc",
        help="Run experiment with SSSC",
        parents=comm_parents
        + [
            gf_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    algo_parsers.add_parser(
        "tvae",
        help="Run experiment with TVAE",
        parents=comm_parents
        + [
            decoding_net_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return parser.parse_args()
