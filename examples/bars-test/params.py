# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse


output_parser = argparse.ArgumentParser(add_help=False)
output_parser.add_argument(
    "--output_directory",
    type=str,
    help="Directory to write training output and visualizations to (will be output/<TIMESTAMP> "
    "if not specified)",
    default=None,
)

bars_parser = argparse.ArgumentParser(add_help=False)
bars_parser.add_argument(
    "-H_gen",
    type=int,
    help="Number of bars used to generate data",
    default=8,
)

bars_parser.add_argument(
    "--bar_amp",
    type=float,
    help="Bar amplitude",
    default=1.0,
)

bars_parser.add_argument(
    "--no_data_points",
    type=int,
    help="Number of datapoints",
    default=500,
)

nor_parser = argparse.ArgumentParser(add_help=False)
nor_parser.add_argument(
    "--pi_gen",
    type=float,
    help="Sparsity used for data generation (defaults to 2/H if not specified)",
    default=None,
)

nor_parser.add_argument(
    "-H",
    type=int,
    help="Number of generative fields to learn (set to H_gen if not specified)",
    default=None,
)

bsc_parser = argparse.ArgumentParser(add_help=False)
bsc_parser.add_argument(
    "--pi_gen",
    type=float,
    help="Sparsity used for data generation (defaults to 2/H if not specified)",
    default=None,
)

bsc_parser.add_argument(
    "--sigma2_gen",
    type=float,
    help="Noise level used for data generation",
    default=0.01,
)

bsc_parser.add_argument(
    "-H",
    type=int,
    help="Number of generative fields to learn (set to H_gen if not specified)",
    default=None,
)

pmca_parser = argparse.ArgumentParser(add_help=False)
pmca_parser.add_argument(
    "--pi_gen",
    type=float,
    help="Sparsity used for data generation (defaults to 2/H if not specified)",
    default=None,
)

pmca_parser.add_argument(
    "-H",
    type=int,
    help="Number of generative fields to learn (set to H_gen if not specified)",
    default=None,
)

sssc_parser = argparse.ArgumentParser(add_help=False)
sssc_parser.add_argument(
    "--pi_gen",
    type=float,
    help="Sparsity used for data generation (defaults to 2/H if not specified)",
    default=None,
)

sssc_parser.add_argument(
    "--sigma2_gen",
    type=float,
    help="Noise level used for data generation",
    default=0.01,
)

sssc_parser.add_argument(
    "--mu_gen",
    type=float,
    help="Latent means used for data generation",
    default=0.0,
)

sssc_parser.add_argument(
    "--Psi_gen",
    type=float,
    help="Latent variances used for data generation",
    default=1.0,
)

sssc_parser.add_argument(
    "-H",
    type=int,
    help="Number of generative fields to learn (set to H_gen if not specified)",
    default=None,
)

variational_parser = argparse.ArgumentParser(add_help=False)
variational_parser.add_argument(
    "--Ksize",
    type=int,
    help="Size of the K sets (i.e., S=|K|)",
    default=10,
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
    default=5,
)

variational_parser.add_argument(
    "--no_children",
    type=int,
    help="Number of children to evolve per generation",
    default=3,
)

variational_parser.add_argument(
    "--no_generations",
    type=int,
    help="Number of generations to evolve",
    default=2,
)


experiment_parser = argparse.ArgumentParser(add_help=False)
experiment_parser.add_argument(
    "--no_epochs",
    type=int,
    help="Number of epochs to train",
    default=40,
)


viz_parser = argparse.ArgumentParser(add_help=False)
viz_parser.add_argument(
    "--viz_every",
    type=int,
    help="Create visualizations every Xth epoch. Set to no_epochs if not specified.",
    default=1,
)

viz_parser.add_argument(
    "--gif_framerate",
    type=str,
    help="Frames per second for gif animation (e.g., 2/1 for 2 fps). If not specified, no gif will "
    "be produced.",
    default=None,
)


def get_args():
    parser = argparse.ArgumentParser(prog="Standard Bars Test")
    algo_parsers = parser.add_subparsers(help="Select model to train", dest="model", required=True)
    comm_parents = [output_parser, bars_parser, variational_parser, experiment_parser, viz_parser]
    algo_parsers.add_parser(
        "nor",
        help="Run experiment with Noisy-OR",
        parents=comm_parents
        + [
            nor_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    algo_parsers.add_parser(
        "bsc",
        help="Run experiment with BSC",
        parents=comm_parents
        + [
            bsc_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    algo_parsers.add_parser(
        "pmca",
        help="Run experiment with PMCA",
        parents=comm_parents
        + [
            pmca_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    algo_parsers.add_parser(
        "sssc",
        help="Run experiment with SSSC",
        parents=comm_parents
        + [
            sssc_parser,
        ],
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    return parser.parse_args()
