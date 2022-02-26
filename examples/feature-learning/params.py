# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import argparse


def get_args():
    p = argparse.ArgumentParser(
        description="Train EBSC on whitened image patches",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--image_file",
        type=str,
        help="Full path to image file (.png, .jpg, ...) used to extract training patches",
        default="./data/barbara.png",
    )

    p.add_argument(
        "--patch_size",
        type=int,
        nargs=2,
        help="Patch size, (height, width) tuple",
        default=(8, 8),
    )

    p.add_argument(
        "--no_patches",
        type=int,
        help="Number of image patches to extract for training",
        default=1000,
    )

    p.add_argument(
        "--output_directory",
        type=str,
        help="Directory to write training output and visualizations to (will be output/<TIMESTAMP> "
        "if not specified)",
        default=None,
    )

    p.add_argument(
        "-H",
        type=int,
        help="Number of generative fields to learn",
        default=20,
    )

    p.add_argument(
        "--Ksize",
        type=int,
        help="Size of the K sets (i.e., S=|K|)",
        default=10,
    )

    p.add_argument(
        "--selection",
        type=str,
        help="Selection operator",
        choices=["fitness", "uniform"],
        default="fitness",
    )

    p.add_argument(
        "--crossover",
        action="store_true",
        help="Whether to apply crossover. Must be False if no_children is specified",
        default=False,
    )

    p.add_argument(
        "--no_parents",
        type=int,
        help="Number of parental states to select per generation",
        default=5,
    )

    p.add_argument(
        "--no_children",
        type=int,
        help="Number of children to evolve per generation",
        default=3,
    )

    p.add_argument(
        "--no_generations",
        type=int,
        help="Number of generations to evolve",
        default=2,
    )

    p.add_argument(
        "--no_epochs",
        type=int,
        help="Number of epochs to train",
        default=40,
    )

    p.add_argument(
        "--viz_every",
        type=int,
        help="Create visualizations every Xth epoch. Set to no_epochs if not specified.",
        default=1,
    )

    return p.parse_args()
