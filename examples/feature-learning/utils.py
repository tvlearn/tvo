# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import sys
import h5py
import imageio
import tifffile
import numpy as np
import torch as to
import torch.distributed as dist
from typing import Tuple
from typing import Dict

from tvo import get_run_policy
from tvo.utils.parallel import init_processes as _init_processes

from tvutil.prepost import apply_zca_whitening, extract_random_patches


def init_processes() -> Tuple[int, int]:
    if get_run_policy() == "seq":
        return 0, 1
    else:
        assert get_run_policy() == "mpi"
        _init_processes()
        return dist.get_rank(), dist.get_world_size()


class stdout_logger(object):
    """Redirect print statements both to console and file

    Source: https://stackoverflow.com/a/14906787
    """

    def __init__(self, txt_file):
        self.terminal = sys.stdout
        self.log = open(txt_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def _store_as_h5(to_store_dict: Dict[str, to.Tensor], output_name: str) -> None:
    """Takes dictionary of tensors and writes to H5 file

    :param to_store_dict: Dictionary of torch Tensors
    :param output_name: Full path of H5 file to write data to
    """
    os.makedirs(os.path.split(output_name)[0], exist_ok=True)
    with h5py.File(output_name, "w") as f:
        for key, val in to_store_dict.items():
            f.create_dataset(key, data=val if isinstance(val, float) else val.detach().cpu())
    print(f"Wrote {output_name}")


def prepare_training_dataset(
    image_file: str,
    patch_size: Tuple[int, int],
    no_patches: int,
    h5_file: str,
    perc_highest_amps: float = 0.02,
    perc_lowest_vars: float = None,
) -> to.Tensor:
    """Read image from file, optionally rescale image size and return as to.Tensor

    :param image_file: Full path to image file (.png, .jpg, ...)
    :param patch_size: Patch size as (patch_height, patch_width) tuple
    :param no_patches: Number of patches to extract
    :param h5_file: Full path to H5 file to write data to (directory will be created if not exists)
    :param perc_highest_amps: Percentage of highest image amplitudes to clamp
    :param perc_lowest_vars: Percentage of patches with lowest variance to clamp
    :return: Whitened image patches as torch tensor
    """
    imread = tifffile.imread if os.path.splitext(image_file)[1] == ".tiff" else imageio.imread
    img = imread(image_file)
    print("Read {}".format(image_file))
    isrgb = np.ndim(img) == 3 and img.shape[2] == 3
    isgrey = np.ndim(img) == 2
    assert isrgb or isgrey, "Expect img image to be either RGB or grey"

    # Clamp highest amplitudes
    if perc_highest_amps is not None:
        img = np.clip(
            img, np.min(img), np.sort(img.flatten())[::-1][int(perc_highest_amps * img.size)]
        )

    # Extract image patches and whiten
    patches = extract_random_patches(
        images=img[None, :, :, :] if isrgb else img[None, :, :, None],
        patch_size=patch_size,
        no_patches=no_patches,
    )
    whitened = apply_zca_whitening(patches)

    # Discard patches with lowest variance (assuming these do not contain significant structure)
    if perc_lowest_vars is not None:
        whitened = whitened[
            np.argsort(np.var(whitened, axis=1))[int(perc_lowest_vars * no_patches) :]
        ]

    whitened_to = to.from_numpy(whitened)

    # Save as H5 file
    _store_as_h5({"data": whitened_to}, h5_file)

    return whitened_to
