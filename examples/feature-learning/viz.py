# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import os
import re
import glob
import math
import numpy as np
import torch as to
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from typing import Tuple, Optional, List, Dict
from tvutil.viz import make_grid_with_black_boxes_and_white_background


class Visualizer(object):
    def __init__(
        self,
        output_directory: str,
        viz_every: int,
        datapoints: to.Tensor,
        patch_size: Tuple[int, int],
        ncol_gfs: int = 16,
        sort_acc_to_desc_priors: bool = True,
        topk_gfs: bool = None,
        positions: Dict[str, List] = {
            "datapoints": [0.0, 0.0, 0.07, 0.94],
            "gfs": [0.08, 0.00, 0.40, 0.94],
            "F": [0.6, 0.6, 0.38, 0.38],
            "pies": [0.6, 0.1, 0.38, 0.38],
        },
        global_clims: bool = False,
        gif_framerate: Optional[str] = None,
    ):
        self._gif_framerate = gif_framerate
        if gif_framerate is not None and viz_every > 1:
            print("Choose --viz_every=1 for best gif results")
        self._output_directory = output_directory
        self._viz_every = viz_every
        self._datapoints = datapoints
        self._patch_size = patch_size
        self._ncol_gfs = ncol_gfs
        self._sort_acc_to_desc_priors = sort_acc_to_desc_priors
        self._topk_gfs = topk_gfs
        self._cmap = plt.cm.jet
        self._global_clims = global_clims
        self._labelsize = 10
        self._legendfontsize = 8

        D = datapoints.shape[1]
        patch_height, patch_width = self._patch_size
        self._no_channels = int(D / patch_height / patch_width)

        self._memory = {"F": []}
        self._fig = plt.figure()
        self._axes = {k: self._fig.add_axes(v) for k, v in positions.items()}
        self._handles = {k: None for k in positions}
        self._viz_datapoints()

    def _viz_datapoints(self):
        assert "datapoints" in self._axes
        ax = self._axes["datapoints"]
        datapoints = self._datapoints
        N = datapoints.shape[0]
        patch_height, patch_width = self._patch_size
        no_channels = self._no_channels
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=datapoints.numpy().reshape(N, no_channels, patch_height, patch_width),
            nrow=int(math.ceil(N / self._ncol_gfs)),
            surrounding=0,
            padding=8,
            repeat=20,
            global_clim=self._global_clims,
            sym_clim=True,
            cmap=self._cmap,
            eps=0.01,
        )

        self._handles["datapoints"] = ax.imshow(np.squeeze(grid), interpolation="none")
        ax.axis("off")

        self._handles["datapoints"].set_cmap(cmap)
        self._handles["datapoints"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$\vec{y}^{\,(n)}$")

    def _viz_weights(self, epoch: int, gfs: np.ndarray, suffix: str = ""):
        assert "gfs" in self._axes
        ax = self._axes["gfs"]
        H = gfs.shape[1]
        patch_height, patch_width = self._patch_size
        no_channels = self._no_channels
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=gfs.T.reshape(H, no_channels, patch_height, patch_width),
            nrow=int(np.ceil(H / self._ncol_gfs)),
            surrounding=2,
            padding=4,
            repeat=10,
            global_clim=False,
            sym_clim=False,
            cmap=self._cmap,
            eps=0.02,
        )

        gfs = grid.transpose(1, 2, 0) if no_channels > 1 else np.squeeze(grid)
        if self._handles["gfs"] is None:
            self._handles["gfs"] = ax.imshow(gfs, interpolation="none")
            ax.axis("off")
        else:
            self._handles["gfs"].set_data(gfs)
        self._handles["gfs"].set_cmap(cmap)
        self._handles["gfs"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title("GFs @ {}".format(epoch) + (" ({})".format(suffix) if suffix else ""))

    def _viz_free_energy(self):
        memory = self._memory
        assert "F" in memory
        assert "F" in self._axes
        ax = self._axes["F"]
        xdata = to.arange(1, len(memory["F"]) + 1)
        ydata_learned = memory["F"]
        if self._handles["F"] is None:
            (self._handles["F"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.set_ylabel(r"$\mathcal{F}(\mathcal{K},\Theta)$", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self._handles["F"].set_xdata(xdata)
            self._handles["F"].set_ydata(ydata_learned)
            ax.set_ylabel(r"$\mathcal{F}(\mathcal{K},\Theta)$", fontsize=self._labelsize)
            ax.relim()
            ax.autoscale_view()

    def _viz_pies(self, epoch: int, pies: np.ndarray, suffix: str = ""):
        assert "pies" in self._axes
        ax = self._axes["pies"]
        xdata = np.arange(1, len(pies) + 1)
        ydata = pies
        if self._handles["pies"] is None:
            (self._handles["pies"],) = ax.plot(
                xdata,
                ydata,
                "b",
                linestyle="none",
                marker=".",
                markersize=4,
            )
            ax.set_ylabel(
                r"$\pi_h$ @ {}".format(epoch) + ("\n" + suffix) if suffix else "",
                fontsize=self._labelsize,
            )
            ax.set_xlabel(r"$h$", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            # ax.tick_params(axis="x", labelrotation=30)
            self._handles["pies_summed"] = ax.text(
                0.81,
                0.85,
                r"$\sum_h \pi_h$ = " + "{:.2f}".format(pies.sum()),
                horizontalalignment="center",
                verticalalignment="center",
                transform=ax.transAxes,
            )
        else:
            self._handles["pies"].set_xdata(xdata)
            self._handles["pies"].set_ydata(ydata)
            ax.set_ylabel(r"$\pi_h$ @ {}".format(epoch) + ("\n" + suffix) if suffix else "")
            ax.relim()
            ax.autoscale_view()
            self._handles["pies_summed"].set_text(
                r"$\sum_h \pi_h$ = " + "{:.2f}".format(pies.sum())
            )

    def viz_epoch(self, epoch: int, pies: to.Tensor, gfs: to.Tensor):
        pies = pies.detach().cpu().numpy()
        gfs = gfs.detach().cpu().numpy()
        inds_sort = (
            np.argsort(pies)[::-1] if self._sort_acc_to_desc_priors else np.arange(len(pies))
        )
        inds_sort_gfs = inds_sort[: self._topk_gfs] if self._topk_gfs is not None else inds_sort
        suffix_gfs = (
            (
                "sorted, top {}".format(self._topk_gfs)
                if self._sort_acc_to_desc_priors
                else "top {}".format(self._topk_gfs)
            )
            if self._topk_gfs
            else ("sorted" if self._sort_acc_to_desc_priors else "")
        )
        self._viz_weights(epoch, gfs.copy()[:, inds_sort_gfs], suffix_gfs)
        self._viz_pies(epoch, pies[inds_sort], "(sorted)" if self._sort_acc_to_desc_priors else "")
        self._viz_free_energy()

    def process_epoch(self, epoch: int, F: float, pies: to.Tensor, gfs: to.Tensor):
        memory = self._memory
        memory["F"].append(F)
        if epoch % self._viz_every == 0:
            self.viz_epoch(epoch, pies, gfs)
            self.save_epoch(epoch)

    def save_epoch(self, epoch: int):
        output_directory = self._output_directory
        png_file = "{}/training{}.png".format(
            output_directory,
            "_epoch{:04d}".format(epoch) if self._gif_framerate is not None else "",
        )
        plt.savefig(png_file)
        print("\tWrote " + png_file)

    def _write_gif(self, framerate: str):
        output_directory = self._output_directory
        gif_file = "{}/training.gif".format(output_directory)
        print("Creating {} ...".format(gif_file), end="")
        # work-around for correct color display from https://stackoverflow.com/a/58832086
        os.system(
            "ffmpeg -y -framerate {} -i {}/training_epoch%*.png -vf palettegen \
            {}/palette.png".format(
                framerate, output_directory, output_directory
            )
        )
        os.system(
            "ffmpeg -y -framerate {} -i {}/training_epoch%*.png -i {}/palette.png -lavfi \
            paletteuse {}/training.gif".format(
                framerate, output_directory, output_directory, output_directory
            )
        )
        print("Done")

        png_files = glob.glob("{}/training_epoch*.png".format(output_directory))
        png_files.sort(
            key=lambda var: [
                int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
            ]
        )
        last_epoch_str = "_epoch{}".format(png_files[-1].split("_epoch")[1].replace(".png", ""))
        for f in png_files:
            if last_epoch_str in f:
                old = f
                new = f.replace(last_epoch_str, "_last_epoch")
                os.rename(old, new)
                print("Renamed {}->{}".format(old, new))
            else:  # keep png of last epoch
                os.remove(f)
                print("Removed {}".format(f))
        os.remove("{}/palette.png".format(output_directory))

    def finalize(self):
        plt.close()
        if self._gif_framerate is not None:
            self._write_gif(framerate=self._gif_framerate)
