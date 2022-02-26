# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.
# Licensed under the Academic Free License version 3.0

import math
import numpy as np
import torch as to
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tvutil.viz import make_grid_with_black_boxes_and_white_background


class Visualizer(object):
    def __init__(
        self,
        output_directory,
        viz_every,
        datapoints,
        patch_size,
        sort_acc_to_desc_priors=True,
        positions={
            "datapoints": [0.0, 0.0, 0.07, 0.94],
            "W": [0.2, 0.0, 0.1, 0.94],
            "F": [0.4, 0.6, 0.58, 0.38],
            "pies": [0.4, 0.1, 0.58, 0.38],
        },
    ):
        self._output_directory = output_directory
        self._viz_every = viz_every
        self._datapoints = datapoints
        self._patch_size = patch_size
        self._sort_acc_to_desc_priors = sort_acc_to_desc_priors
        self._cmap_weights = self._cmap_datapoints = plt.cm.jet
        self._labelsize = 10
        self._legendfontsize = 8

        self._memory = {"F": []}
        self._fig = plt.figure()
        self._axes = {k: self._fig.add_axes(v) for k, v in positions.items()}
        self._handles = {k: None for k in positions}
        self._viz_datapoints()

    def _viz_datapoints(self):
        assert "datapoints" in self._axes
        ax = self._axes["datapoints"]
        datapoints = self._datapoints
        N, D = datapoints.shape
        patch_height, patch_width = self._patch_size
        no_channels = D / patch_height / patch_width
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=datapoints.numpy().reshape(N, no_channels, patch_height, patch_width),
            nrow=int(math.ceil(N / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_datapoints,
            eps=0.02,
        )

        self._handles["datapoints"] = ax.imshow(np.squeeze(grid), interpolation="none")
        ax.axis("off")

        self._handles["datapoints"].set_cmap(cmap)
        self._handles["datapoints"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$\vec{y}^{\,(n)}$")

    def _viz_weights(self, epoch, W, inds_sort=None):
        assert "W" in self._axes
        ax = self._axes["W"]
        D, H = W.shape
        patch_height, patch_width = self._patch_size
        no_channels = D / patch_height / patch_width
        W = W.numpy()[:, inds_sort] if inds_sort is not None else W.numpy().copy()
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=W.T.reshape(H, no_channels, patch_height, patch_width),
            nrow=int(math.ceil(H / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_weights,
            eps=0.02,
        )

        if self._handles["W"] is None:
            self._handles["W"] = ax.imshow(np.squeeze(grid), interpolation="none")
            ax.axis("off")
        else:
            self._handles["W"].set_data(np.squeeze(grid))
        self._handles["W"].set_cmap(cmap)
        self._handles["W"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title("W @ {}".format(epoch))

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

    def _viz_pies(self, epoch, pies, inds_sort=None):
        assert "pies" in self._axes
        ax = self._axes["pies"]
        xdata = to.arange(1, len(pies) + 1)
        ydata_learned = pies[inds_sort] if inds_sort is not None else pies
        if self._handles["pies"] is None:
            (self._handles["pies"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                linestyle="none",
                marker=".",
                markersize=4,
            )
            ax.set_xlabel(r"$h$", fontsize=self._labelsize)
            ax.set_ylabel(r"$\pi_h$ @ {}".format(epoch), fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self._handles["pies"].set_xdata(xdata)
            self._handles["pies"].set_ydata(ydata_learned)
            ax.set_ylabel(r"$\pi_h$ @ {}".format(epoch), fontsize=self._labelsize)
            ax.relim()
            ax.autoscale_view()

    def _viz_epoch(self, epoch, F, theta):
        inds_sort = (
            to.argsort(theta["pies"], descending=True) if self._sort_acc_to_desc_priors else None
        )
        self._viz_weights(epoch, theta["W"])
        self._viz_pies(epoch, theta["pies"], inds_sort=inds_sort)
        self._viz_free_energy()

    def process_epoch(self, epoch, F, theta):
        memory = self._memory
        [
            v.append(np.squeeze({"F": F, **{k_: v_.clone() for k_, v_ in theta.items()}}[k]))
            for k, v in memory.items()
        ]
        if epoch % self._viz_every == 0:
            self._viz_epoch(epoch, F, theta)
            self._save_epoch(epoch)

    def _save_epoch(self, epoch):
        output_directory = self._output_directory
        png_file = "{}/training{}.png".format(
            output_directory,
            "_epoch{:04d}".format(epoch) if self._gif_framerate is not None else "",
        )
        plt.savefig(png_file)
        print("\tWrote " + png_file)
