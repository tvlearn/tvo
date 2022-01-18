# -*- coding: utf-8 -*-
# Copyright (C) 2022 Machine Learning Group of the University of Oldenburg.

import os
import re
import glob
import math
import numpy as np
import torch as to
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D
from tvutil.viz import make_grid_with_black_boxes_and_white_background


class Visualizer(object):
    def __init__(
        self,
        output_directory,
        viz_every,
        datapoints,
        theta_gen,
        L_gen=None,
        sort_acc_to_desc_priors=True,
        memorize=[
            "F",
        ],
        positions={
            "datapoints": [0.0, 0.0, 0.07, 0.94],
            "W_gen": [0.08, 0.0, 0.1, 0.94],
            "W": [0.2, 0.0, 0.1, 0.94],
            "F": [0.4, 0.6, 0.58, 0.38],
            "pies": [0.4, 0.1, 0.58, 0.38],
        },
        gif_framerate=None,
    ):
        self._output_directory = output_directory
        self._viz_every = viz_every
        self._datapoints = datapoints
        self._theta_gen = theta_gen
        self._L_gen = L_gen
        self._sort_acc_to_desc_priors = sort_acc_to_desc_priors
        self._cmap_weights = plt.cm.jet
        self._cmap_datapoints = plt.cm.gray if datapoints.dtype == to.uint8 else plt.cm.jet
        self._gif_framerate = gif_framerate
        self._labelsize = 10
        self._legendfontsize = 8

        self._memory = {k: [] for k in memorize}
        self._fig = plt.figure()
        self._axes = {k: self._fig.add_axes(v) for k, v in positions.items()}
        self._handles = {k: None for k in positions}
        for k in theta_gen.keys():
            self._handles["{}_gen".format(k)] = None
        self._handles["L_gen"] = None
        self._viz_datapoints()
        self._viz_gen_weights()

    def _viz_datapoints(self):
        assert "datapoints" in self._axes
        ax = self._axes["datapoints"]
        datapoints = self._datapoints
        N, D = datapoints.shape
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=datapoints.numpy().reshape(N, 1, int(math.sqrt(D)), int(math.sqrt(D))),
            nrow=int(math.ceil(N / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=self._cmap_datapoints == plt.cm.jet,
            cmap=self._cmap_datapoints,
            eps=0.02,
        )

        self._handles["datapoints"] = ax.imshow(np.squeeze(grid))
        ax.axis("off")

        self._handles["datapoints"].set_cmap(cmap)
        self._handles["datapoints"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$\vec{y}^{\,(n)}$")

    def _viz_gen_weights(self):
        assert "W_gen" in self._axes
        ax = self._axes["W_gen"]
        W_gen = self._theta_gen["W"]
        D, H = W_gen.shape
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=W_gen.numpy().copy().T.reshape(H, 1, int(math.sqrt(D)), int(math.sqrt(D))),
            nrow=int(math.ceil(H / 16)),
            surrounding=4,
            padding=8,
            repeat=20,
            global_clim=True,
            sym_clim=True,
            cmap=self._cmap_weights,
            eps=0.02,
        )

        if self._handles["W_gen"] is None:
            self._handles["W_gen"] = ax.imshow(np.squeeze(grid))
            ax.axis("off")
        else:
            self._handles["W_gen"].set_data(np.squeeze(grid))
        self._handles["W_gen"].set_cmap(cmap)
        self._handles["W_gen"].set_clim(vmin=vmin, vmax=vmax)
        ax.set_title(r"$W^{\mathrm{gen}}$")

    def _viz_weights(self, epoch, W, inds_sort=None):
        assert "W" in self._axes
        ax = self._axes["W"]
        D, H = W.shape
        W = W.numpy()[:, inds_sort] if inds_sort is not None else W.numpy().copy()
        grid, cmap, vmin, vmax, scale_suff = make_grid_with_black_boxes_and_white_background(
            images=W.T.reshape(H, 1, int(math.sqrt(D)), int(math.sqrt(D))),
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
            self._handles["W"] = ax.imshow(np.squeeze(grid))
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
                label=r"$\mathcal{F}(\mathcal{K},\Theta)$",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            add_legend = True
        else:
            self._handles["F"].set_xdata(xdata)
            self._handles["F"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            add_legend = False

        if self._L_gen is not None:
            ydata_gen = self._L_gen * np.ones_like(ydata_learned)
            if self._handles["L_gen"] is None:
                (self._handles["L_gen"],) = ax.plot(
                    xdata,
                    ydata_gen,
                    "b--",
                    label=r"$\mathcal{L}(\Theta^{\mathrm{gen}})$",
                )
            else:
                self._handles["L_gen"].set_xdata(xdata)
                self._handles["L_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

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
                label=r"$\pi_h$ @ {}".format(epoch),
            )
            ax.set_xlabel(r"$h$", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        else:
            self._handles["pies"].set_xdata(xdata)
            self._handles["pies"].set_ydata(ydata_learned)
            self._handles["pies"].set_label(r"$\pi_h$ @ {}".format(epoch))
            ax.relim()
            ax.autoscale_view()

        ydata_gen = self._theta_gen["pies"]
        xdata = to.arange(1, len(ydata_gen) + 1)
        if self._handles["pies_gen"] is None:
            (self._handles["pies_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b",
                linestyle="none",
                marker="o",
                fillstyle=Line2D.fillStyles[-1],
                markersize=4,
                label=r"$\pi_h^{\mathrm{gen}}$",
            )
        else:
            self._handles["pies_gen"].set_xdata(xdata)
            self._handles["pies_gen"].set_ydata(ydata_gen)

        ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_epoch(self, epoch, F, theta):
        try:
            pies = theta["pies"]
        except KeyError:
            pies = theta["pi"]  # NOR currently uses pi as variable name for priors
        inds_sort = to.argsort(pies, descending=True) if self._sort_acc_to_desc_priors else None
        self._viz_weights(epoch, theta["W"])
        self._viz_pies(epoch, pies, inds_sort=inds_sort)
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

    def _write_gif(self, framerate):
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


class BSCVisualizer(Visualizer):
    def __init__(self, **kwargs):
        super(BSCVisualizer, self).__init__(
            memorize=("F", "sigma2"),
            positions={
                "datapoints": [0.0, 0.0, 0.07, 0.94],
                "W_gen": [0.08, 0.0, 0.1, 0.94],
                "W": [0.2, 0.0, 0.1, 0.94],
                "F": [0.4, 0.76, 0.58, 0.23],
                "sigma2": [0.4, 0.43, 0.58, 0.23],
                "pies": [0.4, 0.1, 0.58, 0.23],
            },
            **kwargs
        )

    def _viz_sigma2(self):
        memory = self._memory
        assert "sigma2" in memory
        assert "sigma2" in self._axes
        ax = self._axes["sigma2"]
        xdata = to.arange(1, len(memory["sigma2"]) + 1)
        ydata_learned = np.squeeze(to.tensor(memory["sigma2"]))
        if self._handles["sigma2"] is None:
            (self._handles["sigma2"],) = ax.plot(
                xdata,
                ydata_learned,
                "b",
                label=r"$\sigma^2$",
            )
            ax.set_xlabel("Epoch", fontsize=self._labelsize)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            add_legend = True
        else:
            self._handles["sigma2"].set_xdata(xdata)
            self._handles["sigma2"].set_ydata(ydata_learned)
            ax.relim()
            ax.autoscale_view()
            add_legend = True

        ydata_gen = self._theta_gen["sigma2"] * np.ones_like(ydata_learned)
        if self._handles["sigma2_gen"] is None:
            (self._handles["sigma2_gen"],) = ax.plot(
                xdata,
                ydata_gen,
                "b--",
                label=r"$(\sigma^{\mathrm{gen}})^2$",
            )
        else:
            self._handles["sigma2_gen"].set_xdata(xdata)
            self._handles["sigma2_gen"].set_ydata(ydata_gen)

        if add_legend:
            ax.legend(prop={"size": self._legendfontsize}, ncol=2)

    def _viz_epoch(self, epoch, F, theta):
        super(BSCVisualizer, self)._viz_epoch(epoch, F, theta)
        self._viz_sigma2()
