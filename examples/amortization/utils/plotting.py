import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from models.amortizedbernoulli import compute_probabilities, binarize


def plot_epoch_log(X, Kset, logPs, mean_loss, res, epoch, log_path):
    for n in range(min(10, X.shape[0])):
    
        fig = plt.figure(figsize=(10, 7), layout="constrained")
        gs = GridSpec(5, 3, figure=fig)
     
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(res["p_mean"][n, ...].detach().cpu())
        ax.plot(res["q_mean"][n, ...].detach().cpu())
        ax.set_title("Mean values")
        ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
        
        ax = fig.add_subplot(gs[0, 1])
        ax.imshow(res["p_covar"][n, ...].detach().cpu(), vmin=-1, vmax=1, cmap="seismic")
        ax.set_title("Data covariance")
        
        ax = fig.add_subplot(gs[0, 2])
        ax.imshow(res["q_covar"][n, ...].detach().cpu(), vmin=-1, vmax=1, cmap="seismic")
        ax.set_title("Learned covariance")
        fig.suptitle("Loss: {}".format(mean_loss))
        
        ax = fig.add_subplot(gs[1, :])
        ax.imshow(torch.hstack([
            (compute_probabilities(logPs[n]) * Kset[n].T).detach().cpu(), 
            (res["q_samples"][:, n, :].detach().cpu().T)]))
        
        ax = fig.add_subplot(gs[2, :])
        ax.imshow(torch.hstack([
            Kset[n].T.detach().cpu(), 
            binarize(res["q_samples"][:, n, :].detach().cpu().T)]))

        ax = fig.add_subplot(gs[3, :])
        ax.imshow(res["p_samples"][:, n, :].detach().cpu().T, cmap="binary")
        ax.set_title("Data samples")

        ax = fig.add_subplot(gs[4, :])
        ax.imshow(binarize(res["q_samples"][:, n, :].detach().cpu().T), cmap="binary")
        ax.set_title("Learned samples")
        
        plt.savefig(os.path.join(log_path, "n({})-epoch({}).pdf".format(n, epoch)))
        plt.close()
        
    