# python plot-kset-stats.py --Ksetfile ${TVODATADIR}/training.h5 --N_size 100

import os
from datetime import datetime
import argparse
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
from utils.datasets import TVODataset
from models.amortizedbernoulli import compute_probabilities, sample_mean, batch_sample_covar




def load_Kset(Ksetpath, start=0, maxN=None):
    end = None if maxN is None else start+maxN
    with h5py.File(Ksetpath, "r") as f:
        try:
            Kset = torch.tensor(np.array(f["train_states"])[start:end])
            log_f = torch.tensor(np.array(f["train_lpj"])[start:end], dtype=torch.get_default_dtype())
        except KeyError:
            Kset = torch.tensor(np.array(f["test_states"])[start:end])
            log_f = torch.tensor(np.array(f["test_lpj"])[start:end], dtype=torch.get_default_dtype())
    return Kset, log_f



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("filenames", metavar="filename", type=str, nargs="+",
                help="Kset dataset file, HDF5 (truncated posterior sets file)")
    arg_parser.add_argument("--N_start", type=int, help="Training data slice start", default=0)
    arg_parser.add_argument("--N_size", type=int, help="Training data slice size", default=None)
    arg_parser.add_argument("--outdir", type=str, help="Output directory", default=os.path.join("./out", datetime.now().strftime('%y.%m.%d-%H.%M.%S')+"-plots"))
    cmd_args = arg_parser.parse_args()
    log_path = cmd_args.outdir
    os.makedirs(log_path, exist_ok=True)

    p_mean = {}
    p_covar = {}
    for filename in cmd_args.filenames:
        Kset, log_f = load_Kset(Ksetpath=filename, start=cmd_args.N_start, maxN=cmd_args.N_size)
        p_mean[filename] = sample_mean(Kset, weights=compute_probabilities(log_f), dim=1)
        p_covar[filename] = batch_sample_covar(Kset, weights=compute_probabilities(log_f))
        print("File {}. Loaded Kset shape: {}".format(filename, Kset.shape))

    print("Plotting...")
    M = len(cmd_args.filenames)
    for n in tqdm(range(Kset.shape[0])):
        fig = plt.figure(figsize=(3*(M+1), 3), layout="constrained")
        gs = GridSpec(1, M+1, figure=fig)
        
        ax = fig.add_subplot(gs[0, 0])
        for filename in cmd_args.filenames:
            ax.plot(p_mean[filename][n, ...])
        ax.set_title("Mean values")
        ax.set_aspect(np.diff(ax.get_xlim())[0] / np.diff(ax.get_ylim())[0])
        
        for m, filename in enumerate(cmd_args.filenames):
            ax = fig.add_subplot(gs[0, m+1])
            ax.imshow(p_covar[filename][n, ...].detach().cpu(), vmin=-1, vmax=1, cmap="seismic")
            ax.set_title("Data covariance")
                
        plt.savefig(os.path.join(log_path, "n({}).pdf".format(n)))
        plt.close()

    