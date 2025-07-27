import os
from datetime import datetime
import argparse
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec



def load_vector(filename, dataset):
    with h5py.File(filename, "r") as f:
        res = np.array(f[dataset])
    return res



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    #arg_parser.add_argument("filename", metavar="filename", type=str, help="Inference log file, HDF5")
    arg_parser.add_argument("--outdir", type=str, help="Output directory", default=os.path.join("./out/plots"))
    cmd_args = arg_parser.parse_args()
    log_path = cmd_args.outdir
    os.makedirs(log_path, exist_ok=True)


    f_infer_mean = "./out-saved/24.08.29-11.48.05-barbara-mean-infer/inference.h5"
    f_infer_full = "./out-saved/24.08.29-11.53.21-barbara-full-infer/inference.h5"
    f_evo = "../gaussian-denoising/out-saved/24-08-28-15-51-51-barbara-large/training.h5"

    mean_psnr = load_vector(f_infer_mean, "psnr_mean")
    mean_F = load_vector(f_infer_mean, "test_F")[1:]
    
    full_psnr = load_vector(f_infer_full, "psnr_mean")
    full_F = load_vector(f_infer_full, "test_F")[1:]

    evo_psnr = load_vector(f_evo, "psnr_mean")
    evo_F = load_vector(f_evo, "train_F")[1:]
    
    fig = plt.figure(figsize=(5, 4))
    #plt.plot(range(1, len(mean_psnr)+1), mean_psnr, label="mean only")
    plt.plot(range(1, len(full_psnr)+1), full_psnr, label="amortized")
    plt.plot(range(1, len(full_psnr)+1), full_psnr*0+evo_psnr[-1], "--", label="EVO")
    plt.xticks(np.arange(1, 11, 2, dtype=int))
    plt.legend(loc="right")
    plt.title("Amortized inference performance")
    plt.xlabel("Sampling iteration")
    plt.ylabel("PSNR")
    fig.tight_layout()
    plt.savefig(os.path.join(log_path, "barbara-PSNR.pdf"))
    plt.close()
    #plt.show()

    fig = plt.figure(figsize=(5, 4))
    #plt.plot(range(1, len(mean_F)+1), mean_F, label="mean only")
    plt.plot(range(1, len(full_F)+1), full_F, label="amortized")
    plt.plot(range(1, len(full_F)+1), full_F*0+evo_F[-1], "--", label="EVO")
    plt.xticks(np.arange(1, 11, 2, dtype=int))
    plt.legend(loc="right")
    plt.title("Amortized inference ELBO")
    plt.xlabel("Sampling iteration")
    plt.ylabel("ELBO")
    fig.tight_layout()
    plt.savefig(os.path.join(log_path, "barbara-ELBO.pdf"))
    plt.close()
    #plt.show()

    exit()

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

    