import os
from datetime import datetime
import argparse
import numpy as np
import torch
import h5py
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import soundfile as sf


if __name__ == "__main__":
    log_path = "./out/plots"

    f_noisy = "./out/24.11.04-12.00.45-infer/noisy-0.1-std.wav"
    f_denoised = "./out/24.11.04-12.00.45-infer/reco-mean-epoch(9)-snr(0.71)-pesq(1.11)-psnr(33.09).wav" 

    noisy, _ = sf.read(f_noisy)
    denoised, _ = sf.read(f_denoised) 

    start = 20500
    n = 2000
    fig = plt.figure(figsize=(10, 3))
    plt.plot(noisy[start:start+n], alpha=0.3, label="noisy")
    plt.savefig(os.path.join(log_path, "noisy-bach.pdf"))
    #plt.close()

    #fig = plt.figure(figsize=(5, 3))
    plt.plot(3*denoised[start:start+n], "k", label="denoised")
    plt.legend(loc="lower right")

    plt.savefig(os.path.join(log_path, "denoised-bach.pdf"))
    plt.close()
    