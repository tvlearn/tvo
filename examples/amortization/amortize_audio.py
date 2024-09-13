# Load the data and corresponding trained K-sets, 
# train an amortizing pytorch module mapping, 
# save the mapping as a "*.pt" file.

# python amortize.py \
# --Xfile ../../../relaxed-bernoulli-datasets/House12/sigma25/image_patches.h5 \
# --Ksetfile ../../../relaxed-bernoulli-datasets/House12/sigma25/training.h5 \
# --N_start 10030 \
# --N_size 10

#python amortize.py \
# --Xfile ../gaussian-denoising/out/24-07-31-16-48-59/image_patches.h5 \
# --Ksetfile ../gaussian-denoising/out/24-07-31-16-48-59/training.h5 

import os
import sys
from datetime import datetime
import numpy as np
import torch
import argparse
from enum import Enum
from utils.datasets import ToyDatasetH2Minimal, LargeCorrelatedDataset, TVODataset
from models.amortizedbernoulli import AmortizedBernoulli, compute_probabilities, Objective, binarize
from models.variationalparams import (
    FullCovarGaussianVariationalParams, 
    AmortizedGaussianVariationalParams,
    AmortizedResNetVariationalParams, 
    AmortizedResNetLowRankVariationalParams)
from utils.common import FloatPrecision
from utils.training import train
from utils.plotting import plot_epoch_log
from utils.utils import stdout_logger


if False:
    print("------------------------------------------------")
    print("WARNING! torch.autograd.set_detect_anomaly(True)")
    print("------------------------------------------------")
    torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--Xfile", type=str, help="X dataset file, HDF5", default=None)
    arg_parser.add_argument("--Ksetfile", type=str, help="Kset dataset file, HDF5 (truncated posterior sets file)", default=None)
    arg_parser.add_argument("--N_start", type=int, help="Training data slice start", default=0)
    arg_parser.add_argument("--N_size", type=int, help="Training data slice size", default=None)
    arg_parser.add_argument("--epochs_mean", type=int, help="Number of epochs to train only mean activation", default=10)
    arg_parser.add_argument("--epochs_full", type=int, help="Number of epochs to train full model", default=10)
    arg_parser.add_argument("--batch_size", type=int, help="Training batch size", default=32)
    arg_parser.add_argument("--N_IS", type=int, help="Number of samples for importance sampler estimator", default=128)
    arg_parser.add_argument("--lr_mean", type=float, help="Learning rate for mean optimization", default=1e-3)
    arg_parser.add_argument("--lr_full", type=float, help="Learning rate for full optimization", default=1e-3)
    arg_parser.add_argument("--t_start", type=float, help="Learning start temperature", default=2.0)
    arg_parser.add_argument("--t_end", type=float, help="Learning end temperature", default=2.0)
    arg_parser.add_argument("--CPU", help="Force CPU compute", action="store_true")
    arg_parser.add_argument("--precision", type=FloatPrecision, help="Compute precision", default=FloatPrecision.float32)
    arg_parser.add_argument("--outdir", type=str, help="Output directory", default=os.path.join("./out", datetime.now().strftime('%y.%m.%d-%H:%M:%S')+"-amortize"))

    cmd_args = arg_parser.parse_args()
    log_path = cmd_args.outdir
    os.makedirs(log_path, exist_ok=True)
    sys.stdout = stdout_logger(os.path.join(log_path, "terminal.txt"))
    print("Parameters:")
    for var in vars(cmd_args):
        print("  {}: {}". format(var, vars(cmd_args)[var]))

    assert cmd_args.Xfile is not None
    assert cmd_args.Ksetfile is not None

    torch.set_default_dtype(cmd_args.precision.torch_dtype())
    device = torch.device("cuda" if torch.cuda.is_available() and not cmd_args.CPU else "cpu") 
    #torch.set_default_device(device)
    print("Using PyTorch device/precision: {}/{}".format(device, torch.get_default_dtype()))
    
    dataset = TVODataset(Xpath=cmd_args.Xfile, Ksetpath=cmd_args.Ksetfile, start=cmd_args.N_start, maxN=cmd_args.N_size)
    print("Loaded Kset shape: ", dataset.Kset.shape)
    dataset.to(device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=cmd_args.batch_size, shuffle=True)
    N, D = dataset.X.shape
    H = dataset.Kset.shape[-1]

    model = AmortizedBernoulli(nsamples=cmd_args.N_IS, 
                               #variationalparams=FullCovarGaussianVariationalParams(N, D, H)
                               variationalparams=AmortizedResNetLowRankVariationalParams(N, D, H)
                               ).to(device)
    model.train()
   
    def on_epoch(X, Kset, logPs, mean_loss, res):
        return 
        if (epochs_done + epoch) % 1 == 0:
            plot_epoch_log(X, Kset, logPs, mean_loss, res, epochs_done + epoch, log_path)

    # Optimize mean only
    temperature = np.concatenate([np.linspace(cmd_args.t_start, cmd_args.t_end, cmd_args.epochs_mean)])
    model.objective_type = Objective.MEANKLDIVERGENCE
    optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.lr_mean)
    loss = []
    epochs_done = 0
    for epoch in range(temperature.size):
        model.temperature = temperature[epoch]
        epochloss = train(model, dataloader, optimizer, on_epoch)
        loss.append(np.array(epochloss).mean())
        print("Optimizing mean | Epoch: {:4d} | t: {:9.4f} | Loss: {:9.4f}".format(epoch, model.temperature, loss[-1]))

    torch.save(model, os.path.join(log_path, "trained_sampler_mean.pt"))
    
    # Optimize full model (mean and covariance parameters)
    temperature = np.concatenate([np.linspace(cmd_args.t_start, cmd_args.t_end, cmd_args.epochs_full)])
    model.objective_type = Objective.KLDIVERGENCE
    optimizer = torch.optim.Adam(model.parameters(), lr=cmd_args.lr_full)
    loss = []
    epochs_done = cmd_args.epochs_mean
    for epoch in range(temperature.size):
        model.temperature = temperature[epoch]
        epochloss = train(model, dataloader, optimizer, on_epoch)
        loss.append(np.array(epochloss).mean())
        print("Optimizing all  | Epoch: {:4d} | t: {:9.4f} | Loss: {:9.4f}".format(cmd_args.epochs_mean + epoch, model.temperature, loss[-1]))

    torch.save(model, os.path.join(log_path, "trained_sampler.pt"))
    
    
    #from torch.profiler import profile, record_function, ProfilerActivity
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by="self_cuda_time_total", row_limit=30, max_name_column_width=200))