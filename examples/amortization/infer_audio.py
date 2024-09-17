# Perform amortized TVO inference and store ELBO history w.r.t. samples.

# python infer.py --model ../gaussiandenoising/out/24-07-31-16-48-59/training.h5 
# --sampler ./out/24.08.02-15:06:10/trained_sampler.pt 
# --Xfile ../gaussiandenoising/out/24-07-31-16-48-59/image_patches.h5
# --epochs 10 --CPU


import os
import sys
from datetime import datetime
import numpy as np
import torch
import argparse
import h5py
import tvo
from tvo.models import GaussianTVAE
from tvo.exp import EVOConfig, AmortizedSamplingConfig, ExpConfig, Training
from tvo.utils.parallel import pprint, broadcast, barrier, bcast_shape, gather_from_processes
from tvutil.prepost import OverlappingPatches, MultiDimOverlappingPatches, mean_merger, median_merger
from utils.common import FloatPrecision
from utils.viz import Visualizer
from utils.utils_audio import eval_fn, store_as_h5, stdout_logger
from models.amortizedbernoulli import SamplerModule
import librosa
import soundfile as sf


def load_group_as_dict(hdf5filename, groupname):
    res = {}
    with h5py.File(hdf5filename, "r") as f:
        group = f[groupname]
        for k in group.keys():
            v = group[k]
            if v.shape == ():
                res[k] = v[()]
            elif v.shape == (1,):
                res[k] = v[:] 
            else:
                res[k] = v[:]
    return res
    

def load_var(hdf5filename, key):
    with h5py.File(hdf5filename, "r") as f:
        res = torch.tensor(np.array(f[key]))
    res = res[0] if len(res) == 1 else res
    return res


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(add_help=False)
    arg_parser.add_argument("--model", type=str, help="BSC model parameters file, *.HDF5", default=None)
    arg_parser.add_argument("--sampler", type=str, help="Posterior sampler file, *.pt", default=None)
    arg_parser.add_argument("--clean-test-file", type=str, help="X dataset file, .wav", default=None) #VB
    arg_parser.add_argument("--epochs", type=int, help="Number of epochs to run the sampler", default=10)
    arg_parser.add_argument("--reco-epochs", type=int, help="Save reconstructions every", default=1) #VB
    arg_parser.add_argument("--batch", type=int, help="Batch size", default=128)
    arg_parser.add_argument("--N_samples", type=int, help="Number of samples to draw", default=100)
    arg_parser.add_argument("--CPU", action="store_true")
    arg_parser.add_argument("--precision", type=FloatPrecision, help="Compute precision", default=FloatPrecision.float32)
    arg_parser.add_argument("--outdir", type=str, help="Output directory", default=os.path.join("./out", datetime.now().strftime('%y.%m.%d-%H.%M.%S')+"-infer"))

    cmd_args = arg_parser.parse_args()
    log_path = cmd_args.outdir
    os.makedirs(log_path, exist_ok=True)
    sys.stdout = stdout_logger(os.path.join(log_path, "terminal.txt"))
    print("Parameters:")
    for var in vars(cmd_args):
        print("  {}: {}". format(var, vars(cmd_args)[var]))

    assert cmd_args.model is not None
    assert cmd_args.sampler is not None
    assert cmd_args.clean_test_file is not None #VB
    
    eval("torch.set_default_dtype(torch.{})".format(cmd_args.precision))
    device = torch.device("cuda" if torch.cuda.is_available() and not cmd_args.CPU else "cpu") 
    print("Using PyTorch device/precision: {}/{}".format(device, torch.get_default_dtype()))
    tvo._set_device(device)

    # Load BSC model parameters
    theta = load_group_as_dict(cmd_args.model, "theta")
    model_config_dict = load_group_as_dict(cmd_args.model, "model_config")
    estep_config_dict = load_group_as_dict(cmd_args.model, "estep_config")
    exp_config_dict = load_group_as_dict(cmd_args.model, "exp_config")

    # Load posterior sampler object from a *.pt file
    sampler = torch.load(cmd_args.sampler, map_location=device)
    assert isinstance(sampler, SamplerModule)
    sampler.eval()

    # Load clean audio test file, create noisy audio and extract audio chunks
    clean_test, sr = librosa.load(cmd_args.clean_test_file, sr=16000)
    clean_test = torch.tensor(clean_test).to(device)
    sigma = 0.1
    noisy_test = clean_test + sigma * torch.randn_like(clean_test)

    # Save noisy test file 
    wav_file = f"{cmd_args.outdir}/noisy-{sigma}-std.wav"
    noisy_wav = noisy_test.detach().cpu().numpy()
    sf.write(wav_file, noisy_wav, sr)

    patch_width = 400
    patch_height = 1

    OVP = OverlappingPatches
    ovp = OVP(noisy_test[None, ...], patch_height, patch_width, patch_shift=1) #VB

    # Test chunks are saved as an h5 file
    data_file, training_file = (
        cmd_args.outdir + "/image_patches.h5",
        cmd_args.outdir + "/training.h5",
    )

    test_data = ovp.get().t()
    store_as_h5({"data": test_data}, data_file)

    D = patch_height * patch_width # calculated D

    with h5py.File(data_file, "r") as f:
        data = torch.tensor(f["data"][...])
    N, D_read = data.shape
    assert D == D_read # check if D calculated corresponds to D read in from the chunks
    del data 

    epochs_per_half_cycle = 20

    # Construct TVAE model
    model = GaussianTVAE(
        min_lr=0.0001,
        max_lr=0.001,
        cycliclr_step_size_up=np.ceil(N / exp_config_dict["batch_size"]) * epochs_per_half_cycle,
        W_init = [torch.Tensor(theta[key]) for key in theta.keys() if 'W_' in key and key[2].isnumeric()], #if weights are loaded, no shape parameter must be given
        b_init=[torch.Tensor(theta[key]) for key in theta.keys() if 'b_' in key and key[2].isnumeric()],
        sigma2_init=torch.Tensor(theta["sigma2"]),
        pi_init=torch.Tensor(theta["pies"]),
        precision=cmd_args.precision.torch_dtype(),
    )

    # Construct TVO configuration
    estep_conf = AmortizedSamplingConfig(
        n_states=estep_config_dict["n_states"],
        n_samples=cmd_args.N_samples,
    )

    # Setup the experiment
    exp_config = ExpConfig(
        batch_size=cmd_args.batch,
        output=os.path.join(log_path, "inference.h5"),
        reco_epochs=exp_config_dict["reco_epochs"],
        log_blacklist=[],
        log_only_latest_theta=True,
    )
    exp = Training(conf=exp_config, estep_conf=estep_conf, model=model, 
                   train_data_file=None, val_data_file=data_file)
    logger, trainer = exp.logger, exp.trainer
    exp.test_states.set_posterior_sampler(sampler)


    # run epochs
    comm_rank = 0
    merge_strategies = {"mean": mean_merger, "median": median_merger}

    with torch.inference_mode():
        for epoch, summary in enumerate(exp.run(cmd_args.epochs)):
            summary.print()

            # merge reconstructed image patches and generate reconstructed image
            gather = ((epoch) % cmd_args.reco_epochs) == 0
            assert hasattr(trainer, "test_reconstruction")
            rec_patches = gather_from_processes(trainer.test_reconstruction) if gather else None
            merge = gather and comm_rank == 0
            recos = {
                k: ovp.set_and_merge(rec_patches.t(), merge_method=v) if merge else None
                for k, v in merge_strategies.items()
            }

            # assess reconstruction quality in terms of PSNR
            metrics = {k: eval_fn(clean_test[None, ...], v) if merge else None for k, v in recos.items()}

            to_log = (
                {
                    **{f"reco_image_{k}": v for k, v in recos.items()},
                    **{f"psnr_{k}": metrics[k][0] for k, v in metrics.items()},
                    **{f"snr_{k}": metrics[k][1] for k, v in metrics.items()},
                    **{f"pesq_{k}": metrics[k][2] for k, v in metrics.items()},
                }
                if merge
                else None
            )

            if to_log is not None:
                logger.append_and_write(**to_log)

            # Save reconstructed audio if merge
            if merge:

                psnr_mean_str, psnr_median_str = (
                    f"{metrics[k][0]:.2f}"#.replace(".", "_")
                    for k, v in metrics.items()
                )
                snr_mean_str, snr_median_str = (
                    f"{metrics[k][1]:.2f}"#.replace("-", "m").replace(".", "_")
                    for k, v in metrics.items()
                )
                pesq_mean_str, pesq_median_str = (
                    f"{metrics[k][2]:.2f}"#.replace(".", "_") 
                    for k, v in metrics.items()
                )

                wav_file_mean = f"{cmd_args.outdir}/reco-mean-epoch({epoch})-snr({snr_mean_str})-pesq({pesq_mean_str})-psnr({psnr_mean_str}).wav"
                wav_file_median = f"{cmd_args.outdir}/reco-median-epoch({epoch})-snr({snr_median_str})-pesq({pesq_median_str})-psnr({psnr_median_str}).wav"

                reco_audio_mean = recos['mean'].squeeze(0).detach().cpu().numpy()
                reco_audio_median = recos['median'].squeeze(0).detach().cpu().numpy()

                sf.write(wav_file_mean, reco_audio_mean, sr)
                sf.write(wav_file_median, reco_audio_median, sr)
            
                print(f"Wrote {wav_file_mean}, {wav_file_median}", flush=True)

    print("Finished")

    #from torch.profiler import profile, record_function, ProfilerActivity
    #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=100, max_name_column_width=200))