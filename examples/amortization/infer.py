# Perform amortized TVO inference and store ELBO history w.r.t. samples.

# python infer.py --model ../gaussiandenoising/out/24-07-31-16-48-59/training.h5 
# --sampler ./out/24.08.02-15:06:10/trained_sampler.pt 
# --Xfile ../gaussiandenoising/out/24-07-31-16-48-59/image_patches.h5
# --epochs 10 --CPU


import os
from datetime import datetime
import numpy as np
import torch
import argparse
from enum import Enum
import h5py
import tvo
from tvo.models import BSC
from tvo.exp import EVOConfig, AmortizedSamplingConfig, ExpConfig, Training
from tvo.utils.parallel import pprint, broadcast, barrier, bcast_shape, gather_from_processes
from tvutil.prepost import OverlappingPatches, MultiDimOverlappingPatches, mean_merger, median_merger
from utils.viz import Visualizer
from utils.utils import eval_fn
from models.amortizedbernoulli import SamplerModule
#from models.amortizedbernoulli import AmortizedBernoulli, compute_probabilities, Objective, binarize
#from models.variationalparams import FullCovarGaussianVariationalParams, AmortizedGaussianVariationalParams
#from utils.training import train
#from utils.plotting import plot_epoch_log


class FloatPrecision(Enum):
    float16 = "float16"  # torch.float16
    float32 = "float32"  # torch.float32
    float64 = "float64"  # torch.float64

    def __str__(self):
        return str(self.value)

    @staticmethod
    def from_string(s):
        try:
            return FloatPrecision[s]
        except KeyError:
            raise ValueError()
        

def load_group_as_dict(hdf5filename, groupname):
    res = {}
    with h5py.File(hdf5filename, "r") as f:
        group = f[groupname]
        for k in group.keys():
            v = group[k]#[:]
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
    arg_parser.add_argument("--Xfile", type=str, help="X dataset file, HDF5", default=None)
    arg_parser.add_argument("--N_start", type=int, help="Data slice start", default=0)
    arg_parser.add_argument("--N_size", type=int, help="Data slice size", default=None)
    arg_parser.add_argument("--epochs", type=int, help="Number of epochs to run the sampler", default=10)
    arg_parser.add_argument("--batch", type=int, help="Batch size", default=32)
    arg_parser.add_argument("--N_samples", type=int, help="Number of samples to draw", default=128)
    arg_parser.add_argument("--CPU", action="store_true")
    arg_parser.add_argument("--precision", type=FloatPrecision, help="Compute precision", default=FloatPrecision.float64)
    arg_parser.add_argument("--outdir", type=str, help="Output directory", default=os.path.join("./out", datetime.now().strftime('%y.%m.%d-%H:%M:%S')))

    cmd_args = arg_parser.parse_args()
    print("Parameters:")
    for var in vars(cmd_args):
        print("  {}: {}". format(var, vars(cmd_args)[var]))

    eval("torch.set_default_dtype(torch.{})".format(cmd_args.precision))
    device = torch.device("cuda" if torch.cuda.is_available() and not cmd_args.CPU else "cpu") 
    print("Using PyTorch device/precision: {}/{}".format(torch.get_default_device(), torch.get_default_dtype()))
    tvo._set_device(device)

    os.makedirs(cmd_args.outdir, exist_ok=True)

    # Load BSC model parameters
    theta = load_group_as_dict(cmd_args.model, "theta")
    model_config_dict = load_group_as_dict(cmd_args.model, "model_config")
    estep_config_dict = load_group_as_dict(cmd_args.model, "estep_config")
    exp_config_dict = load_group_as_dict(cmd_args.model, "exp_config")

    # Load posterior sampler object from a *.pt file
    assert cmd_args.sampler is not None
    sampler = torch.load(cmd_args.sampler)
    assert isinstance(sampler, SamplerModule)
    sampler.to(device)

    # Load noisy image and extract image patches
    clean = load_var(cmd_args.model, "clean_image")
    noisy = load_var(cmd_args.model, "noisy_image")

    patch_width = 5
    patch_height = 5
    isrgb = clean.dim() == 3 and clean.shape[2] == 3
    OVP = MultiDimOverlappingPatches if isrgb else OverlappingPatches
    ovp = OVP(noisy, patch_height, patch_width, patch_shift=1)

    # Construct BCS model
    model = BSC(
            H=model_config_dict["H"],
            D=model_config_dict["D"],
            W_init=torch.Tensor(theta["W"]),
            sigma2_init=torch.Tensor(theta["sigma2"]),
            pies_init=torch.Tensor(theta["pies"]),
            precision= torch.float32 if model_config_dict["precision"] == "torch.float32" else torch.float64,
            #device=torch.get_default_device(),
        )

    # Construct EVO configuration
    estep_conf = EVOConfig(
        n_states=estep_config_dict["n_states"],
        n_parents=estep_config_dict["n_parents"],
        n_children=estep_config_dict["n_children"],
        n_generations=estep_config_dict["n_generations"],
        parent_selection=estep_config_dict["parent_selection"].decode("utf-8"),
        crossover=estep_config_dict["crossover"],
    )
    estep_conf = AmortizedSamplingConfig(
        n_states=estep_config_dict["n_states"],
        n_samples=cmd_args.N_samples,
    )

    # Setup the experiment
    exp_config = ExpConfig(
        batch_size=int(exp_config_dict["batch_size"]),
        output=os.path.join(cmd_args.outdir, "inference.h5"),
        reco_epochs=exp_config_dict["reco_epochs"],
        log_blacklist=[],
        log_only_latest_theta=True,
    )
    exp = Training(conf=exp_config, estep_conf=estep_conf, model=model, 
                   train_data_file=None, val_data_file=cmd_args.Xfile)
    logger, trainer = exp.logger, exp.trainer
    exp.test_states.set_posterior_sampler(sampler)

    # initialize visualizer
    print("Initializing visualizer")
    visualizer = Visualizer(
            viz_every=1,
            output_directory=cmd_args.outdir,
            clean_image=clean,
            noisy_image=noisy,
            patch_size=(patch_height, patch_width),
            sort_gfs=True,
            ncol_gfs=3,
            gif_framerate=None,
        )

    # run epochs
    comm_rank = 0
    merge_strategies = {"mean": mean_merger, "median": median_merger}
    for epoch, summary in enumerate(exp.run(cmd_args.epochs)):
        summary.print()

        # merge reconstructed image patches and generate reconstructed image
        gather = epoch in (exp_config_dict["reco_epochs"] + 1)
        assert hasattr(trainer, "test_reconstruction")
        rec_patches = gather_from_processes(trainer.test_reconstruction) if gather else None
        merge = gather and comm_rank == 0
        imgs = {
            k: ovp.set_and_merge(rec_patches.t(), merge_method=v) if merge else None
            for k, v in merge_strategies.items()
        }

        # assess reconstruction quality in terms of PSNR
        psnrs = {k: eval_fn(clean, v) if merge else None for k, v in imgs.items()}

        to_log = (
            {
                **{f"reco_image_{k}": v for k, v in imgs.items()},
                **{f"psnr_{k}": v for k, v in psnrs.items()},
            }
            if merge
            else None
        )

        if to_log is not None:
            logger.append_and_write(**to_log)
        barrier()

        # visualize epoch
        if comm_rank == 0:
            gfs = model.theta["W"]
            visualizer.process_epoch(
                epoch=epoch,
                pies=model.theta["pies"],
                gfs=gfs,
                rec=imgs["mean"] if merge else None,
            )

    print("Finished")
