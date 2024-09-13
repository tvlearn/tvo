#!/bin/bash

#SBATCH --job-name="tvae-amortize"
#SBATCH --nodelist=gold15
#SBATCH --nodes=1                    
#SBATCH --ntasks=1                  
#SBATCH --mem=60G        
#SBATCH --time=10-00:00
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/%j.out        
#SBATCH --error=slurm/%j.err          

source ~/miniconda3/bin/activate
conda activate tvo

python amortize_audio.py \
--Xfile "/users/ml/gawo7931/ecml2022/experiments/denoising/waveform_denoising/out/1-saved states/LJ-Speech/LJ006-0097/1104150/image_patches.h5" \
--Ksetfile "/users/ml/gawo7931/ecml2022/experiments/denoising/waveform_denoising/out/1-saved states/LJ-Speech/LJ006-0097/1104150/training.h5" \
--epochs_mean "200" \
--epochs_full "200" \
--lr_mean 0.001 \
--lr_full 0.001 \
--outdir "/users/ml/gawo7931/tvo/examples/amortization/results/trained samplers/LJ006-0097/"