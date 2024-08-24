#!/bin/bash

#SBATCH --job-name="TVO-AMORTIZE"
#SBATCH --output=out/slurm.%j.out
#SBATCH --error=out/slurm.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --time=10-00:00
#SBATCH --exclude=gold[01,02,03,04,05,06,07,08,09,10,15]
#SBATCH --gres=gpu:1 

# load the modules you need

# activate local conda
source ~/miniconda3/bin/activate

# activate the conda environment
conda activate tvo-env

# get the path of the last generated training data 
export TVODATADIR=$(ls -d ../gaussiandenoising/out/* | tail -n 1)

# run the model training script
python amortize.py \
  --Xfile ${TVODATADIR}/image_patches.h5 \
  --Ksetfile ${TVODATADIR}/training.h5 \
  --precision float32 \
  --epochs_mean 200 \
  --epochs_full 200  \
  --batch_size 32 \
  --N_IS 32 \
  --t_start 2.0 \
  --t_end 2.0 \
  --lr_mean 0.001 \
  --lr_full 0.001


export TVODATADIR=$(ls -d ../gaussiandenoising/out/* | tail -n 1)
export SAMPLERDIR=$(ls -d ./out/*-amortize | tail -n 1)

python infer.py \
  --model ${TVODATADIR}/training.h5 \
  --Xfile ${TVODATADIR}/image_patches.h5 \
  --sampler ${SAMPLERDIR}/trained_sampler.pt \
  --precision float32 \
  --epochs 10 \
  --batch 1000 \
  --N_samples 100
