#!/bin/bash

echo "$(date +'%F %T')    Executing job commands"

# Commands to execute
cd ./examples/gaussian-denoising
env HDF5_USE_FILE_LOCKING='FALSE' python main.py bsc \
--clean_image ./img/lena.png \
--output_directory $OUT_PATH \
--no_epochs 4001 \
--viz_every 100 \
--batch_size 64 \
--rescale 1 \
--Ksize 200 \
--patch_height 8 \
-H 256 \
--no_parents 10 \
--no_children None \
--no_generations 4 \
--noise_level 25 \
--crossover

# End of commands to execute
