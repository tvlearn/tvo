#!/bin/bash

#SBATCH --job-name="template"
#SBATCH --output=./.slurm-out/job.%j.out
#SBATCH --error=./.slurm-out/job.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --time=10-02:00:00
#SBATCH --partition=hfag.p
#SBATCH --gpus-per-node=1
# #SBATCH --reservation=conf-paper

if command -v squeue >/dev/null 2>&1
then
    # load the modules you need
    module load CUDA/12.4.0

    # activate local conda
    source ~/miniconda3/bin/activate

    # activate the conda environment
    conda activate torch-env

    # configuration file of the job:
fi

if [[ -z $TVO_GPU ]]
then
    unset TVO_GPU
fi

# run the python script
if [[ $PARALLEL_RUNS ]]
then
    for ((i=1;i<=$PARALLEL_RUNS;i++)); do
        cd $REPO_DIR
        env OUT_PATH=$OUT_PATH/run-$i $MAKEFILE_DIR/.run.sh &
    done
    wait
else
    cd $REPO_DIR
    source $MAKEFILE_DIR/.run.sh 
fi
