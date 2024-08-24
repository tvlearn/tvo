#!/bin/bash

# rsync -a ./slurm-job.sh kega9926@gold.medizin.uni-oldenburg.de:~/projects/uol/tvo/examples/amortization/

ssh kega9926@gold.medizin.uni-oldenburg.de <<-'ENDSSH'
	cd ~/projects/uol/tvo/examples/amortization
    sbatch slurm-job.sh
ENDSSH
