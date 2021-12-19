#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4
python3 run_experiment.py --config gbov_0 --model 12hr_baseline --experiment standard --data simulation --random_seed $1 &
python3 run_experiment.py --config gbov_0 --model 12hr_baseline --experiment standard --data simulation --random_seed $(($1 + 1)) &
python3 run_experiment.py --config gbov_0 --model 12hr_baseline --experiment standard --data simulation --random_seed $(($1 + 2)) &
python3 run_experiment.py --config gbov_0 --model 12hr_baseline --experiment standard --data simulation --random_seed $(($1 + 3)) &
python3 run_experiment.py --config gbov_0 --model 12hr_baseline --experiment standard --data simulation --random_seed $(($1 + 4)) 
