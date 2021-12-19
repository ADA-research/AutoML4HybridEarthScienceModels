#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4
python3 run_experiment.py --config gbov_50 --model GPR_AL --random_seed $1 &
python3 run_experiment.py --config gbov_100 --model GPR_AL --random_seed $1 &
python3 run_experiment.py --config gbov_250 --model GPR_AL --random_seed $1 &
python3 run_experiment.py --config gbov_1000 --model GPR_AL --random_seed $1 &
python3 run_experiment.py --config gbov_2500 --model GPR_AL --random_seed $1 