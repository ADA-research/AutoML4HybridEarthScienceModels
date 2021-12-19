#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4
python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed $1 &
python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed $1 &
python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed $1 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed $1 &
python3 run_experiment.py --config yield_2500 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed $1 