#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4
python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed $1 &
python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed $1 &
python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed $1 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed $1 &
python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed $1 