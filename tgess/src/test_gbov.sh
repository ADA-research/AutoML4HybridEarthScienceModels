#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

python3 run_experiment.py --config test_gbov --model proposed_ensemble --experiment custom --data both --random_seed 1 &
python3 run_experiment.py --config test_gbov --model proposed_ensemble --experiment custom --data both --random_seed 1 &
python3 run_experiment.py --config test_gbov --model proposed_ensemble --experiment custom --data both --random_seed 1 &
python3 run_experiment.py --config test_gbov --model proposed_ensemble --experiment custom --data both --random_seed 1 &
python3 run_experiment.py --config test_gbov --model proposed_ensemble --experiment custom --data both --random_seed 1 


sleep 1000m