#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 6 &
python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 6 &
python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 6 &
python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 6 &
python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed 14

python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 7 &
python3 run_experiment.py --config yield_500 --model proposed_ensemble --experiment ts --data both_ts --random_seed 7 &
python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 7 &
python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 7 &
python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 8 &
python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 8 &
python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 8 &
python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 8 &
python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed 15


python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 9 &
python3 run_experiment.py --config yield_500 --model proposed_ensemble --experiment ts --data both_ts --random_seed 9 &
python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 9 &
python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 9 &
python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 10 &
python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 10 &
python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 10 &
python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 10 
