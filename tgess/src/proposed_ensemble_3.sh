#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

python3 run_experiment.py --config chla_50 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
python3 run_experiment.py --config chla_100 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
python3 run_experiment.py --config chla_250 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
python3 run_experiment.py --config chla_50 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config chla_100 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config chla_250 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config chla_50 --model proposed_ensemble --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config chla_100 --model proposed_ensemble --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config chla_250 --model proposed_ensemble --experiment custom --data both --random_seed 12 
python3 run_experiment.py --config chla_50 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config chla_100 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config chla_250 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config chla_50 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config chla_100 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config chla_250 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config chla_50 --model proposed_ensemble --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config chla_100 --model proposed_ensemble --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config chla_250 --model proposed_ensemble --experiment custom --data both --random_seed 15 
# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 8 
# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 10 #&
