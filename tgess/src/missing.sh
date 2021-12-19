#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 13 &

python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 14 

python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 15 


python3 run_experiment.py --config gbov_250 --model GPR --experiment standard --data in_situ --random_seed 13 &
python3 run_experiment.py --config gbov_250 --model GPR --experiment standard --data in_situ --random_seed 14 &
python3 run_experiment.py --config gbov_250 --model GPR --experiment standard --data in_situ --random_seed 15 &
python3 run_experiment.py --config gbov_250 --model RF --experiment standard --data in_situ --random_seed 14 &
python3 run_experiment.py --config gbov_250 --model RF --experiment standard --data in_situ --random_seed 15 &

python3 run_experiment.py --config gbov_50 --model RF --experiment standard --data in_situ --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model GPR --experiment standard --data in_situ --random_seed 13 &
python3 run_experiment.py --config gbov_2500 --model GPR --experiment standard --data in_situ --random_seed 14 &
python3 run_experiment.py --config gbov_2500 --model GPR --experiment standard --data in_situ --random_seed 15 &
python3 run_experiment.py --config gbov_2500 --model GPR --experiment standard --data in_situ --random_seed 13 &
