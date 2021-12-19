source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

python3 run_experiment.py --config agb_50 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config agb_100 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config agb_250 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config agb_1000 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config agb_2500 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config agb_50 --model proposed_ensemble --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config agb_100 --model proposed_ensemble --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config agb_250 --model proposed_ensemble --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config agb_1000 --model proposed_ensemble --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config agb_2500 --model proposed_ensemble --experiment custom --data both --random_seed 12 
python3 run_experiment.py --config agb_50 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config agb_100 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config agb_250 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config agb_1000 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config agb_2500 --model proposed_ensemble --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config agb_50 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config agb_100 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config agb_250 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config agb_1000 --model proposed_ensemble --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config agb_2500 --model proposed_ensemble --experiment custom --data both --random_seed 14 
# python3 run_experiment.py --config agb_50 --model proposed_ensemble --experiment custom --data both --random_seed 5 &
# python3 run_experiment.py --config agb_100 --model proposed_ensemble --experiment custom --data both --random_seed 5 &
# python3 run_experiment.py --config agb_250 --model proposed_ensemble --experiment custom --data both --random_seed 5 &
# python3 run_experiment.py --config agb_1000 --model proposed_ensemble --experiment custom --data both --random_seed 5 &
# python3 run_experiment.py --config agb_2500 --model proposed_ensemble --experiment custom --data both --random_seed 5 #&

# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 6 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 6 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 6 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 6 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 6 &
# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 7 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 7 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 7 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 7 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 7 

# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 9 

# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_50 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
# python3 run_experiment.py --config gbov_100 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
# python3 run_experiment.py --config gbov_250 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
# python3 run_experiment.py --config gbov_1000 --model proposed_ensemble --experiment custom --data both --random_seed 11 &
# python3 run_experiment.py --config gbov_2500 --model proposed_ensemble --experiment custom --data both --random_seed 11 #&

# python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 1 &
# python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 1 &
# python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 1 &
# python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 1 &
# python3 run_experiment.py --config yield_2500 --model proposed_ensemble --experiment ts --data both_ts --random_seed 1 &
# python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 2 &
# python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 2 &
# python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 2 &
# python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 2 &
# python3 run_experiment.py --config yield_2500 --model proposed_ensemble --experiment ts --data both_ts --random_seed 2 

# python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 3 &
# python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 3 &
# python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 3 &
# python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 3 &
# python3 run_experiment.py --config yield_2500 --model proposed_ensemble --experiment ts --data both_ts --random_seed 3 &
# python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 4 &
# python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 4 &
# python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 4 &
# python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 4 &
# python3 run_experiment.py --config yield_2500 --model proposed_ensemble --experiment ts --data both_ts --random_seed 4 

# python3 run_experiment.py --config yield_50 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
# python3 run_experiment.py --config yield_100 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
# python3 run_experiment.py --config yield_250 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
# python3 run_experiment.py --config yield_1000 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &
# python3 run_experiment.py --config yield_2500 --model proposed_ensemble --experiment ts --data both_ts --random_seed 5 &

# python3 run_experiment.py --config gbov_0 --model GPR --experiment standard --data simulation --random_seed 1 &
# python3 run_experiment.py --config gbov_0 --model GPR --experiment standard --data simulation --random_seed 2 &
# python3 run_experiment.py --config gbov_0 --model GPR --experiment standard --data simulation --random_seed 3 &
# python3 run_experiment.py --config gbov_0 --model GPR --experiment standard --data simulation --random_seed 4 &
# python3 run_experiment.py --config gbov_0 --model GPR --experiment standard --data simulation --random_seed 5 