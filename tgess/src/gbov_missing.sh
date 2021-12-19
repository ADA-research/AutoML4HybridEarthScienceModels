#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

# # 0
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 7 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 8 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 9 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 10 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 11 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 12 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 13 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 14 &
# python3 run_experiment.py --config gbov_0 --model autosklearn-5 --experiment standard --data simulation --random_seed 15 &

# python3 run_experiment.py --config gbov_0 --model MLP --experiment standard --data simulation --random_seed 11 &
# python3 run_experiment.py --config gbov_0 --model MLP --experiment standard --data simulation --random_seed 12 &
# python3 run_experiment.py --config gbov_0 --model MLP --experiment standard --data simulation --random_seed 13 &
# python3 run_experiment.py --config gbov_0 --model MLP --experiment standard --data simulation --random_seed 14 &
# python3 run_experiment.py --config gbov_0 --model MLP --experiment standard --data simulation --random_seed 15 

# # 50
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 6 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 7 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 8 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 9 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 10 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_50 --model autosklearn-5 --experiment standard --random_seed 15 &

# python3 run_experiment.py --config gbov_50 --model MLP --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_50 --model MLP --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_50 --model MLP --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_50 --model MLP --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_50 --model MLP --experiment standard --random_seed 15 

# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 6 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 7 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 11 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 12 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 13 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 14 &
# python3 run_experiment.py --config gbov_50 --model proposed --experiment custom --data both --random_seed 15 &

# python3 run_experiment.py --config gbov_50 --model RF --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_50 --model RF --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_50 --model RF --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_50 --model RF --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_50 --model RF --experiment standard --random_seed 15 

# # 100
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 7 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 8 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 9 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 10 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_100 --model autosklearn-5 --experiment standard --random_seed 15 &

# python3 run_experiment.py --config gbov_100 --model MLP --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_100 --model MLP --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_100 --model MLP --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_100 --model MLP --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_100 --model MLP --experiment standard --random_seed 15 

# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 11 &
# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 12 &
# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 13 &
# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 14 &
# python3 run_experiment.py --config gbov_100 --model proposed --experiment custom --data both --random_seed 15 &

# python3 run_experiment.py --config gbov_100 --model RF --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_100 --model RF --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_100 --model RF --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_100 --model RF --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_100 --model RF --experiment standard --random_seed 15 

# # 250
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 8 &
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 9 &
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 10 &
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_250 --model autosklearn-5 --experiment standard --random_seed 15 &

# python3 run_experiment.py --config gbov_250 --model MLP --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_250 --model MLP --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_250 --model MLP --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_250 --model MLP --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_250 --model MLP --experiment standard --random_seed 15 

# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 6 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 7 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 8 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 9 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 10 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 11 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 12 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 13 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 14 &
# python3 run_experiment.py --config gbov_250 --model proposed --experiment custom --data both --random_seed 15 &

# python3 run_experiment.py --config gbov_250 --model RF --experiment standard --random_seed 11 &
# python3 run_experiment.py --config gbov_250 --model RF --experiment standard --random_seed 12 &
# python3 run_experiment.py --config gbov_250 --model RF --experiment standard --random_seed 13 &
# python3 run_experiment.py --config gbov_250 --model RF --experiment standard --random_seed 14 &
# python3 run_experiment.py --config gbov_250 --model RF --experiment standard --random_seed 15 

# 1000
python3 run_experiment.py --config gbov_1000 --model autosklearn-5 --experiment standard --random_seed 9 &
python3 run_experiment.py --config gbov_1000 --model autosklearn-5 --experiment standard --random_seed 10 &
python3 run_experiment.py --config gbov_1000 --model autosklearn-5 --experiment standard --random_seed 11 &
python3 run_experiment.py --config gbov_1000 --model autosklearn-5 --experiment standard --random_seed 12 &
python3 run_experiment.py --config gbov_1000 --model autosklearn-5 --experiment standard --random_seed 13 &
python3 run_experiment.py --config gbov_1000 --model autosklearn-5 --experiment standard --random_seed 14 &
python3 run_experiment.py --config gbov_1000 --model autosklearn-5 --experiment standard --random_seed 15 &

python3 run_experiment.py --config gbov_1000 --model MLP --experiment standard --random_seed 12 &
python3 run_experiment.py --config gbov_1000 --model MLP --experiment standard --random_seed 13 &
python3 run_experiment.py --config gbov_1000 --model MLP --experiment standard --random_seed 14 &
python3 run_experiment.py --config gbov_1000 --model MLP --experiment standard --random_seed 15 

python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 8 &
python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 9 &
python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 10 &
python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config gbov_1000 --model proposed --experiment custom --data both --random_seed 15 &

python3 run_experiment.py --config gbov_1000 --model RF --experiment standard --random_seed 12 &
python3 run_experiment.py --config gbov_1000 --model RF --experiment standard --random_seed 13 &
python3 run_experiment.py --config gbov_1000 --model RF --experiment standard --random_seed 14 &
python3 run_experiment.py --config gbov_1000 --model RF --experiment standard --random_seed 15 

# 2500
python3 run_experiment.py --config gbov_2500 --model autosklearn-5 --experiment standard --random_seed 11 &
python3 run_experiment.py --config gbov_2500 --model autosklearn-5 --experiment standard --random_seed 12 &
python3 run_experiment.py --config gbov_2500 --model autosklearn-5 --experiment standard --random_seed 13 &
python3 run_experiment.py --config gbov_2500 --model autosklearn-5 --experiment standard --random_seed 14 &
python3 run_experiment.py --config gbov_2500 --model autosklearn-5 --experiment standard --random_seed 15 &

python3 run_experiment.py --config gbov_2500 --model MLP --experiment standard --random_seed 11 &
python3 run_experiment.py --config gbov_2500 --model MLP --experiment standard --random_seed 12 &
python3 run_experiment.py --config gbov_2500 --model MLP --experiment standard --random_seed 13 &
python3 run_experiment.py --config gbov_2500 --model MLP --experiment standard --random_seed 14 &
python3 run_experiment.py --config gbov_2500 --model MLP --experiment standard --random_seed 15 

python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 11 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 12 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 13 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 14 &
python3 run_experiment.py --config gbov_2500 --model proposed --experiment custom --data both --random_seed 15 &

python3 run_experiment.py --config gbov_2500 --model RF --experiment standard --random_seed 12 &
python3 run_experiment.py --config gbov_2500 --model RF --experiment standard --random_seed 13 &
python3 run_experiment.py --config gbov_2500 --model RF --experiment standard --random_seed 14 &
python3 run_experiment.py --config gbov_2500 --model RF --experiment standard --random_seed 15 