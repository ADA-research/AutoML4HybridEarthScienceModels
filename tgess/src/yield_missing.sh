#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

# 50

# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_50 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 15 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 4 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 5 &

# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 6 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 7 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_50 --model MLP --experiment ts --data in_situ_ts --random_seed 15 

# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 6 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 7 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 8 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 9 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 10 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 11 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 12 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 13 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 14 &
# python3 run_experiment.py --config yield_50 --model proposed --experiment ts --data both_ts --random_seed 15 &

# python3 run_experiment.py --config yield_50 --model RF --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_50 --model RF --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_50 --model RF --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_50 --model RF --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_50 --model RF --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_50 --model RF --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_50 --model RF --experiment ts --data in_situ_ts --random_seed 15 

# 100

# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 6 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 7 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_100 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 15 &

# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 6 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 7 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 8 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 9 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 10 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 11 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 12 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 13 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 14 &
# python3 run_experiment.py --config yield_100 --model proposed --experiment ts --data both_ts --random_seed 15 

# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_100 --model MLP --experiment ts --data in_situ_ts --random_seed 15 &

# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 6 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 7 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_100 --model RF --experiment ts --data in_situ_ts --random_seed 15 

# # 250
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 7 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 8 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 9 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 10 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 11 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 12 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 13 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 14 &
# python3 run_experiment.py --config yield_250 --model proposed --experiment ts --data both_ts --random_seed 15 &

# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 6 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 7 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_250 --model MLP --experiment ts --data in_situ_ts --random_seed 15 

# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_250 --model RF --experiment ts --data in_situ_ts --random_seed 15 &

# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 5 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 6 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 7 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 8 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 9 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 10 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_250 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 15 

# 1000
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 8 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 9 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 10 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 11 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 12 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 13 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 14 &
python3 run_experiment.py --config yield_1000 --model proposed --experiment ts --data both_ts --random_seed 15 

python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 10 &
python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 11 &
python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 12 &
python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 13 &
python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 14 &
python3 run_experiment.py --config yield_1000 --model MLP --experiment ts --data in_situ_ts --random_seed 15 

python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 10 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 11 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 12 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 13 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 14 &
python3 run_experiment.py --config yield_1000 --model RF --experiment ts --data in_situ_ts --random_seed 15 

python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 7 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 10 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 11 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 12 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 13 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 14 &
python3 run_experiment.py --config yield_1000 --model autosklearn-5 --experiment ts --data in_situ_ts --random_seed 15 

# 2500

python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed 11 &
python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed 12 &
python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed 13 &
python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed 14 &
python3 run_experiment.py --config yield_2500 --model proposed --experiment ts --data both_ts --random_seed 15 &

python3 run_experiment.py --config yield_2500 --model MLP --experiment ts --data in_situ_ts --random_seed 11 &
python3 run_experiment.py --config yield_2500 --model MLP --experiment ts --data in_situ_ts --random_seed 12 &
python3 run_experiment.py --config yield_2500 --model MLP --experiment ts --data in_situ_ts --random_seed 13 &
python3 run_experiment.py --config yield_2500 --model MLP --experiment ts --data in_situ_ts --random_seed 14 &
python3 run_experiment.py --config yield_2500 --model MLP --experiment ts --data in_situ_ts --random_seed 15 &

python3 run_experiment.py --config yield_2500 --model RF --experiment ts --data in_situ_ts --random_seed 14 &
python3 run_experiment.py --config yield_2500 --model RF --experiment ts --data in_situ_ts --random_seed 15 

