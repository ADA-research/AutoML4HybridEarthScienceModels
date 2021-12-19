#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4

python3 run_experiment.py --config chla_50 --model GPR --experiment standard --data in_situ --random_seed 13 &
python3 run_experiment.py --config chla_100 --model GPR --experiment standard --data in_situ --random_seed 13 &
python3 run_experiment.py --config chla_250 --model GPR --experiment standard --data in_situ --random_seed 13 &
python3 run_experiment.py --config chla_50 --model GPR --experiment standard --data in_situ --random_seed 14 &
python3 run_experiment.py --config chla_100 --model GPR --experiment standard --data in_situ --random_seed 14 &
python3 run_experiment.py --config chla_250 --model GPR --experiment standard --data in_situ --random_seed 14 &
python3 run_experiment.py --config chla_50 --model GPR --experiment standard --data in_situ --random_seed 15 &
python3 run_experiment.py --config chla_100 --model GPR --experiment standard --data in_situ --random_seed 15 &
python3 run_experiment.py --config chla_250 --model GPR --experiment standard --data in_situ --random_seed 15 &
python3 run_experiment.py --config chla_1000 --model GPR --experiment standard --data in_situ --random_seed 15 &
python3 run_experiment.py --config chla_2500 --model GPR --experiment standard --data in_situ --random_seed 15 

sleep 20m
rm -r /local/s1281437/tmp_*
sleep 10m

python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 1 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 1 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 1 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 1 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 1 &
python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 2 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 2 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 2 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 2 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 2 

sleep 20m
rm -r /local/s1281437/tmp_*
sleep 10m

python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 3 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 3 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 3 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 3 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 3 &
python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 4 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 4 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 4 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 4 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 4 

sleep 20m
rm -r /local/s1281437/tmp_*
sleep 10m

python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 5 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 5 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 5 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 5 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 5 &
python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 6 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 6 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 6 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 6 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 6 

sleep 20m
rm -r /local/s1281437/tmp_*
sleep 10m


python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 7 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 7 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 7 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 7 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 7 &
python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 8 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 8 

sleep 20m
rm -r /local/s1281437/tmp_*
sleep 10m

python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 9 &
python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 10 &
python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 10 &
python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 10 &
python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 10 &
python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 10 

sleep 20m
rm -r /local/s1281437/tmp_*
sleep 10m


# python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 11 &
# python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 12 &
# python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 13 &
# python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 14 &
# python3 run_experiment.py --config yield_50 --model GPR --experiment ts --data in_situ_ts --random_seed 15 &
# python3 run_experiment.py --config yield_100 --model GPR --experiment ts --data in_situ_ts --random_seed 15 &
# python3 run_experiment.py --config yield_250 --model GPR --experiment ts --data in_situ_ts --random_seed 15 &
# python3 run_experiment.py --config yield_1000 --model GPR --experiment ts --data in_situ_ts --random_seed 15 &
# python3 run_experiment.py --config yield_2500 --model GPR --experiment ts --data in_situ_ts --random_seed 15 

# sleep 20m
# rm -r /local/s1281437/tmp_*
# sleep 10m
