#!/bin/sh
source /software/anaconda3/etc/profile.d/conda.sh
conda activate conda_master_thesis
export OPENBLAS_NUM_THREADS=4
python3 run_experiment.py --config agb_50 --model 12hr_baseline --random_seed $1 &
python3 run_experiment.py --config agb_100 --model 12hr_baseline --random_seed $1 &
python3 run_experiment.py --config agb_250 --model 12hr_baseline --random_seed $1 &
python3 run_experiment.py --config agb_1000 --model 12hr_baseline --random_seed $1 &
python3 run_experiment.py --config agb_2500 --model 12hr_baseline --random_seed $1 