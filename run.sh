#!/bin/bash
#SBATCH --time=0-10:00 # time (DD-HH:MM)

python -u main.py \
    --config configs/config.yml \
