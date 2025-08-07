#!/bin/bash
#SBATCH -t 24:00:00
#SBATCH -c 1
#SBATCH --mem-per-cpu=48G
#SBATCH -p stud
#SBATCH --gres=gpu:1
python run_ppo_plus.py --algo p3o --seed 5
