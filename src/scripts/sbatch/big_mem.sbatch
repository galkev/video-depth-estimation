#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus=1
#SBATCH --gres=gpu:p6000
#SBATCH --mem=128GB
#SBATCH --time=3-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
eval $@
