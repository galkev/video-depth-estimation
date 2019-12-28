#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus=1
#SBATCH --gres=gpu:titanxpascal:1
#SBATCH --mem=8GB
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
eval $@
