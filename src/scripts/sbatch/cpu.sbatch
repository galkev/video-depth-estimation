#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=5-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
eval $@
