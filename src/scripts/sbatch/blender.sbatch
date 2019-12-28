#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus=4
#SBATCH --mem=24GB
#SBATCH --time=7-00:00:00
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,TIME_LIMIT,TIME_LIMIT_90
eval $@
