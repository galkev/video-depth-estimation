#!/usr/bin/env bash
SCRIPT_DIR="Documents/master-thesis/code/mt-project/scripts"
SSH_HOST="atbeetz25"
VIRT_ENV=".virtualenvs/pytorch/bin/activate"
SYNC_DIR="slurm_sync"

slurm_job_args="$(printf ' %q' "$@")"

rsync -av ${SYNC_DIR} ${SSH_HOST}:${SCRIPT_DIR}

ssh ${SSH_HOST} "
. ${VIRT_ENV}
cd ${SCRIPT_DIR}
git pull
./run_slurm.sh ${slurm_job_args}
"
