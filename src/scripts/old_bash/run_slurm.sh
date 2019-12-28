#!/usr/bin/env bash

. slurm_support.sh

SLURM_LOG_DIR="slurm_logs"

OPTIND=1
while getopts ":b:j:" opt; do
  case ${opt} in
    b) batch_file="${OPTARG}"
    ;;
    j) job_name="${OPTARG}"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done
shift $((OPTIND-1))

prog_name=$1

case "${prog_name}" in
  *.py)
  interpreter="python3"
  if hasSlurmSupport ${prog_name}; then
    add_args='--job_id ${SLURM_JOBID}'
  fi
  ;;
  *)
  interpreter=""
esac

JID=$(sbatch --parsable \
  -o ${SLURM_LOG_DIR}/slurm-%j.out \
  -J "${job_name:-$*}" \
  ${batch_file:-sbatch/default.sbatch} \
  ${interpreter} $@ ${add_args})

slurm_out_file="${SLURM_LOG_DIR}/slurm-${JID}.out"

echo "Job ${JID} started"
echo "Listening to output"
touch ${slurm_out_file}
tail -f ${slurm_out_file}