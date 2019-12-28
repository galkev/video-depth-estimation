#!/usr/bin/env bash
slurm_support_scripts=("train_model.py", "train_model_json.py")

function elementExists() {
    elements=${1}
    element=${2}
    for i in ${elements[@]} ; do
        if [[ ${i} == ${element} ]] ; then
            return 0
        fi
    done
    return 1
}

function hasSlurmSupport() {
    return $(elementExists ${slurm_support_scripts} $1)
}
