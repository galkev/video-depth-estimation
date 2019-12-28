#!/usr/bin/env bash
MODEL_DIR="Documents/master-thesis/models"
SSH_HOST="atbeetz25"

ssh ${SSH_HOST} "ls -lt ${MODEL_DIR}/*"
