#!/bin/bash

if [ $(hostname -d) == "fit.vutbr.cz" ]; then
  unset PYTHONHOME # Specific setup of BUT
  CONDA_ENV=/mnt/matylda5/iveselyk/CONDA_ENVS/punctuation
#elif [ $(hostname -d) == "tilde.com" ]; then
else
  echo "Error: Unknown environment (hostname $(hostname -d))"
  exit 1
fi

# make 'conda activate' findable:
[ -z "${CONDA_EXE}" ] && echo "Error, missing $CONDA_EXE !" && exit 1
CONDA_BASE=$(${CONDA_EXE} info --base)
source $CONDA_BASE/etc/profile.d/conda.sh

# activate the conda environment
conda activate ${CONDA_ENV}

