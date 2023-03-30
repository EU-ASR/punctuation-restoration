#!/bin/bash

# Download multi-lingual pre-trained BERT model
# - 94 languages, incl. Czech, Estonian, Greek
# - it is cased
#
# https://huggingface.co/xlm-roberta-base

# Disable downloading big files (there are files we don't need)
export GIT_LFS_SKIP_SMUDGE=1

git lfs install
git clone https://huggingface.co/xlm-roberta-base

# Download the Pytorch model
{
  cd xlm-roberta-base/
  git lfs pull --include "pytorch_model.bin" # 1.1GB ...
}

