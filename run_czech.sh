#!/bin/bash

# Example script for training, testing and inference.

. src/conda-activate.sh

set -euxo pipefail


# Train the model
true && {

  # download the model first:
  # ../pretrained_models/download.sh

  cuda=True
  if [ $cuda == "True" ]; then
    export CUDA_VISIBLE_DEVICES=$(./src/free-gpu.py --ngpus=1)
  fi

  # Hint: run with --epoch=0 to get scores from test-sets (no training, only if model is already trained)
  python src/train.py \
    --cuda=${cuda} \
    --pretrained-model=xlm-roberta-base \
    --freeze-bert=False \
    --lstm-dim=-1 \
    --language=czech \
    --seed=1 \
    --lr=5e-6 \
    --epoch=10 \
    --use-crf=False \
    --augment-type=all \
    --augment-rate=0.15 \
    --alpha-sub=0.4 \
    --alpha-del=0.4 \
    --data-path=data \
    --save-path=out_czech
}


# Get precision/recall, f-score and confusion matrix from test sets
false && {

  cuda=True
  if [ $cuda == "True" ]; then
    export CUDA_VISIBLE_DEVICES=$(./src/free-gpu.py --ngpus=1)
  fi

  python src/test.py \
    --cuda=${cuda} \
    --pretrained-model=xlm-roberta-base \
    --weight-path=out_czech/weights.pt \
    --test-data data/cz/dev data/cz/pdtsc_test data/cz/bnc_ldc_LDC2000S89_test data/cz/bnc_ldc_LDC2004S01_test data/cz/bnc_ldc_LDC2009S02_test \
    --log-file=out_czech/punctuation-restore_test_logs.txt
}


# Inference applied to raw text:
false && {
  python src/inference.py \
    --cuda=False \
    --pretrained-model=xlm-roberta-base \
    --in-file=data/test_cz.txt \
    --weight-path=out_czech/weights.pt \
    --out-file=data/test_cz.txt.out
}


# Inference applied to a Kaldi 'text' file:
false && {
  #in_file=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_karafiat_setup/data/train_16k_bison/text
  #out_file=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_karafiat_setup/data/train_16k_bison/text.CasePunct

  in_file=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_karafiat_setup/data/train_16k_momv/text
  out_file=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_karafiat_setup/data/train_16k_momv/text.CasePunct

  cuda=True
  if [ $cuda == "True" ]; then
    export CUDA_VISIBLE_DEVICES=$(./src/free-gpu.py --ngpus=1)
  fi

  # class CasePunctuator, more ready to be integrated...
  python src/inference_class.py \
    --cuda=${cuda} \
    --pretrained-model=xlm-roberta-base \
    --weight-path=out_czech/weights.pt \
    --in-file=${in_file} \
    --out-file=${out_file}
}

