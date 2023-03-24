#!/bin/bash

# we need train, dev, test1, test2

set -euxo pipefail

### SOURCE TEXT DATA
# - data with punctuation .,?! and casing (Title case, ALL CAPS)

PDTSC_TEXT=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_PDTSC_2.0/data/Czech-PDTSC20/prep/text2
BNC_LDC_TEXT=/mnt/matylda5/iveselyk/KALDI_DATAPREPS/CZECH_BroadcastNewsConvs_LDC/data/Czech-BNC-LDC/prep/text2


# copy the data:
mkdir -p src_data
cp $PDTSC_TEXT src_data/pdtsc
cp $BNC_LDC_TEXT src_data/bnc_ldc

# split bnc_ldc into 3 databases:
grep Czech-BNC_LDC2000S89 src_data/bnc_ldc >src_data/bnc_ldc_LDC2000S89
grep Czech-BNC_LDC2004S01 src_data/bnc_ldc >src_data/bnc_ldc_LDC2004S01
grep Czech-BNC_LDC2009S02 src_data/bnc_ldc >src_data/bnc_ldc_LDC2009S02



### SPLIT PDTSC (train, dev, test)
# ~ 10% test, 10% dev
pdtsc_test_lines=5000
pdtsc_dev_lines=5000
head -n $pdtsc_test_lines src_data/pdtsc >src_data/pdtsc_test
head -n $[ pdtsc_dev_lines + pdtsc_test_lines ] src_data/pdtsc | tail -n $pdtsc_dev_lines >src_data/pdtsc_dev || true
tail -n +$[ pdtsc_dev_lines + pdtsc_test_lines + 1 ] src_data/pdtsc >src_data/pdtsc_train



### SPLIT BNC_LDC
# ~ 10% test, 10% dev
LDC2000S89_10pct=880
LDC2004S01_10pct=1350
LDC2009S02_10pct=1500

##### LDC2000S89
text=src_data/bnc_ldc_LDC2000S89
n_10pct=${LDC2000S89_10pct}
head -n ${n_10pct} ${text} >${text}_test
head -n $[2*n_10pct] ${text} | tail -n ${n_10pct} >${text}_dev || true
tail -n +$[2*n_10pct +1] ${text} >${text}_train

##### LDC2004S01
text=src_data/bnc_ldc_LDC2004S01
n_10pct=${LDC2004S01_10pct}
head -n ${n_10pct} ${text} >${text}_test
head -n $[2*n_10pct] ${text} | tail -n ${n_10pct} >${text}_dev || true
tail -n +$[2*n_10pct +1] ${text} >${text}_train

##### LDC2009S02
text=src_data/bnc_ldc_LDC2009S02
n_10pct=${LDC2009S02_10pct}
head -n ${n_10pct} ${text} >${text}_test
head -n $[2*n_10pct] ${text} | tail -n ${n_10pct} >${text}_dev || true
tail -n +$[2*n_10pct +1] ${text} >${text}_train


### MERGE TRAIN,DEV
cat src_data/*_train >src_data/train
cat src_data/*_dev >src_data/dev
#
# test sets are kept separately:
#
# `ls src_data/*_test`
# src_data/bnc_ldc_LDC2000S89_test
# src_data/bnc_ldc_LDC2004S01_test
# src_data/bnc_ldc_LDC2009S02_test
# src_data/pdtsc_test


### CONVERT DATA TO TRAINING LABELS
for dset in train dev pdtsc_test \
              bnc_ldc_LDC2000S89_test \
              bnc_ldc_LDC2004S01_test \
              bnc_ldc_LDC2009S02_test; do
  cut -d' ' -f2- src_data/${dset} | ./prepare_labels.py >../../data/cz/${dset}
done


