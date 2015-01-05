#!/bin/bash -e

# This is the highest level of testing.
# Use "diff" to strictly compare accuracies in nn-train and nn-predict.

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Create temp file and directory
temp_dir=$(mktemp -d -u -p .).level-3
mkdir -p $temp_dir

temp_log=$temp_dir/log
printf "\33[34m[Info]\33[0m Log saved to $temp_log\n"

# Download testing case and extract it
model_url=https://www.dropbox.com/s/yed28svefu3zff5/init.xml.tar.gz?dl=0
wget $model_url -qO- | tar zxv -C $temp_dir
model=$temp_dir/model/train2.cnn.init.xml
gold_log=$temp_dir/train_predict.log

# Start strict testing
TRAIN=../example/data/train2.dat
model_mature=$temp_dir/model/train2.cnn.mature.xml

struct="--struct 10x5x5-2s-10x3x3-2s-512-512"
dim="--input-dim 32x32"

../bin/nn-train $dim $TRAIN $model - $model_mature --base 1 --min-acc 0.8 2>&1 |\
  grep "%" | cut -c 1-61 >> $temp_log

../bin/nn-predict $dim $TRAIN $model_mature --base 1 2>&1 |\
  grep "%" >> $temp_log

printf "\33[33m[Warning]\33[0m Since CUDA is parallel computing, the result may differ from devices to devices.\n"
diff $gold_log $temp_log

rm -r $temp_dir
