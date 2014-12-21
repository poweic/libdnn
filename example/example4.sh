#!/bin/bash -e

# Example 4
TRAIN=data/train2.dat
struct="--struct 10x5x5-2s-10x3x3-2s-512-512"
model=model/train2.init.xml
dim="--input-dim 32x32"
../bin/dnn-init $dim $struct -o $model --output-dim 12
../bin/cnn-train $dim $TRAIN $model --base 1 --min-acc 0.8
