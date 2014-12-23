#!/bin/bash -e

# Example 2
TRAIN=data/train.dat
TEST=data/test.dat
stacked_rbm=model/train.dat.rbm
model=model/train.dat.model

opts="--normalize 1 --input-dim 20"

../bin/nn-init $TRAIN $opts --output-dim 2 --struct 64-64 -o $stacked_rbm
../bin/nn-train $opts $TRAIN $stacked_rbm $TEST $model --min-acc 0.74
../bin/nn-predict $opts $TEST $model
