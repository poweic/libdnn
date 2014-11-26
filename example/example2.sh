#!/bin/bash -e

# Example 2
TRAIN=data/train.dat
TEST=data/test.dat
stacked_rbm=model/train.dat.rbm
model=model/train.dat.model

opts="--normalize 1 --input-dim 20"

../bin/dnn-init $opts --type 1 --output-dim 2 --nodes 64-64 $TRAIN $stacked_rbm
../bin/dnn-train $opts $TRAIN $stacked_rbm $model --min-acc 0.74
../bin/dnn-predict $opts $TEST $model
