#!/bin/bash -e

# Example 1
TRAIN=data/a1a.train.dat
TEST=data/a1a.test.dat
stacked_rbm=model/a1a.rbm
model=model/a1a.model

opts="--input-dim 123 --type 1"

../bin/dnn-init $TRAIN $opts --output-dim 2 --struct 256-256 -o $stacked_rbm
../bin/dnn-train $opts $TRAIN $stacked_rbm $TEST $model --min-acc 0.8 --learning-rate 0.5
../bin/dnn-predict $opts $TEST $model
