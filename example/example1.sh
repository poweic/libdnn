#!/bin/bash -e

# Example 1
TRAIN=data/a1a
TEST=data/a1a.t
stacked_rbm=model/a1a.rbm
model=model/a1a.model

opts="--input-dim 123"

../dnn-init $opts --output-dim 2 --nodes 256-256 $TRAIN $stacked_rbm
../dnn-train $opts $TRAIN $stacked_rbm $model --min-acc 0.8
../dnn-predict $opts $TEST $model
