#!/bin/bash -e

# Example 2: random initialization
TRAIN=data/train.dat
TEST=data/test.dat
init_model=model/train.dat.xml
model=model/train.dat.mature.xml

opts="--normalize 1 --input-dim 20"

../bin/nn-init $TRAIN $opts --output-dim 2 --struct 64-64 -o $init_model
../bin/nn-train $opts $TRAIN $init_model $TEST $model --min-acc 0.74
../bin/nn-predict $opts $TEST $model
