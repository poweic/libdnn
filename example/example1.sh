#!/bin/bash -e

# Example 1: random initialization
TRAIN=data/a1a.train.dat
TEST=data/a1a.test.dat
init_model=model/a1a.xml
model=model/a1a.mature.xml

opts="--input-dim 123"

../bin/nn-init $TRAIN $opts --output-dim 2 --struct 256-256 -o $init_model
../bin/nn-train $opts $TRAIN $init_model $TEST $model --min-acc 0.8 --learning-rate 0.5
../bin/nn-predict $opts $TEST $model
