#!/bin/bash -e

# Example 3
TRAIN=data/train2.dat
TEST=data/test2.dat
stacked_rbm=model/train2.rbm.xml
model=model/train2.dnn.mature.xml

opts="--input-dim 1024 --normalize 1"

../bin/dnn-init $TRAIN $opts --type 1 --output-dim 12 --struct 1024-1024-1024 -o $stacked_rbm
../bin/dnn-train $opts $TRAIN $stacked_rbm $model --min-acc 0.78 --base 1
../bin/dnn-predict $opts $TEST $model --base 1
