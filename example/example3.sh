#!/bin/bash -e

# Example 3: RBM pre-training (Bernoulli-Bernoulli)
TRAIN=data/train2.dat
TEST=data/test2.dat
stacked_rbm=model/train2.rbm.xml
model=model/train2.dnn.mature.xml

opts="--input-dim 1024 --normalize 1"

../bin/nn-init $TRAIN $opts --type 1 --output-dim 12 --struct 1024-1024-1024 -o $stacked_rbm
../bin/nn-train $opts $TRAIN $stacked_rbm $TEST $model --min-acc 0.75 --base 1
../bin/nn-predict $opts $TEST $model --base 1
