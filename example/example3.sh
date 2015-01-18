#!/bin/bash -e

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Example 3: RBM pre-training (Bernoulli-Bernoulli)
TRAIN=$DIR/data/train2.dat
TEST=$DIR/data/test2.dat
stacked_rbm=$DIR/model/train2.rbm.xml
model=$DIR/model/train2.dnn.mature.xml

opts="--normalize 1 --input-dim 1024"

$DIR/../bin/nn-init $TRAIN $opts --type 1 --output-dim 12 --struct 1024-1024-1024 -o $stacked_rbm
$DIR/../bin/nn-train $opts $TRAIN $stacked_rbm $TEST $model --min-acc 0.75 --base 1
$DIR/../bin/nn-predict $opts $TEST $model --base 1
