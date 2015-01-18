#!/bin/bash -e

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Example 2: random initialization
TRAIN=$DIR/data/train.dat
TEST=$DIR/data/test.dat
init_model=$DIR/model/train.dat.xml
model=$DIR/model/train.dat.mature.xml

opts="--normalize 1 --input-dim 20"

$DIR/../bin/nn-init $TRAIN $opts --output-dim 2 --struct 64-64 -o $init_model
$DIR/../bin/nn-train $opts $TRAIN $init_model $TEST $model --min-acc 0.74
$DIR/../bin/nn-predict $opts $TEST $model
