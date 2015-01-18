#!/bin/bash -e

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Example 1: random initialization
TRAIN=$DIR/data/a1a.train.dat
TEST=$DIR/data/a1a.test.dat
init_model=$DIR/model/a1a.xml
model=$DIR/model/a1a.mature.xml

opts="--input-dim 123"

$DIR/../bin/nn-init $TRAIN $opts --output-dim 2 --struct 256-256 -o $init_model
$DIR/../bin/nn-train $opts $TRAIN $init_model $TEST $model --min-acc 0.8 --learning-rate 0.5
$DIR/../bin/nn-predict $opts $TEST $model
