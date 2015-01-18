#!/bin/bash -e

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Example 4: CNN-DNN with random initialization
TRAIN=$DIR/data/train2.dat
model=$DIR/model/train2.cnn.init.xml
model_mature=$DIR/model/train2.cnn.mature.xml

struct="--struct 10x5x5-2s-10x3x3-2s-512-512"
dim="--input-dim 32x32"

$DIR/../bin/nn-init $dim $struct -o $model --output-dim 12
$DIR/../bin/nn-train $dim $TRAIN $model - $model_mature --base 1 --min-acc 0.8
$DIR/../bin/nn-predict $dim $TRAIN $model_mature --base 1 
