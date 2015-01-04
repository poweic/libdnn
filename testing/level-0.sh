#!/bin/bash -xe

../example/example1.sh
../example/example2.sh

# Test non-square images and kernels
# All the following codes were modified from ../example/example4.sh

DIR=../example/

TRAIN=$DIR/data/train2.dat
model=$DIR/model/train2.cnn.init.xml
model_mature=$DIR/model/train2.cnn.mature.xml

go_test() {
  ../bin/nn-init $dim $struct -o $model --output-dim 12
  ../bin/nn-train $dim $TRAIN $model - $model_mature --base 1 --max-epoch 5
  ../bin/nn-predict $dim $TRAIN $model_mature --base 1 
}

# Case 1: one 16x64 image
struct="--struct 10x3x9-2s-10x2x7-2s-512"
dim="--input-dim 16x64"
go_test

# Case 2: 4 16x16 images
struct="--struct 10x3x3-2s-10x2x2-2s-512"
dim="--input-dim 4x16x16"
go_test

# Case 3: 4 16x16 images
struct="--struct 10x3x3-2s-10x2x2-512"
dim="--input-dim 4x16x16"
go_test

# Case 4: 4 16x16 images
struct="--struct 10x3x3-10x2x2-2s-512"
dim="--input-dim 4x16x16"
go_test

# Case 5: one 8x128 image
struct="--struct 10x3x13-10x3x12-512"
dim="--input-dim 8x128"
go_test

# Case 6: 8 8x16 images
struct="--struct 10x3x8-10x3x4-512"
dim="--input-dim 8x8x16"
go_test
