#!/bin/bash -e

# Example 4
TRAIN=data/train2.dat
opts="--input-dim 32x32 --base 1"
../bin/cnn-train $opts $TRAIN --output-dim 12 --struct 10x5x5-2s-10x3x3-2s-512-512 --min-acc 0.8
