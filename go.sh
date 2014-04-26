#!/bin/bash

./cnn-train example/data/train2.dat --input-dim 32x32 --output-dim 12 --base 1 --struct 4x5x5-2s-4x5x5-2s-20 --batch-size 32

#./cnn-train example/data/train2.dat --input-dim 32x32 --struct 12x8x8-2s-6x5x5-2s-256-256

#./cnn-train example/data/train2.dat --input-dim 32x32 --struct 12x8x8-2s-6x5x5-2s-256-256
# 10個epoch的執行時間: 306.73 sec
