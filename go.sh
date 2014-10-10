#!/bin/bash

bin/cnn-train example/data/train2.dat --input-dim 32x32 --output-dim 12 --base 1 --struct 12x9x9-2s-12x5x5-2s-512-512 --batch-size 32
#./cnn-train example/data/train2.dat --input-dim 32x32 --output-dim 12 --base 1 --struct 10x9x9-2s-20x5x5-2s-512-512 --batch-size 32
#./cnn-train example/data/train2.dat --input-dim 32x32 --output-dim 12 --base 1 --struct 8x7x7-2s-8x5x5-2s-256-256 --batch-size 32
#./cnn-train example/data/train2.dat --input-dim 32x32 --output-dim 12 --base 1 --struct 7x7x7-2s-13x5x5-2s-40 --batch-size 32
#./cnn-train example/data/train2.dat --input-dim 32x32 --output-dim 12 --base 1 --struct 10x13x13-2s-10x5x5-2s-100 --batch-size 32

#./cnn-train example/data/train2.dat --input-dim 32x32 --struct 12x8x8-2s-6x5x5-2s-256-256

#./cnn-train example/data/train2.dat --input-dim 32x32 --struct 12x8x8-2s-6x5x5-2s-256-256
# 10個epoch的執行時間: 306.73 sec
