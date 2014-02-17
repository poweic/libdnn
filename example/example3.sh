#!/bin/bash -xe

# Example 3
Train=data/train2.dat
Test=data/test2.dat
Model=train2.dat.model

echo 12 | dnn-init --nodes 1024-1024-1024 $Train $Model --rescale true
dnn-train $Train --pre 2 -f $Model --min-acc 0.75 --rescale true
dnn-predict $Test $Model --rescale true
