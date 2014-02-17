#!/bin/bash -xe

# Example 2
Train=data/train.dat
Test=data/test.dat
Model=train.dat.model

echo 2 | dnn-init --nodes 64-64 $Train $Model --rescale true
dnn-train $Train --pre 2 -f $Model --min-acc 0.8 --rescale true
dnn-predict $Test $Model --rescale true
