#!/bin/bash -xe

# Example 1
Train=data/a1a
Test=data/a1a.t
Model=a1a.model

echo 2 | dnn-init --nodes 256-256 $Train $Model
dnn-train $Train --pre 2 -f $Model --min-acc 0.8
dnn-predict $Test $Model

# Example 2
Train=data/train.dat
Test=data/test.dat
Model=train.dat.model

echo 2 | dnn-init --nodes 64-64 $Train $Model --rescale true
dnn-train $Train --pre 2 -f $Model --min-acc 0.8 --rescale true
dnn-predict $Test $Model --rescale true

# Example 3
Train=data/train2.dat
Test=data/test2.dat
Model=train2.dat.model

echo 12 | dnn-init --nodes 1024-1024-1024 $Train $Model --rescale true
dnn-train $Train --pre 2 -f $Model --min-acc 0.75 --rescale true
dnn-predict $Test $Model --rescale true
