#!/bin/bash -xe

# Example 1
Train=data/a1a
Test=data/a1a.t
Model=a1a.model

echo 2 | dnn-init --nodes 256-256 $Train $Model
dnn-train $Train --pre 2 -f $Model --min-acc 0.8
dnn-predict $Test $Model
