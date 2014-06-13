libdnn
======

C++ Library of Deep Neural Network

# Prerequisite
You need
- A Graphic Processing Unit (GPU)
- Linux / Ubuntu
- Install [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (at least CUDA 5.0)

# Quick Start
1. ```git clone https://github.com/botonchou/libdnn.git && cd libdnn && ./install-sh```
2. ```cd example/```
3. ```dnn-init    data/a1a   model/a1a.rbm   --input-dim 123 --output-dim 2 --nodes 256-256```
4. ```dnn-train   data/a1a   model/a1a.rbm   --input-dim 123 model/a1a.model --min-acc 0.8```
5. ```dnn-predict data/a1a.t model/a1a.model --input-dim 123 ```

# Examples
There're 3 examples in example/:
- example1.sh
- example2.sh
- example3.sh

You can run all of them by ```./go_all.sh```

# How to Install ?
```bash
git clone https://github.com/botonchou/libdnn.git
cd libdnn/
./install-sh
```

# Tutorial

Libdnn aims to an easy quick machine learning using **deep neural network**.

### Prepare your data

#### Training data and testing data

You need at least two data files, one with labels (called training data) and the other with/without labels (called testing data.)

If you just want to mess around but without data at hand, you can download some from the [LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). 

#### Data Format
The data can be provided either in the LIBSVM format or the dense format.

##### LIBSVM format:
```
-1 5:1 6:1 15:1 22:1 36:1 42:1
+1 3:1 6:1 17:1 19:1 39:1 42:1
-1 5:1 7:1 14:1 22:1 36:1 40:1
-1 1:1 6:1 17:1 22:1 36:1 42:1
+1 4:1 6:1 14:1 29:1 39:1 42:1
-1 3:1 6:1 15:1 22:1 36:1 42:1
+1 5:1 6:1 15:1 22:1 36:1 40:1
```

Each row is one feature vector. In this case, 7 rows means 7 feature vector (i.e. 7 training data.)
The first column of each row are the labels, and the rest are feature vectors in the sparse format. Take the first row for example:
```
-1 5:1 6:1 15:1 22:1 36:1 42:1
```
**-1** is the label. The n:x format means the value of n-th dimension of this vector is x. In this example, it's a vector consists most of 0 with only few exceptions at 5, 6, 15, 22, 36, 42.

##### Dense format:

```
-1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
-1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
```
(this is the same data as the above)

## How to Use ?
There're 3 programs, dnn-init, dnn-train, dnn-predict.

### dnn-init
```
dnn-init [options] training_data [model_out]
```
This program performs a model initialization using stacked Restricted Boltzmann Machine (stacked RBM). For example:
```
dnn-init --nodes 1024-1024 train.dat
```
The program will first ask you for the number of target classes (i.e. the number of nodes in output layer, **Dout**). Also, it'll found the dimension of input feature vector (**Din**) from *train.dat*. 
It then perform stacked RBM initialization for a neural network of the structure:
```
Din-1024-1024-Dout
```
(**Note**: If the values in train.dat does not lie in the range **[0, 1]**, it would cause an assertion. Try to add ```--normalize 1``` after ```dnn-init``` like: ```dnn-init --normalize 1 --nodes 1024-1024 train.dat```.)

### dnn-train
```
dnn-train [options] training_data model_in [model_out]
```
This program perform mini-batch stochastic gradient descent (mini-batch SGD) to train the model initialized by ```dnn-init```.
```
dnn-train train.dat train.dat.model
```

### dnn-predict
```
dnn-predict testing_data model_in [predict_out]
```
For example:
```
dnn-predict test.dat train.dat.model
```
