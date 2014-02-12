libdnn
======

C++ Library of Deep Neural Network

## Prerequisite
You need to install NVIDIA CUDA toolkit (at least 5.0)
(Here'e the link https://developer.nvidia.com/cuda-toolkit)



## Quick Start
1. Download the [training data](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a) and the [testing data](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t).
2. ```git clone https://github.com/botonchou/libdnn.git```
3. ```./install-sh```

## How to Install ?
```
./install-sh
```

## How to Use ?

Libdnn aims to an easy quick machine learning using **deep neural network**.

### Prepare your data

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
**-1** is the label. The **n:x** format means the value of **n-th** dimension of this vector is **x**. In this example, it's a vector consists most of 0 with only few exceptions at 5, 6, 15, 22, 36, 42.

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

#### Training data and testing data

You need at least two data files, one with labels (called training data) and the other with/without labels (called testing data.)

If you just want to mess around but without data at hand, you can download some from the [LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). 

### Start training
There're 3 programs, dnn-init, dnn-train, dnn-predict.

#### Example Usage
