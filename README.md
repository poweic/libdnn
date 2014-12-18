libdnn
======

[libdnn](https://github.com/botonchou/libdnn) is an open source CUDA-based C++ Library of Deep Neural Network. It aims to provide an user-friendly neural network library, which allow researchers, developers, or anyone interested in it to harness and experience the power of DNN and extend it whenever you need.

Deep Neural Network (DNN) is a very powerful Machine Learning (ML) algorithm which has shown significant success on numerous difficult supervised ML tasks in
- Speech Recognition
- Pattern Recognition and Computer Vision (CV)
- Natural Language Processing (NLP)

# Prerequisite
You need
- g++ (>= 4.6)
- NVIDIA's Graphic Processing Unit (GPU)
- Linux/Unix (I use Ubuntu. Mac OS X should be fine, but not tested yet.)
- [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (>= CUDA 5.0) with CUDA Samples

I use **Ubuntu 14.04** and **NVIDIA GTX-660**.

# Quick Start

### Install
1. `git clone https://github.com/botonchou/libdnn.git`
2. `cd libdnn/`
3. `./install-sh`

### Examples

There're 3 example scripts in `example/`, you should give it a try:
- `./example1.sh`
- `./example2.sh`
- `./example3.sh`

Alternatively, you can run all of them by `./go_all.sh`

### Prepare your data

#### Training data and testing data

In general, you'll need two data, training data (with labels) and test data (optionally labelled).
Of course, you can always split your data into two, using a ratio about 5:1 or something like that (5 for training, 1 for testing). If you just want to play around but without your own data, you can simply run through the **example** provided above or download some from the [LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). 

#### Data Format
The data can be provided either in sparse (like those in LIBSVM) or in dense format.

##### Sparse Format (like LibSVM):
```
-1 5:1 6:1 15:1 22:1 36:1 42:1
+1 3:1 6:1 17:1 19:1 39:1 42:1
-1 5:1 7:1 14:1 22:1 36:1 40:1
-1 1:1 6:1 17:1 22:1 36:1 42:1
+1 4:1 6:1 14:1 29:1 39:1 42:1
-1 3:1 6:1 15:1 22:1 36:1 42:1
+1 5:1 6:1 15:1 22:1 36:1 40:1
```

Each row is one data (**label** + **feature vector**). In this case, 7 rows means 7 feature vector (e.g. 7 patients)
The first column of each row are the labels (e.g., 1 for cancer, -1 for no cancer) , and the rest are feature vectors (e.g., the height and the weight of a patient) in the sparse format. Take the first row for example: `-1 5:1 6:1 15:1 22:1 36:1 42:1`, **-1** is the label. The **n**:**x** format means the value of **n**-th dimension of this vector is **x**. In this example, it's a vector consists most of 0 with only few exceptions at 5, 6, 15, 22, 36, 42.

##### Dense Format:

```
-1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
-1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
```
You can also store the data in a dense format, this is the same data as the above (but in dense format).

## How to Use ?
There're mainly 3 programs, `dnn-init`, `dnn-train`, `dnn-predict`.

### dnn-init
```
dnn-init [options] training_data [model_out]
```
This program will initialize a deep belief network (DBN) using stacked Restricted Boltzmann Machine (stacked RBM). For example:
```
dnn-init --input-dim 1024 --nodes 1024-1024 --output-dim 12 train.dat
```
`--input-dim` stands for the dimensional of input feature vector, `--output-dim` is the number of target classes.
In this example, `dnn-init` will built you a new neural network model of the structure `1024-1024-1024-12`.

### dnn-train
```
dnn-train [options] training_data model_in [model_out]
```
This program will use mini-batch stochastic gradient descent (mini-batch SGD) to train the model initialized by `dnn-init`.
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

For more detail, please check [Wiki](https://github.com/botonchou/libdnn/wiki).

libdnn 中文說明
======

[libdnn](https://github.com/botonchou/libdnn) 是一個用CUDA C++寫成的**深層神經網路**開源函式庫。目標是提供一個簡易易懂的神經網路函式庫 (library)，讓開發人員、研究員、或任何有興趣的人都可以輕鬆體驗並駕馭深層神經網路所帶來的威力。

**深層神經網路 (deep neural network)**是一種非常強大的機器學習演算法。近年來，由於硬體技術的逐漸成熟（主要是來自家用顯示卡的所提供的高效能運算，一般民眾或研究單位與實驗室均可用很便宜的價格輕易取得），深層神經網路在諸多領域上皆獲得前所未有的成功，包括了
- 語音辨識
- 影像辨識
- 自然語言處理

# 系統配備需求 

你需要：
- g++ (>= 4.6)
- 一張NVIDIA的顯示卡 (ex: GTX-660)
- Linux/Unix 作業系統 (我用Ubuntu。Mac OS X應該也行，但還沒空測試。）
- 安裝 [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (至少要CUDA 5.0，或更新)

我的是 **Ubuntu 14.04** 和 **NVIDIA GTX-660**.

# 快速上手

### 安裝
1. `git clone https://github.com/botonchou/libdnn.git`
2. `cd libdnn/`
3. `./install-sh`

### 使用範例

在`example/`下有三個使用範例：
- `./example1.sh`
- `./example2.sh`
- `./example3.sh`

如果想要一次執行全部的範例，你也可以執行位在`example/`下的`./go_all.sh`

### 資料準備

#### 訓練資料與測試資料

一般來說，你會需要準備兩種資料：訓練資料（有答案）和測試資料（答案可有可無）。你也可以用大約5:1的比例，將你手邊的資料切成兩份，一份當作訓練資料(5)，另一份當作測試資料(1)。如果你還沒有準備好自己的資料，只是想要簡單玩玩看，你可以簡單走過一遍上面所提供的**example**，或是到[LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)下載。

#### 資料格式

資料格式有兩種，一種是LibSVM的格式（稀疏矩陣），另一種則是緊密排列的方式(dense)。

##### Sparse Format (like LibSVM):
```
-1 5:1 6:1 15:1 22:1 36:1 42:1
+1 3:1 6:1 17:1 19:1 39:1 42:1
-1 5:1 7:1 14:1 22:1 36:1 40:1
-1 1:1 6:1 17:1 22:1 36:1 42:1
+1 4:1 6:1 14:1 29:1 39:1 42:1
-1 3:1 6:1 15:1 22:1 36:1 42:1
+1 5:1 6:1 15:1 22:1 36:1 40:1
```

每一個橫列(row)代表一筆資料（**正確答案**加上**特徵向量**）。在上面的例子中，7 列就代表有 7 筆資料 (e.g. 7 位病人)。每一橫列的第一欄是正確答案（例如：用 1 代表有癌症，用 -1 代表沒癌症），該列剩下的部份就是特徵向量，以稀疏矩陣的方式表示（例如：身高多少，體重多少等等）。以第一橫列作為例子: `-1 5:1 6:1 15:1 22:1 36:1 42:1`，其中**-1**是正確答案。剩下的部份用 **n**:**x** 的方式代表該向量的第**n**維的值為**x**。在這個例子中，這個向量大部分的值都是0，只有少數幾維的值為1（第5, 6, 15, 22, 36, 42維）。

##### Dense Format:

```
-1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 0 0 1 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
-1 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 1 0 1 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 1 0 0 1
-1 0 0 1 0 0 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 0 0 1
+1 0 0 0 0 1 1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0 1 0 0
```

你也可以不管資料是否為0，將其每一維的值列出來，如上所示。（同一組資料，只是換了格式。）

## 如何使用 

主要有三個程式，一個是`dnn-init`, `dnn-train`, `dnn-predict`.

### dnn-init
```
dnn-init [options] training_data [model_out]
```
透過這個程式，你可以建立一個全新的神經網路模型。它會將很多個 Restricted Boltzmann Machine (RBM) 疊在一起建立出一個Deep Belief Network (DBN) 。如下所示：
```
dnn-init --input-dim 1024 --nodes 1024-1024 --output-dim 12 train.dat
```
`--input-dim`就是資料（或特徵向量）的維度，而`--output-dim`則是**總共要分成幾類**。在上述的例子中，`dnn-init`會建立一個結構為`1024-1024-1024-12`的神經網路模型。

### dnn-train
```
dnn-train [options] training_data model_in [model_out]
```
有了上述`dnn-init`產生出來的神經網路模型後，你可以透過`dnn-train`所提供的mini-batch stochastic gradient descent (mini-batch SGD)對神經網路模型進行訓練。
```
dnn-train train.dat train.dat.model
```

### dnn-predict
```
dnn-predict testing_data model_in [predict_out]
```
當你訓練完神經網路模型後，即可用`dnn-predict`對新的資料進行預測。
```
dnn-predict test.dat train.dat.model
```

更多的教學和細節，請參考[Wiki](https://github.com/botonchou/libdnn/wiki).

# License
Copyright (c) 20013-2014 Po-Wei Chou Licensed under the Apache License.

