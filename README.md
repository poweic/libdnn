libdnn 中文說明
======
(See English version below)

[libdnn](https://github.com/botonchou/libdnn) 是一個輕量、好讀、人性化的**深層學習**函式庫。由 C++ 和 CUDA 撰寫而成，目的是讓開發人員、研究人員、或任何有興趣的人都可以輕鬆體驗並駕馭深層學習所帶來的威力。

詳細的教學和使用說明，請參考[Wiki](https://github.com/botonchou/libdnn/wiki)
和[常見問題 (FAQ)](https://github.com/botonchou/libdnn/wiki/Frequently-Asked-Questions)

## 特色
- 輕量、好讀、人性化
- 支援 [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) 的資料格式，讓你無痛上手！
- 深層神經網路 (deep neural network, DNN)
- 卷積神經網路 (convolutional neural network, CNN)
- dropout, sigmoid, tanh, ReLU 及[其他非線性函數](https://github.com/botonchou/libdnn/wiki/XML-model#changing-activation-functions)
- 以 [XML 格式](https://github.com/botonchou/libdnn/wiki/XML-model)儲存模型
- 遞迴式神經網路 (recurrent neural network, RNN 開發中 )

**深層神經網路 (deep neural network)** 和 **卷積神經網路 (convolutional neural network)** 是種非常強大的機器學習模型。近年來，由於硬體技術的逐漸成熟（主要是來自家用顯示卡的所提供的高效能運算，一般民眾或研究單位與實驗室均可用很便宜的價格輕易取得），深層神經網路在諸多領域上皆獲得前所未有的成功，包括了
- 語音辨識
- 影像辨識
- 自然語言處理

## 系統配備需求 

你需要：
- **g++** (>= 4.6)
- **一張NVIDIA的顯示卡** (ex: GTX-660)
- **Linux/Unix 作業系統**
- **安裝 [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)**  （CUDA 5.0 以上或更新的版本）  
(如果你不知道怎麼安裝 CUDA toolkit，請參考[FAQ](https://github.com/botonchou/libdnn/wiki/Frequently-Asked-Questions))

我的是 **Ubuntu 14.04** 和 **NVIDIA GTX-660**. ( Mac OS X 應該也行，但還沒空測試。）

## 快速上手

安裝前，確認一下你可以執行`g++`, `nvcc` ，並且正確地設定環境變數 `PATH`, `LD_LIBRARY_PATH`.
如果你有點不太確定我在講什麼，我建議你看過一遍
 [FAQ](https://github.com/botonchou/libdnn/wiki/Frequently-Asked-Questions) 再回來。

如果上述這些都搞定了，那就可以開始安裝了!!

### 安裝
1. `git clone https://github.com/botonchou/libdnn.git`
2. `cd libdnn/`
3. `./install-sh`

### 使用範例

在`example/`下有四個使用範例：
- `./example1.sh`
- `./example2.sh`
- `./example3.sh`
- `./example4.sh`

如果想要一次執行全部的範例，你也可以執行位在`example/`下的`./go_all.sh`

### 資料準備

#### 訓練資料與測試資料

一般來說，你會需要準備兩種資料：訓練資料（有答案）和測試資料（答案可有可無）。你也可以用大約5:1的比例，將你手邊的資料切成兩份，一份當作訓練資料(5)，另一份當作測試資料(1)。如果你還沒有準備好自己的資料，只是想要簡單玩玩看，你可以簡單走過一遍上面所提供的**example**，或是到[LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)下載。

#### 資料格式

資料格式有兩種，一種是稀疏矩陣的格式（像 LibSVM 那樣），另一種則是緊密排列的方式(dense)。

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

主要有以下三個程式:

1. nn-init
2. nn-train
3. nn-predict

### nn-init
```
nn-init [options] [training_data] -o model_out
```
透過這個程式，你可以初始化一個全新的神經網路模型（存成 XML 格式），方法有：
1. 隨機初始化
2. Bernoulli-Bernoulli RBM
3. Gaussian-Bernoulli RBM

(詳細說明請參考 command line option `--type`) 。

例如：
```
nn-init --input-dim 600 --struct 1024-1024 --output-dim 12 -o train.init.xml
```
其中`--input-dim`就是資料（或特徵向量）的維度，而`--output-dim`則是**總共要分成幾類**。在上述的例子中，`nn-init`會建立一個結構為`600-1024-1024-12`的神經網路模型，並將模型存在`train.init.xml`。

你也可以建立一個 Convolutional Neural Network ，如下所示：
```
nn-init --input-dim 32x24 --struct 20x5x5-2s-10x3x3-2s-512-512 --output-dim 10
```
這邊的 `32x24`(以`高x寬`的形式) 代表這一張高 32 像素，寬 24 像素的影像，而後面的 `20x5x5-2s-10x3x3-2s-512-512` 則代表：

|   字串  |    說明    |
|:-------:|:-----------|
| 20x5x5  | 20 kernels. 第一個 5 是高度，第二個 5 是寬度（同上，單位一樣是像素） |
| 2s      | 減縮取樣的比例是 2 |
| 10x3x3  | 20x10 kernels instead of only 10 kernels （因為前一層已經是 20 個 kernel ） |
| 2s      | 減縮取樣的比例是 2 |
| 512-512 | 兩層寬度均為 512 的隱藏層。 |

由於 **BLAS** 記憶體布局的方式是 **column-major**，所以資料也要以 **column-major** 的方式儲存 (  ↓  而不是 → ) 。譬如說，有一張 `5x6` 的英文字母 E 如下所示：

```
111111
1     
111111
1     
111111
```
則應該將該圖存成以下形式：
```
1:1 2:1 3:1 4:1 5:1 6:1 8:1 10:1 11:1 13:1 15:1 16:1 18:1 20:1 21:1 23:1 25:1 26:1 28:1 30:1
```

### nn-train
```
nn-train [options] training_data model_in [validion_data] [model_out]
```
有了上述`nn-init`產生出來的神經網路模型後，你可以透過`nn-train`所提供的mini-batch stochastic gradient descent (mini-batch SGD) 對神經網路模型進行訓練。
```
nn-train train.dat train.dat.model
```

### nn-predict
```
nn-predict testing_data model_in [prediction_out]
```
當你訓練完神經網路模型後，即可用`nn-predict`對新的資料進行預測。
```
nn-predict test.dat train.dat.model
```

libdnn
======

[libdnn](https://github.com/botonchou/libdnn) is a lightweight, user-friendly, and readable C++ library for deep learning, which allows researchers, developers, or anyone interested in it to harness and experience the power of deep learning.

For more detail, please check [Wiki](https://github.com/botonchou/libdnn/wiki)
and [Frequently Asked Questions (FAQ)](https://github.com/botonchou/libdnn/wiki/Frequently-Asked-Questions)

## Features
- lightweight, user-friendly, and readable
- support data in [LibSVM](http://www.csie.ntu.edu.tw/~cjlin/libsvm/) format
- deep neural network (DNN)
- convolutional neural network (CNN)
- dropout, sigmoid, tanh, ReLU and [other nonlinearities ...](https://github.com/botonchou/libdnn/wiki/XML-model#changing-activation-functions)
- model in [XML format](https://github.com/botonchou/libdnn/wiki/XML-model)
- recurrent neural network (RNN, under development)

DNN and CNN are powerful machine learning algorithms, which have shown significant success on numerous difficult supervised ML tasks in
- Speech Recognition
- Pattern Recognition and Computer Vision (CV)
- Natural Language Processing (NLP)

## Prerequisite
You need
- **g++** (>= 4.6)
- **an NVIDIA GPU**
- **Linux/Unix** 
- **[NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit)** (>= CUDA 5.0) with CUDA Samples  
(If you don't know how to install CUDA, please go check [FAQ](https://github.com/botonchou/libdnn/wiki/Frequently-Asked-Questions))

I use **Ubuntu 14.04** and **NVIDIA GTX-660**. (Mac OS X should be fine, but not tested yet.)

## Quick Start

Before you install, you should be able to run `g++`, `nvcc` and have your
 environment variable `PATH` and `LD_LIBRARY_PATH` set.
If you feel that you have little doubt about what I'm talking about, I refer you to go through
 [FAQ](https://github.com/botonchou/libdnn/wiki/Frequently-Asked-Questions)

### Install
1. `git clone https://github.com/botonchou/libdnn.git`
2. `cd libdnn/`
3. `./install-sh`

### Examples

There're 4 example scripts in `example/`, you should give it a try:
- `./example1.sh`
- `./example2.sh`
- `./example3.sh`
- `./example4.sh`

Alternatively, you can run all of them by `./go_all.sh`

### Prepare your data

#### Training data and testing data

In general, you'll need two data, training data (with labels) and test data (optionally labelled).
Of course, you can always split your data into two, using a ratio about 5:1 or something like that (5 for training, 1 for testing). If you just want to play around but without your own data, you can simply run through the **example** provided above or download some from the [LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). 

#### Data Format
The data can be provided either in sparse (like those in LibSVM) or in dense format.

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
There're mainly 3 programs:

1. nn-init
2. nn-train
3. nn-predict

### nn-init
```
nn-init [options] [training_data] -o model_out
```
This program will initialize a new neural network model (in XML format) in three ways:

0. random initialization
1. Bernoulli-Bernoulli RBM
2. Gaussian-Bernoulli RBM

(see command line option `--type` for more detail)

For example:
```
nn-init --input-dim 600 --struct 1024-1024 --output-dim 12 train.dat -o train.init.xml
```
where `--input-dim` stands for the dimensional of input feature vector, `--output-dim` is the number of target classes to predict.
In this example, `nn-init` will built you a new neural network model of the structure `600-1024-1024-12`, and save the model as `train.init.xml`.

You can also initialize a Convolutional Nerual Network like this:
```
nn-init --input-dim 32x24 --struct 20x5x5-2s-10x3x3-2s-512-512 --output-dim 10
```
Here, `32x24` (in `hxw` format) means an input image of `32` pixels in height by `24` pixels wide, and `20x5x5-2s-10x3x3-2s-512-512` means:

|  Token  | Explanation |
|:-------:|:-----------|
| 20x5x5  | 20 kernels. The first 5 is height, the second 5 is width. Just like the above |
| 2s      | down-sampling factor 2 |
| 10x3x3  | 20x10 kernels instead of only 10 kernels (because the previous conv layer has 20 kernels) |
| 2s      | down-sampling factor 2 |
| 512-512 | 2 hidden layers, each with 512 hidden nodes. |

Because **BLAS** use **column-major**, you have to provide data in **column-major** (i.e. in ↓ order, not → order).  
For example, if you have an `5x6` image letter **E** like this (5 pixels in height by 6 pixels wide):
```
111111
1     
111111
1     
111111
```
You should provide this image in sparse format like this:
```
1:1 2:1 3:1 4:1 5:1 6:1 8:1 10:1 11:1 13:1 15:1 16:1 18:1 20:1 21:1 23:1 25:1 26:1 28:1 30:1
```

### nn-train
```
nn-train [options] training_data model_in [validation_data] [model_out]
```
This program will use mini-batch stochastic gradient descent (mini-batch SGD) to train the model initialized by `nn-init`.
```
nn-train train.dat train.dat.model
```

### nn-predict
```
nn-predict testing_data model_in [prediction_out]
```
For example:
```
nn-predict test.dat train.dat.model
```

## License
Copyright (c) 20013-2014 Po-Wei Chou Licensed under the Apache License.
