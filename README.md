libdnn
======

[libdnn](https://github.com/botonchou/libdnn) is an open source CUDA-based C++ Library of Deep Neural Network. It aims to provide an user-friendly neural network library, which allow researchers, developers, or anyone interested in it to harness and experience the power of DNN and extend it whenever you need.

Neural Network (NN), esp. the **Deep Neural Network (DNN)**, is a very powerful machine learning (ML) algorithm which have shown significant success on numerous difficult supervised ML tasks such as **Automatic Speech Recognition (ASR)**, **Pattern Recognition and Computer Vision (CV)**, **Natural Language Processing (NLP)**, etc.

# Prerequisite
You need
- A Graphic Processing Unit (GPU) of NVIDIA
- Linux/Unix (Ubuntu is fine. But I haven't had the time to tested it Mac OS X yet.)
- Install [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (at least CUDA 5.0)

Mine is **Ubuntu 14.04** and **NVIDIA GTX-660**.

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

# Tutorial

Here, I'll briefly discuss how to prepare your data and train your own neural network.

### Introduction to neural network.

Neural network is one of the many **classifiers** that can do classification, such as telling a man from women, diagnosing whether a patient has a cancer or not, etc.

To train a classifier, you need to provide a bunch of data with labels.
Then, you show it all the data you got and telling it which category each one of them belongs to.
Once it's well trained, it's able to tell you which category an unseen datum belongs to. (Pretty cool, huh?)

Take fruits for example, suppose you have some pictures of apple and orange and you want a machine that can tell apple from orange.
All you have to do is to show NN a bunch of pictures of fruit and tell it which one is apple and which one is orange.
After the training procedure, you might want to take some new photos and ask the machine "Which one is apple?" or "What does it look like? An apple or an orange?".
Of course, you can always fool a well trained machine by giving it an orange-like apple.
But as you might expect, once the machine seen it all, you'll find it hard to trick it anymore.

Another example is healthcare diagnostic system, where you're trying to find potential diseases for the patients given their medical records.
Age, gender, weight, height, habits, family history, medical allergies, etc. are all the information (or **features**) we can use.
Suppose we use only **age** (0 - 150 years old), **gender** (0 for female, 1 for male), **weight** (0 - 500 kg), **height** (0 - 250 meters), these 4 values then compose a 4-dimensional **feature vector** that represent the characteristics of a patient.
(Other features such as family history are also important, but it's your job to find a way to represent them in **real number**.)

Most of the classifiers in machine learning are **binary classifiers** (e.g., apple or orange, man or woman, diseased or not).
Support Vector Machine (SVM) is also one of them.
(Binary classifiers can always extends to multiclass classification by performing one-vs-all or all-pairs comparison.)
Neural network, on the other hand, is a **multiclass classifier**, which is known for its capability of performing multiclass classification with up to tens of thousands of categories (see [clarifai](http://www.clarifai.com/) and [speech recognition on YouTube performed by Google](http://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41403.pdf)).

### Prepare your data

#### Training data and testing data

In general, you'll need two data, training data (with labels) and test data (optionally labelled).
Of course, you can always split your data into two, using a ratio about 5:1 or something like that (5 for training, 1 for testing). If you just want to play around but without your own data, you can simply run through the **example** provided above or download some from the [LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/). 

#### Data Format
The data can be provided either in the LIBSVM format (sparse) or in dense format.

##### LibSVM Format:
```
-1 5:1 6:1 15:1 22:1 36:1 42:1
+1 3:1 6:1 17:1 19:1 39:1 42:1
-1 5:1 7:1 14:1 22:1 36:1 40:1
-1 1:1 6:1 17:1 22:1 36:1 42:1
+1 4:1 6:1 14:1 29:1 39:1 42:1
-1 3:1 6:1 15:1 22:1 36:1 42:1
+1 5:1 6:1 15:1 22:1 36:1 40:1
```

Each row is one data (**label** + **feature vector**). In this case, 7 rows means 7 feature vector (i.e. 7 training data or 7 patients in the previous example)
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

libdnn 中文說明
======

[libdnn](https://github.com/botonchou/libdnn) 是一個用CUDA C++寫成的**深層神經網路**開源函式庫。目標是提供一個簡易易懂的神經網路函式庫 (library)，讓開發人員、研究員、或任何有興趣的人都可以輕鬆體驗並駕馭深層神經網路所帶來的威力。


**深層神經網路 (deep neural network)**是一種非常強大的機器學習演算法。近年來，由於硬體技術的逐漸成熟（主要是來自家用顯示卡的所提供的高效能運算，一般民眾或研究單位與實驗室均可用很便宜的價格輕易取得），深層神經網路在諸多領域上皆獲得前所未有的成功，其中包括了**語音辨識** (automatic speech recognition, ASR)，**圖像辨識**，以及**自然語言處理** (natural language processing, NLP)等。

# 系統配備需求 

你需要：
- 一張NVIDIA的顯示卡 (ex: GTX-660)
- Linux/Unix 作業系統 (Ubuntu也行。但我還沒有時間在Mac OS X上面測試。）
- 安裝 [NVIDIA CUDA toolkit](https://developer.nvidia.com/cuda-toolkit) (CUDA的版本號至少要大於5.0)

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

# 使用教學 

### 神經網路簡介及運作方式 

神經網路是眾多**分類器**的其中一種，它主要的功用在於幫你做分類，譬如說分辨出男生女生或是診斷有沒有得癌症等等。

要訓練這樣一個分類器（或機器），首先你需要有一些訓練資料(training data)以及這每筆訓練資料所對應的類別，稱作標記(label)或正確答案。
接著，你只要把這些訓練資料拿給機器看，並告訴他哪一筆屬於哪一種類別，機器就會在這學習的過程中，想辦法找出其中的關聯性。
往後你只要拿一筆它沒看過得新資料，它就有辦法告訴你這筆新資料屬於哪個類別。

舉個例子，假設你想弄出一台可以單從照片就分辨出蘋果還是橘子的機器，作法很簡單：拿一堆橘子和蘋果的照片給機器看，告訴它哪個是橘子哪個是蘋果（這個過程叫做學習）。當它看完了這些照片後，你就可以隨手拍一堆新的照片並問它「哪些是蘋果？」或是「請問這張比較像橘子還是蘋果？」之類的問題。
（你當然也可以拿一些很像橘子的蘋果耍它。）

另一個例子是健康診斷系統：試圖從病人的病歷中和各種資料中，找出潛在疾病的可能性。你可以拿年齡、性別、身高、體重、習慣、家族病史、藥物過敏等等當作判別的依據。假設我拿**年齡** (0 - 150歲)、**性別** (0是女性，1是男性)、**身高** (0 - 250公尺)、**體重** (0 - 500公斤)當作我的診斷依據，那我就可以用這四個**特徵**所組成的4維**特徵向量**來代表一位病人。

在機器學習的領域中，大多數的分類器都屬於**二元分類器** (例如: 蘋果與橘子，男人或女人，有生病還是沒有生病）。支持向量機（Support Vector Machine, SVM. 我找不到更好的翻譯了@_@）也是二元分類器的其中一種。這些二元分類器當然也是可以透過「一對多」或是「兩兩相比」的方法，對兩種以上的類別進行多類別的分類。

神經網路則是一種非常著名的**多類別分類器**，可以成千上萬的類別進行分類。（見 [clarifai](http://www.clarifai.com/) 以及 [Google在YouTube上做的語音辨識](http://static.googleusercontent.com/media/research.google.com/zh-TW//pubs/archive/41403.pdf))

### 資料準備

#### 訓練資料與測試資料

一般來說，你會需要準備兩種資料：訓練資料（有答案）和測試資料（答案可有可無）。你也可以用大約5:1的比例，將你手邊的資料切成兩份，一份當作訓練資料(5)，另一份當作測試資料(1)。如果你還沒有準備好自己的資料，只是想要簡單玩玩看，你可以簡單走過一遍上面所提供的**example**，或是到[LibSVM website](http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)下載。

#### 資料格式

資料格式有兩種，一種是LibSVM的格式（稀疏矩陣），另一種則是緊密排列的方式(dense)。

##### LibSVM Format:
```
-1 5:1 6:1 15:1 22:1 36:1 42:1
+1 3:1 6:1 17:1 19:1 39:1 42:1
-1 5:1 7:1 14:1 22:1 36:1 40:1
-1 1:1 6:1 17:1 22:1 36:1 42:1
+1 4:1 6:1 14:1 29:1 39:1 42:1
-1 3:1 6:1 15:1 22:1 36:1 42:1
+1 5:1 6:1 15:1 22:1 36:1 40:1
```

每一個橫列(row)代表一筆資料（**正確答案**加上**特徵向量**）。在上面的例子中，7 列就代表有 7 筆資料 (e.g. 前述例子中的 7 位病人)。每一橫列的第一欄是正確答案（例如：用 1 代表有癌症，用 -1 代表沒癌症），該列剩下的部份就是特徵向量，以稀疏矩陣的方式表示（例如：身高多少，體重多少等等）。以第一橫列作為例子: `-1 5:1 6:1 15:1 22:1 36:1 42:1`，其中**-1**是正確答案。剩下的部份用 **n**:**x** 的方式代表該向量的第**n**維的值為**x**。在這個例子中，這個向量大部分的值都是0，只有少數幾維的值為1（第5, 6, 15, 22, 36, 42維）。

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

# License
Copyright (c) 20013-2014 Po-Wei Chou Licensed under the Apache License.

