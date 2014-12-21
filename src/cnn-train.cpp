// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cuda_profiler_api.h>

#include <cmdparser.h>
#include <pbar.h>

#include <dataset.h>
#include <dnn.h>
#include <cnn.h>

Config config;

void cuda_profiling_ground();

size_t cnn_predict(CNN& cnn, DataSet& data, ERROR_MEASURE errorMeasure);

void cnn_train(CNN& cnn, DataSet& train, DataSet& valid,
    size_t batchSize, const string& fn, ERROR_MEASURE errorMeasure);

int main(int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
     .add("model_in")
     .add("valid_set_file", false)
     .add("model_out", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n"
	 "For example: --input-dim 39x9 \n")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma ."
	"filename -- Read mean and variance from file", "0")
     .add("--base", "Label id starts from 0 or 1 ?", "0");

  cmd.addGroup("Training options:")
     .add("-v", "ratio of training set to validation set (split automatically)", "5")
     .add("--max-epoch", "number of maximum epochs", "100000")
     .add("--min-acc", "Specify the minimum cross-validation accuracy", "0.5")
     .add("--learning-rate", "learning rate in back-propagation", "0.1")
     .add("--batch-size", "number of data per mini-batch", "32");

  cmd.addGroup("Hardward options:")
     .add("--cache", "specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: cnn-train data/train3.dat --struct=12x5x5-2-8x3x3-2");
  
  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn     = cmd[1];
  string model_in     = cmd[2];
  string valid_fn     = cmd[3];
  string model_out    = cmd[4];

  NormType n_type   = (NormType) (int) cmd["--normalize"];
  int base	    = cmd["--base"];

  int ratio	      = cmd["-v"];
  size_t batchSize    = cmd["--batch-size"];
  float learningRate  = cmd["--learning-rate"];
  float minValidAcc   = cmd["--min-acc"];
  size_t maxEpoch     = cmd["--max-epoch"];

  size_t cache_size   = cmd["--cache"];
  CudaMemManager<float>::setCacheSize(cache_size);

  // Parse input dimension
  SIZE imgSize = parseInputDimension((string) cmd["--input-dim"]);
  size_t input_dim = imgSize.m * imgSize.n;

  // Set configurations
  config.learningRate = learningRate;
  config.minValidAccuracy = minValidAcc;
  config.maxEpoch = maxEpoch;

  // Filename for output model
  if (model_out.empty())
    model_out = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  // Load data
  DataSet train, valid;

  if ((valid_fn.empty() or valid_fn == "-" ) && ratio != 0) {
    DataSet data(train_fn, input_dim, base, n_type);
    DataSet::split(data, train, valid, ratio);
  }
  else {
    train = DataSet(train_fn, input_dim, base, n_type);
    valid = DataSet(valid_fn, input_dim, base, n_type);
  }

  train.showSummary();
  valid.showSummary();

  // Load CNN
  CNN cnn;
  cnn.read(model_in);
  cnn.status();

  cnn_train(cnn, train, valid, batchSize, model_out, CROSS_ENTROPY);

  cnn.save(model_out);

  return 0;
}

void cnn_train(CNN& cnn, DataSet& train, DataSet& valid,
    size_t batchSize, const string& model_out, ERROR_MEASURE errorMeasure) {

  perf::Timer timer;
  timer.start();

  // FIXME merge class CNN and DNN.
  // Then merge src/dnn-train.cpp and src/cnn-train.cpp
  const size_t MAX_EPOCH = 1024;
  config.maxEpoch = std::min(config.maxEpoch, MAX_EPOCH);

  size_t nTrain = train.size(),
	 nValid = valid.size();

  mat fout;
  float t_start = timer.getTime();

  for (size_t epoch=0; epoch<config.maxEpoch; ++epoch) {

    Batches batches(batchSize, nTrain);
    for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
      auto data = train[itr];
      auto x = removeBiasAndTranspose(data.x);

      cnn.feedForward(fout, x);

      mat error = getError( data.y, fout, errorMeasure);

      cnn.backPropagate(error, x, fout, config.learningRate / itr->nData);
    }

    size_t Ein  = cnn_predict(cnn, train, errorMeasure),
	   Eout = cnn_predict(cnn, valid, errorMeasure);

    float trainAcc = 1.0f - (float) Ein / nTrain;
    float validAcc = 1.0f - (float) Eout / nValid;
    printf("Epoch #%lu: Training Accuracy = %.4f %% ( %lu / %lu ), Validation Accuracy = %.4f %% ( %lu / %lu ), elapsed %.3f seconds.\n",
      epoch, trainAcc * 100, nTrain - Ein, nTrain, validAcc * 100, nValid - Eout, nValid, (timer.getTime() - t_start) / 1000); 

    if (validAcc > config.minValidAccuracy)
      break;

    cnn.save("." + model_out);
    t_start = timer.getTime();
  }

  timer.elapsed();
  printf("# of total epoch = %lu\n", config.maxEpoch);
}

size_t cnn_predict(CNN& cnn, DataSet& data, ERROR_MEASURE errorMeasure) {

  size_t nError = 0;
  mat fout;

  // TODO automatically compute the best batch_size !!
  Batches batches(256, data.size());
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    auto d = data[itr];
    auto x = removeBiasAndTranspose(d.x);

    cnn.feedForward(fout, x);
    nError += zeroOneError(fout, d.y, errorMeasure);
  }

  return nError;
}

void cuda_profiling_ground() {
  mat x = randn(128, 128),
      h = randn(20, 20);

  perf::Timer timer;
  timer.start();
  cudaProfilerStart(); 
  
  mat z;
  for (int i=0; i<10000; ++i) {
    z = convn(x, h, VALID_SHM);
  }

  CCE(cudaDeviceSynchronize());
  cudaProfilerStop();
  timer.elapsed();
}

