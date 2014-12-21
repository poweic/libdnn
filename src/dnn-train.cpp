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

#include <iostream>
#include <string>
#include <cnn.h>
#include <cmdparser.h>
#include <rbm.h>
#include <batch.h>
using namespace std;

size_t cnn_predict(CNN& cnn, DataSet& data, ERROR_MEASURE errorMeasure);
void cnn_train(CNN& cnn, DataSet& train, DataSet& valid, size_t batchSize, ERROR_MEASURE errorMeasure);
bool isEoutStopDecrease(const std::vector<size_t> Eout, size_t epoch, size_t nNonIncEpoch);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
     .add("model_in")
     .add("valid_set_file", false)
     .add("model_out", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--nf", "Load pre-computed statistics from file", "")
     .add("--base", "Label id starts from 0 or 1 ?", "0");

  cmd.addGroup("Training options:")
     .add("-v", "ratio of training set to validation set (split automatically)", "5")
     .add("--max-epoch", "number of maximum epochs", "200")
     .add("--min-acc", "Specify the minimum cross-validation accuracy", "0.5")
     .add("--learning-rate", "learning rate in back-propagation", "0.1")
     .add("--batch-size", "number of data per mini-batch", "32");

  cmd.addGroup("Hardward options:")
     .add("--cache", "specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: dnn-train data/train3.dat --nodes=16-8");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn     = cmd[1];
  string model_in     = cmd[2];
  string valid_fn     = cmd[3];
  string model_out    = cmd[4];

  NormType n_type     = (NormType) (int) cmd["--normalize"];
  string n_filename   = cmd["--nf"];
  int base	      = cmd["--base"];

  int ratio	      = cmd["-v"];
  size_t batchSize    = cmd["--batch-size"];
  float learningRate  = cmd["--learning-rate"];
  float minValidAcc   = cmd["--min-acc"];
  size_t maxEpoch     = cmd["--max-epoch"];

  size_t cache_size   = cmd["--cache"];
  CudaMemManager<float>::setCacheSize(cache_size);

  // Parse input dimension
  size_t input_dim = parseInputDimension((string) cmd["--input-dim"]);
  
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

  // Set configurations
  Config config;
  config.learningRate = learningRate;
  config.minValidAccuracy = minValidAcc;
  config.maxEpoch = maxEpoch;
  config.print();

  // Load model
  CNN cnn(model_in);
  cnn.status();
  cnn.setConfig(config);

  // Start Training
  cnn_train(cnn, train, valid, batchSize, CROSS_ENTROPY);

  cnn.save(model_out);

  return 0;
}

void cnn_train(CNN& cnn, DataSet& train, DataSet& valid, size_t batchSize, ERROR_MEASURE errorMeasure) {

  printf("Training...\n");
  perf::Timer timer;
  timer.start();

  size_t Ein = 1;
  size_t MAX_EPOCH = cnn.getConfig().maxEpoch, epoch;
  std::vector<size_t> Eout;

  float lr = cnn.getConfig().learningRate / batchSize;

  size_t nTrain = train.size(),
	 nValid = valid.size();

  mat fout;

  printf("._______._________________________._________________________.___________.\n"
         "|       |                         |                         |           |\n"
         "|       |        In-Sample        |      Out-of-Sample      |  Elapsed  |\n"
         "| Epoch |__________.______________|__________.______________|   Time    |\n"
         "|       |          |              |          |              | (seconds) |\n"
         "|       | Accuracy | # of correct | Accuracy | # of correct |           |\n"
         "|_______|__________|______________|__________|______________|___________|\n");

  perf::Timer etimer;
  for (epoch=0; epoch<MAX_EPOCH; ++epoch) {
    etimer.reset();
    etimer.start();

    Batches batches(batchSize, nTrain);
    for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {

      // Copy a batch of data from host to device
      auto data = train[itr];

      cnn.feedForward(fout, data.x);

      mat error = getError( data.y, fout, errorMeasure);

      cnn.backPropagate(error, data.x, fout, lr);
    }

    Ein = cnn_predict(cnn, train, errorMeasure);
    Eout.push_back(cnn_predict(cnn, valid, errorMeasure));

    float trainAcc = 1.0f - (float) Ein / nTrain;

    if (trainAcc < 0) {
      cout << "."; cout.flush();
      continue;
    }

    float validAcc = 1.0f - (float) Eout[epoch] / nValid;

    float time = etimer.getTime() / 1000;

    printf("|%4lu   | %6.2f %% |  %7lu     | %6.2f %% |  %7lu     |  %8.2f |\n",
      epoch, trainAcc * 100, nTrain - Ein, validAcc * 100, nValid - Eout[epoch], time);

    if (validAcc > cnn.getConfig().minValidAccuracy &&
	isEoutStopDecrease(Eout, epoch, cnn.getConfig().nNonIncEpoch))
      break;
  }

  // Show Summary
  printf("|_______|__________|______________|__________|______________|___________|\n");
  printf("\n%ld epochs in total\n", epoch);
  timer.elapsed();

  printf("[   In-Sample   ] ");
  showAccuracy(Ein, train.size());
  printf("[ Out-of-Sample ] ");
  showAccuracy(Eout.back(), valid.size());
}

size_t cnn_predict(CNN& cnn, DataSet& data, ERROR_MEASURE errorMeasure) {

  const size_t batchSize = 256;
  size_t nError = 0;
  mat prob;

  cnn.setDropout(false);
  Batches batches(batchSize, data.size());
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    auto d = data[itr];
    cnn.feedForward(prob, d.x);
    nError += zeroOneError(prob, d.y, errorMeasure);
  }
  cnn.setDropout(true);

  return nError;
}

bool isEoutStopDecrease(const std::vector<size_t> Eout, size_t epoch, size_t nNonIncEpoch) {

  for (size_t i=0; i<nNonIncEpoch; ++i) {
    if (epoch - i > 0 && Eout[epoch] > Eout[epoch - i])
      return false;
  }

  return true;
}

