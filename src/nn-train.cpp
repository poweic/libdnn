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
#include <nnet.h>
#include <cmdparser.h>
#include <rbm.h>
#include <batch.h>
using namespace std;

bool save_model_per_epoch = false;
size_t nnet_predict(NNet& nnet, DataSet& data);
void nnet_train(NNet& nnet, DataSet& train, DataSet& valid, string model_out);
bool isEoutStopDecrease(const std::vector<size_t> Eouts, size_t epoch, size_t nNonIncEpoch);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
     .add("model_in")
     .add("valid_set_file", false)
     .add("model_out", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "Specify the input dimension (dimension of feature).")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--nf", "Load pre-computed statistics from file", "")
     .add("--base", "Label id starts from 0 or 1 ?", "0");

  cmd.addGroup("Training options:")
     .add("-v", "ratio of training set to validation set (split automatically)", "5")
     .add("--max-epoch", "number of maximum epochs", "200")
     .add("--min-acc", "Specify the minimum accuracy to achieve in validation "
	 "set before the training process stopped.", "0.5")
     .add("--learning-rate", "learning rate in back-propagation", "0.1")
     .add("--batch-size", "number of data per mini-batch", "32")
     .add("--save-model-per-epoch", "Save model after each epoch of training.", "false");

  cmd.addGroup("Hardward options:")
     .add("--card-id", "Specify which GPU card to use", "0")
     .add("--cache", "Specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: nn-train train.dat init.xml --input-dim 123");

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
  save_model_per_epoch= (bool) cmd["--save-model-per-epoch"];

  size_t card_id      = cmd["--card-id"];
  size_t cache_size   = cmd["--cache"];
  CudaMemManager<float>::setCacheSize(cache_size);
  SetGpuCardId(card_id);

  // Parse input dimension
  size_t input_dim = parseInputDimension((string) cmd["--input-dim"]);
  
  // Filename for output model
  if (model_out.empty())
    model_out = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  // Load data
  DataSet train, valid;

  if ((valid_fn.empty() or valid_fn == "-" ) && ratio != 0) {
    DataSet data(train_fn, input_dim, base, n_type, n_filename);
    DataSet::split(data, train, valid, ratio);
  }
  else {
    train = DataSet(train_fn, input_dim, base, n_type, n_filename);
    valid = DataSet(valid_fn, input_dim, base, n_type, n_filename);
  }

  train.showSummary();
  valid.showSummary();

  // Set configurations
  Config config;
  config.learningRate = learningRate;
  config.minValidAccuracy = minValidAcc;
  config.maxEpoch = maxEpoch;
  config.batchSize = batchSize;
  config.errorMeasure = CROSS_ENTROPY;
  config.print();

  // Load model
  NNet nnet(model_in);
  nnet.status();
  nnet.setConfig(config);

  // Start Training
  nnet_train(nnet, train, valid, model_out);

  nnet.save(model_out);

  return 0;
}

void nnet_train(NNet& nnet, DataSet& train, DataSet& valid, string model_out) {

  printf("Training...\n");
  perf::Timer timer;
  timer.start();

  size_t Ein = 0, Eout = 0;
  size_t MAX_EPOCH = nnet.getConfig().maxEpoch, epoch;
  std::vector<size_t> Eouts;

  float lr = nnet.getConfig().learningRate;

  size_t nTrain = train.size();
  size_t nValid = valid.size();

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

    Batches batches(nnet.getConfig().batchSize, nTrain);
    for (auto itr = batches.begin(); itr != batches.end(); ++itr) {

      // Copy a batch of data from host to device
      auto data = train[itr];
      mat x = ~mat(data.x);

      nnet.feedForward(fout, x);

      mat error = getError( data.y, fout, nnet.getConfig().errorMeasure );

      nnet.backPropagate(error, x, fout, lr / itr->nData);
    }

    Ein  = nnet_predict(nnet, train);
    Eout = nnet_predict(nnet, valid);

    Eouts.push_back(Eout);

    float trainAcc = 1.0f - (float) Ein / nTrain;
    float validAcc = 1.0f - (float) Eouts[epoch] / nValid;

    if (save_model_per_epoch)
      nnet.save(model_out + "." + to_string(epoch));

    printf("|%4lu   | %6.2f %% |  %7lu     | %6.2f %% |  %7lu     |  %8.2f |\n",
      epoch, trainAcc * 100, nTrain - Ein, validAcc * 100, nValid - Eouts[epoch],
      etimer.getTime() / 1000);

    if (validAcc > nnet.getConfig().minValidAccuracy &&
	isEoutStopDecrease(Eouts, epoch, nnet.getConfig().nNonIncEpoch))
      break;
  }

  // Show Summary
  printf("|_______|__________|______________|__________|______________|___________|\n");
  printf("\n%ld epochs in total\n", epoch);
  timer.elapsed();

  printf("[   In-Sample   ] ");
  showAccuracy(Ein, train.size());
  printf("[ Out-of-Sample ] ");
  showAccuracy(Eouts.back(), valid.size());
}

size_t nnet_predict(NNet& nnet, DataSet& data) {

  const size_t batchSize = 256;
  size_t nError = 0;

  nnet.setDropout(false);
  Batches batches(batchSize, data.size());
  for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
    auto d = data[itr];
    mat x = ~mat(d.x);
    mat prob = nnet.feedForward(x);
    nError += zeroOneError(prob, d.y);
  }
  nnet.setDropout(true);

  return nError;
}

bool isEoutStopDecrease(const std::vector<size_t> Eouts, size_t epoch, size_t nNonIncEpoch) {

  for (size_t i=0; i<nNonIncEpoch; ++i) {
    if (epoch - i > 0 && Eouts[epoch] > Eouts[epoch - i])
      return false;
  }

  return true;
}
