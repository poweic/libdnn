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
using namespace std;

size_t AskUserForOutputDimension();

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("train_set_file", false);

  cmd.addGroup("General options:")
     .add("--type", "type of Pretraining. Choose one of the following:\n"
	"0 -- Don't train. Create neural network topology only.\n"
	"1 -- Bernoulli-Bernoulli RBM (not for CNN)\n"
	"2 -- Gaussian-Bernoulli  RBM (not for CNN)", "0")
     .add("-o", "Output model filename");

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--nf", "Load pre-computed statistics from file", "");

  cmd.addGroup("Structure of Neural Network: ")
     .add("--struct",
      "Specify the structure of Convolutional neural network\n"
      "For example: --struct=9x5x5-3s-4x3x3-2s-256-128\n"
      "\"9x5x5-3s\" means a convolutional layer consists of 9 output feature maps\n"
      "with a 5x5 kernel, which is followed by a sub-sampling layer with scale\n"
      "of 3. After \"9x5x5-3s-4x3x3-2s\", a neural network of of 2 hidden layers\n"
      "of width 256 and 128 is appended to it. Each layer should be seperated\n"
      "by a hyphen \"-\".\n"
      "(Note: This does not include input and output layer)")
     .add("--output-dim", "specify the output dimension (# of classes).", "0");

  cmd.addGroup("Pre-training options:")
     .add("--batch-size", "number of data per mini-batch", "32")
     .add("--max-epoch", "number of maximum epochs", "128")
     .add("--slope-thres", "threshold of ratio of slope in RBM pre-training", "0.05")
     .add("--learning-rate", "specify learning rate in constrastive divergence "
	 "algorithm", "0.1")
     .add("--init-momentum", "initial momentum.", "0.5")
     .add("--final-momentum", "final momentum.", "0.9")
     .add("--l2-penalty", "L2 penalty", "0.0002");

  cmd.addGroup("Hardward options:")
     .add("--card-id", "Specify which GPU card to use", "0")
     .add("--cache", "specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: nn-init train.dat --input-dim 123 -o init.xml --struct=16-8");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn = cmd[1];

  int type        = cmd["--type"];
  string model_fn = cmd["-o"];

  string structure  = cmd["--struct"];
  size_t output_dim = cmd["--output-dim"];

  size_t card_id    = cmd["--card-id"];
  size_t cache_size = cmd["--cache"];
  CudaMemManager<float>::setCacheSize(cache_size);
  SetGpuCardId(card_id);

  if (output_dim == 0)
    output_dim = AskUserForOutputDimension();

  // If it's convolutional neural network, there's a "x" in string structure
  if (structure.find("x") != string::npos) {

    string input_dim = cmd["--input-dim"];

    // If there's only one 'x' in --input-dim like 64x64, change it to 1x64x64
    if (std::count(input_dim.begin(), input_dim.end(), 'x') == 1)
      input_dim = "1x" + input_dim;

    NNet nnet;
    nnet.init(input_dim + "-" + structure + "-" + to_string(output_dim));
    nnet.save(model_fn);

  }
  else {  // Codes for RBM pre-training

    size_t input_dim  = cmd["--input-dim"];
    NormType n_type   = (NormType) (int) cmd["--normalize"];

    size_t max_epoch     = cmd["--max-epoch"];
    float slope_thres    = cmd["--slope-thres"];
    float learning_rate  = cmd["--learning-rate"];
    float init_momentum  = cmd["--init-momentum"];
    float final_momentum = cmd["--final-momentum"];
    float l2_penalty     = cmd["--l2-penalty"];

    auto dims = StackedRbm::parseDimensions(input_dim, structure, output_dim);

    StackedRbm srbm(dims);
    srbm.setParams(max_epoch, slope_thres, learning_rate,
	init_momentum, final_momentum, l2_penalty);

    // Run RBM pre-training,
    // otherwise just save randomly initialized result to file.
    if (type != 0) {
      DataSet data(train_fn, input_dim, 0, n_type);
      data.showSummary();

      srbm.init();
      srbm.train(data, (UNIT_TYPE) type);
    }

    srbm.save(model_fn);

  }

  return 0;
}

// Show a dialogue and ask user for the output dimension
size_t AskUserForOutputDimension() {
  string userInput = "";

  while (!is_number(userInput)) {
    printf("\33[33m Enter how many nodes you want in the output layer.\33[0m "
	   "[      ]\b\b\b\b\b");
    cin >> userInput;
  }

  return atoi(userInput.c_str());
}
