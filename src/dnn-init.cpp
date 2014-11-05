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
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
#include <rbm.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
    .add("model_file", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).\n")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--nf", "Load pre-computed statistics from file", "");

  cmd.addGroup("Structure of Neural Network: ")
     .add("--nodes", "specify the width(nodes) of each hidden layer seperated by \"-\":\n"
	"Ex: 1024-1024-1024 for 3 hidden layer, each with 1024 nodes. \n"
	"(Note: This does not include input and output layer)")
     .add("--output-dim", "specify the output dimension (# of classes).\n", "0");

  cmd.addGroup("Pre-training options:")
     .add("--type", "type of Pretraining. Choose one of the following:\n"
	"0 -- Bernoulli-Bernoulli RBM\n"
	"1 -- Gaussian-Bernoulli  RBM", "0")
     .add("--slope-thres", "threshold of ratio of slope in RBM pre-training", "0.05")
     .add("--batch-size", "number of data per mini-batch", "32")
     .add("--learning-rate", "specify learning rate in constrastive divergence "
	 "algorithm", "0.1");

  cmd.addGroup("Hardward options:")
     .add("--cache", "specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: dnn-init data/train3.dat --nodes=16-8");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn   = cmd[1];
  string model_fn   = cmd[2];

  size_t input_dim  = cmd["--input-dim"];
  NormType n_type   = (NormType) (int) cmd["--normalize"];
  string n_filename = cmd["--nf"];

  string structure  = cmd["--nodes"];
  size_t output_dim = cmd["--output-dim"];

  UNIT_TYPE type  = UNIT_TYPE ((int) cmd["--type"]);
  float slopeThres    = cmd["--slope-thres"];
  float learning_rate = cmd["--learning-rate"];

  size_t cache_size   = cmd["--cache"];
  CudaMemManager<float>::setCacheSize(cache_size);

  if (model_fn.empty())
    model_fn = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  DataSet data(train_fn, input_dim, 0, n_type);
  data.showSummary();

  if (output_dim == 0)
    output_dim = StackedRbm::AskUserForOutputDimension();

  auto dims = StackedRbm::parseDimensions(input_dim, structure, output_dim);

  // Initialize using RBM
  StackedRbm srbm(type, dims, slopeThres, learning_rate);
  srbm.train(data);
  srbm.save(model_fn);

  return 0;
}
