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
#include <batch.h>
using namespace std;

mat getPriorProbability(const string& fn);
void printLabels(const mat& prob, FILE* fid, int base);
FILE* openFileOrStdout(const string& filename);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("testing_set_file")
    .add("model_file")
    .add("output_file", false);

  cmd.addGroup("Feature options:")
     .add("--input-dim", "specify the input dimension (dimension of feature).")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--nf", "Load pre-computed statistics from file", "")
     .add("--base", "Label id starts from 0 or 1 ?", "0");

  cmd.addGroup("Options:")
    .add("--acc", "calculate prediction accuracy", "true")
    .add("--prior", "prior probability for each classes.", "")
    .add("--output", "output posterior probabilities if true\n"
	"0 -- Do not output posterior probabilities. Output class-id.\n"
	"1 -- Output posterior probabilities. (range in [0, 1]) \n"
	"2 -- Output natural log of posterior probabilities. (range in [-inf, 0])", "0")
    .add("--silent", "Suppress all log messages", "false");

  cmd.addGroup("Hardward options:")
     .add("--cache", "specify cache size (in MB) in GPU used by cuda matrix.", "16");

  cmd.addGroup("Example usage: dnn-predict test3.dat train3.dat.model");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string test_fn    = cmd[1];
  string model_fn   = cmd[2];
  string output_fn  = cmd[3];

  NormType n_type   = (NormType) (int) cmd["--normalize"];
  string n_filename = cmd["--nf"];
  int base	    = cmd["--base"];
  string prior_fn   = cmd["--prior"];

  int output_type   = cmd["--output"];
  bool silent	    = cmd["--silent"];
  bool calcAcc	    = cmd["--acc"];

  size_t cache_size   = cmd["--cache"];
  CudaMemManager<float>::setCacheSize(cache_size);

  // Parse Input dimension.
  size_t input_dim  = parseInputDimension((string) cmd["--input-dim"]);
  
  // Use Log(prior) because log(x) would check whether x has zero in it.
  mat log_prior = log(getPriorProbability(prior_fn));
  bool prior = !prior_fn.empty();

  // Load data from file
  DataSet test(test_fn, input_dim, base, n_type);

  // Load model from file
  NNet nnet(model_fn);
  if (!silent)
    nnet.status();

  size_t nError = 0;

  FILE* fid = openFileOrStdout(output_fn);

  mat log_priors;
  Batches batches(256, test.size());
  for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
    auto data = test[itr];
    mat x = ~mat(data.x);

    mat prob = nnet.feedForward(x);

    if (calcAcc && !silent)
      nError += zeroOneError(prob, data.y);

    if (calcAcc && output_fn.empty() && output_type == 0)
      continue;

    if (prior && log_priors.getRows() != prob.getRows())
      log_priors = mat(prob.getRows(), 1, 1) * log_prior;

    switch (output_type) {
      case 0:
	printLabels(prob, fid, base);
	break;

      case 1:
	if (prior)
	  prob = exp(log(prob) - log_priors);

	prob.print(fid, 7);
	break;

      case 2:
	prob = log(prob);
	if (prior)
	  prob -= log_priors;

	prob.print(fid, 7);
	break;

      default:
	throw std::runtime_error(RED_ERROR + "unknown output type " + to_string(output_type));
    }
  }

  if (fid != stdout)
    fclose(fid);

  if (calcAcc && !silent)
    showAccuracy(nError, test.size());

  return 0;
}

mat getPriorProbability(const string& fn) {
  if (fn.empty())
    return mat();

  clog << util::blue("[INFO] ") << "Load prior prob from: " << util::green(fn) << endl;

  mat prior(fn);
  double sum = ((hmat) (prior * mat(prior.getCols(), 1, 1)))[0];
  return prior / sum;
}

// DO NOT USE device_matrix::print()
// (since labels should be printed as integer not as floating point)
void printLabels(const mat& prob, FILE* fid, int base) {
  auto h_labels = copyToHost(posteriorProb2Label(prob));
  for (size_t i=0; i<h_labels.size(); ++i)
    fprintf(fid, "%d\n", (int) h_labels[i] + base);
}

FILE* openFileOrStdout(const string& filename) {

  FILE* fid = filename.empty() ? stdout : fopen(filename.c_str(), "w");

  if (fid == NULL) {
    fprintf(stderr, "Failed to open output file");
    exit(-1);
  }

  return fid;
}
