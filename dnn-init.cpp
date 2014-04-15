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

  cmd.addGroup("Structure of Neural Network: ")
     .add("--nodes", "specify the width(nodes) of each hidden layer seperated by \"-\":\n"
	"Ex: 1024-1024-1024 for 3 hidden layer, each with 1024 nodes. \n"
	"(Note: This does not include input and output layer)");

  cmd.addGroup("Pre-training options:")
     .add("--type", "type of Pretraining. Choose one of the following:\n"
	"0 -- Gaussian-Bernoulli  RBM\n"
	"1 -- Bernoulli-Bernoulli RBM", "0")
     .add("--normalize", "Feature normalization: \n"
	"0 -- Do not normalize.\n"
	"1 -- Rescale each dimension to [0, 1] respectively.\n"
	"2 -- Normalize to standard score. z = (x-u)/sigma .", "0")
     .add("--slope-thres", "threshold of ratio of slope in RBM pre-training", "0.05")
     .add("--batch-size", "number of data per mini-batch", "32");

  cmd.addGroup("Example usage: dnn-init data/train3.dat --nodes=16-8");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn   = cmd[1];
  string model_fn   = cmd[2];
  string structure  = cmd["--nodes"];

  RBM_TYPE type	    = RBM_TYPE ((int) cmd["--type"]);
  size_t batchSize  = cmd["--batch-size"];
  float slopeThres  = cmd["--slope-thres"];
  int n_type	    = cmd["--normalize"];

  if (model_fn.empty())
    model_fn = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  DataSet data(train_fn);
  data.normalize(n_type);
  data.shuffle();
  data.showSummary();

  auto dims = getDimensionsForRBM(data, structure);

  // Initialize by RBM
  auto weights = initStackedRBM(data, dims, slopeThres, type);

  FILE* fid = fopen(model_fn.c_str(), "w");

  for (size_t i=0; i<weights.size() - 1; ++i)
    FeatureTransform::print(fid, weights[i], "sigmoid");
  FeatureTransform::print(fid, weights.back(), "softmax");

  fclose(fid);

  return 0;
}
