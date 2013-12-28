#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
    .add("model_file", false);

  cmd.addGroup("Training options: ")
    .add("-v", "ratio of training set to validation set (split automatically)", "5")
    .add("--epoch", "number of maximum epochs", "inf")
    .add("--batch-size", "number of data per mini-batch", "32")
    .add("--type", "choose one of the following:\n"
	"0 -- classfication\n"
	"1 -- regression", "0");

  cmd.addGroup("Structure of Neural Network: ")
    .add("--hidden-struct", "specify the width of each hidden layer seperated by \"-\":\n"
	"Ex: 1024-1024-1024 for 3 hidden layer, each with 1024 nodes");

  cmd.addGroup("Pre-training options:")
    .add("--pre", "type of Pretraining. Choose one of the following:\n"
	"0 -- RBM (Restricted Boltzman Machine)\n"
	"1 -- Layer-wise", "0");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn   = cmd[1];
  string model_fn   = cmd[2];
  string structure  = cmd["--hidden-struct"];
  int ratio	    = cmd["-v"];
  size_t batchSize  = cmd["--batch-size"];

  if (model_fn.empty())
    model_fn = train_fn + ".model";

  mat data, labels;
  getFeature(train_fn, data, labels);

  showSummary(data, labels);

  ERROR_MEASURE err = CROSS_ENTROPY;
  // ERROR_MEASURE err = L2ERROR;
  if (err == CROSS_ENTROPY) {
    reformatLabels(labels);
    label2PosteriorProb(labels);
  }
  else
    zeroOneLabels(labels);

  DataSet train, valid;
  splitIntoTrainingAndValidationSet(
      train.X, train.y,
      valid.X, valid.y,
      ratio,
      data, labels);

  // Initialize hidden structure
  vector<size_t> dims = splitAsInt(structure, '-');
  dims.insert(dims.begin(), data.getCols() - 1);
  dims.push_back(labels.getCols());
  DNN dnn(dims);

  // Start Training
  dnn.train(train, valid, batchSize, err);

  // Save the model
  dnn.save(model_fn);

  return 0;
}
