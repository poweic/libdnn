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
	"Ex: 1024-1024-1024 for 3 hidden layer, each with 1024 nodes", "32-32");

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
    model_fn = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  DataSet data;
  getFeature(train_fn, data);
  shuffleFeature(data);
  showSummary(data);

  ERROR_MEASURE err = CROSS_ENTROPY;
  
  DataSet train, valid;
  splitIntoTrainingAndValidationSet(train, valid, data, ratio);

  // Initialize hidden structure
  size_t input_dim  = data.X.getCols() - 1;
  size_t output_dim = data.prob.getCols();

  vector<size_t> dims = splitAsInt(structure, '-');
  dims.insert(dims.begin(), input_dim);
  dims.push_back(output_dim);
  DNN dnn(dims);

  // Start Training
  dnn.train(train, valid, batchSize, err);

  // Save the model
  dnn.save(model_fn);

  return 0;
}
