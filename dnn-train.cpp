#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
using namespace std;

std::vector<size_t> getDimensions(const DataSet& data, const string& structure);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
    .add("model_file", false);

  cmd.addGroup("Training options: ")
     .add("-v", "ratio of training set to validation set (split automatically)", "5")
     .add("--max-epoch", "number of maximum epochs", "100000")
     .add("--min-acc", "Specify the minimum cross-validation accuracy", "0.5")
     .add("--learning-rate", "learning rate in back-propagation", "0.01")
     .add("--variance", "the variance of normal distribution when initializing the weights", "0.01")
     .add("--batch-size", "number of data per mini-batch", "32")
     .add("--type", "choose one of the following:\n"
	"0 -- classfication\n"
	"1 -- regression", "0");

  cmd.addGroup("Structure of Neural Network: ")
     .add("--nodes", "specify the width(nodes) of each hidden layer seperated by \"-\":\n"
	"Ex: 1024-1024-1024 for 3 hidden layer, each with 1024 nodes. \n"
	"(Note: This does not include input and output layer)");

  /*cmd.addGroup("Pre-training options:")
     .add("--pre", "type of Pretraining. Choose one of the following:\n"
	"0 -- RBM (Restricted Boltzman Machine)\n"
	"1 -- Layer-wise", "0");*/

  cmd.addGroup("Example usage: dnn-train data/train3.dat --nodes=16-8");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string train_fn   = cmd[1];
  string model_fn   = cmd[2];
  string structure  = cmd["--nodes"];
  int ratio	    = cmd["-v"];
  size_t batchSize  = cmd["--batch-size"];
  float learningRate= cmd["--learning-rate"];
  float variance    = cmd["--variance"];
  float minValidAcc = cmd["--min-acc"];
  size_t maxEpoch   = cmd["--max-epoch"];

  if (model_fn.empty())
    model_fn = train_fn.substr(train_fn.find_last_of('/') + 1) + ".model";

  DataSet data;
  getFeature(train_fn, data);
  shuffleFeature(data);
  showSummary(data);

  ERROR_MEASURE err = CROSS_ENTROPY;
  
  DataSet train, valid;
  splitIntoTrainingAndValidationSet(train, valid, data, ratio);

  // Set configurations
  Config config;
  config.variance = variance;
  config.learningRate = learningRate;
  config.minValidAccuracy = minValidAcc;
  config.maxEpoch = maxEpoch;

  // Initialize Deep Neural Network
  DNN dnn(config);
  dnn.init(getDimensions(data, structure));

  // Start Training
  dnn.train(train, valid, batchSize, err);

  // dnn.setLearningRate(learningRate);

  // Save the model
  dnn.save(model_fn);

  return 0;
}

std::vector<size_t> getDimensions(const DataSet& data, const string& structure) {

  // Initialize hidden structure
  size_t input_dim  = data.X.getCols() - 1;
  size_t output_dim = data.prob.getCols();

  vector<size_t> dims = splitAsInt(structure, '-');
  dims.insert(dims.begin(), input_dim);
  dims.push_back(output_dim);

  return dims;
}
