#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
using namespace std;

void dnn_train(DNN& dnn, mat& trainX, mat& trainY, mat& validX, mat& validY);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("training_set_file")
    .add("model_file", false);

  cmd.addGroup("Training options: ")
    .add("-v", "ratio of training set to validation set (split automatically)", "5")
    .add("--itr", "number of maximum iteration", "inf")
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

  if (model_fn.empty())
    model_fn = train_fn + ".model";

  mat data, labels;
  getFeature(train_fn, data, labels);
  zeroOneLabels(labels);

  mat trainX, trainY, validX, validY;
  splitIntoTrainingAndValidationSet(
      trainX, trainY,
      validX, validY,
      ratio,
      data, labels);

  showSummary(data, labels);

  // Initialize hidden structure
  vector<size_t> dims = splitAsInt(structure, '-');
  dims.insert(dims.begin(), data.getCols());
  dims.push_back(labels.getCols());
  DNN dnn(dims);

  // Start Training
  dnn_train(dnn, trainX, trainY, validX, validY);

  // Save the model
  dnn.save(model_fn);

  return 0;
}

void dnn_train(DNN& dnn, mat& trainX, mat& trainY, mat& validX, mat& validY) {

  printf("Training...\n");
  perf::Timer timer;
  timer.start();

  vector<mat> O(dnn.getNLayer());
  std::vector<mat> gradient;

  size_t Ein, Eout;
  size_t minEout = validY.size();
  int nIteration = 10240, itr;

  for (itr=0; itr<nIteration; ++itr) {
    cout << "."; cout.flush();

    dnn.feedForward(validX, &O);
    Eout = zeroOneError(O.back(), validY);

    dnn.feedForward(trainX, &O);

    if (Eout < minEout) {
      minEout = Eout; cout << "+";
      cout.flush();
    }

    if ((float) Eout / validY.size() < 0.2 )
      break;

    mat error = O.back() - trainY;

    dnn.getEmptyGradient(gradient);
    dnn.backPropagate(error, O, gradient);
    dnn.updateParameters(gradient, 5 * 1e-3);
  }

  // Show Summary
  printf("\n%d iteration in total\n", itr);
  timer.elapsed();

  Ein = zeroOneError(O.back(), trainY);

  printf("[   In-Sample   ] ");
  showAccuracy(Ein, trainY.size());
  printf("[ Out-of-Sample ] ");
  showAccuracy(Eout, validY.size());
  printf("[ Minimum Eout  ] ");
  showAccuracy(minEout, validY.size());
}
