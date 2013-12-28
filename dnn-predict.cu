#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
using namespace std;

void dnn_predicts(const DataSet& data, string model_fn, string output_fn);

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("testing_set_file")
    .add("model_file")
    .add("output_file", false);

  cmd.addGroup("Prediction options: ")
    .add("--itr", "number of maximum iteration", "inf")
    .add("--type", "choose one of the following:\n"
	"0 -- classfication\n"
	"1 -- regression", "0");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string test_fn = cmd[1];
  string model_fn = cmd[2];
  string output_fn = cmd[3];

  DataSet test;
  getFeature(test_fn, test.X, test.y);
  // zeroOneLabels(test.y);
  showSummary(test.X, test.y);

  reformatLabels(test.y);
  label2PosteriorProb(test.y);

  // Make predictions
  dnn_predicts(test, model_fn, output_fn);

  return 0;
}

void dnn_predicts(const DataSet& data, string model_fn, string output_fn) {

  FILE* fid = output_fn.empty() ? stdout : fopen(output_fn.c_str(), "w");

  DNN dnn(model_fn);

  vector<mat> O(dnn.getNLayer());
  dnn.feedForward(data, O);

  if (isLabeled(data.y)) {
    size_t nError = zeroOneError(O.back(), data.y, CROSS_ENTROPY);
    showAccuracy(nError, data.y.size());
  }

  if (fid != stdout)
    fclose(stdout);
}
