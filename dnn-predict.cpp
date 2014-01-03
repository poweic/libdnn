#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
using namespace std;

int main (int argc, char* argv[]) {

  CmdParser cmd(argc, argv);

  cmd.add("testing_set_file")
    .add("model_file")
    .add("output_file", false);

  cmd.addGroup("Options:")
    .add("--rescale", "Rescale each feature to [0, 1]", "false")
    .add("--prob", "output posterior probabilities if true", "false");

  cmd.addGroup("Example usage: dnn-predict test3.dat train3.dat.model");

  if (!cmd.isOptionLegal())
    cmd.showUsageAndExit();

  string test_fn    = cmd[1];
  string model_fn   = cmd[2];
  string output_fn  = cmd[3];
  bool rescale      = cmd["--rescale"];
  bool isOutputProb = cmd["--prob"];

  DataSet test;
  getFeature(test_fn, test, rescale);

  showSummary(test);

  DNN dnn(model_fn);
  mat prob = dnn.predict(test);

  ERROR_MEASURE errorMeasure = CROSS_ENTROPY;

  bool hasAnswer = isLabeled(test.y);

  if (hasAnswer) {
    size_t nError = zeroOneError(prob, test.y, errorMeasure);
    showAccuracy(nError, test.y.size());
  }

  if (hasAnswer && output_fn.empty())
    return 0;

  FILE* fid = output_fn.empty() ? stdout : fopen(output_fn.c_str(), "w");
  if (fid == NULL) {
    fprintf(stderr, "Failed to open output file");
    return -1;
  }

  mat predictions;
  if (!isOutputProb)
    predictions = posteriorProb2Label(prob);
  else
    predictions = prob;

  predictions.print(fid);

  if (fid != stdout)
    fclose(fid);

  return 0;
}

