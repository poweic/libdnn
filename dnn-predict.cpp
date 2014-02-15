#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
#include <cmdparser.h>
#include <batch.h>
using namespace std;

void printLabels(const mat& prob, FILE* fid);
FILE* openFileOrStdout(const string& filename);

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

  // cout << "Reading data..." << endl;
  DataSet test(test_fn, rescale);
  test.showSummary();

  // cout << "Test if there's a label" << endl;
  bool hasAnswer = test.isLabeled();
  // cout << "Test done." << endl;

  ERROR_MEASURE errorMeasure = CROSS_ENTROPY;

  // cout << "Loading DNN model from " << model_fn << endl;
  DNN dnn(model_fn);
  // cout << "Model loaded" << endl;

  size_t nError = 0;

  FILE* fid = openFileOrStdout(output_fn);

  Batches batches(1024, test.size());
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    // printf("offset = %lu, nData = %lu\n", itr->offset, itr->nData);
    mat prob = dnn.feedForward(test.getX(itr->offset, itr->nData));

    if (hasAnswer)
      nError += zeroOneError(prob, test.getY(itr->offset, itr->nData), errorMeasure);

    if (hasAnswer && output_fn.empty())
      continue;

    if (isOutputProb)
      prob.print(fid);
    else
      printLabels(prob, fid);
  }

  if (fid != stdout)
    fclose(fid);

  if (hasAnswer)
    showAccuracy(nError, test.size());

  return 0;
}

void printLabels(const mat& prob, FILE* fid) {
  std::vector<float> labels = copyToHost(posteriorProb2Label(prob));
  for (size_t i=0; i<labels.size(); ++i)
    fprintf(fid, "%d\n", (int) labels[i]);
}

FILE* openFileOrStdout(const string& filename) {

  FILE* fid = filename.empty() ? stdout : fopen(filename.c_str(), "w");

  if (fid == NULL) {
    fprintf(stderr, "Failed to open output file");
    exit(-1);
  }

  return fid;
}
