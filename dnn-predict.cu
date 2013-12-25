#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
using namespace std;

void dnn_predicts(mat& data, mat& labels, string model_fn, string output_fn);
void showUsageAndExit();

int main (int argc, char* argv[]) {

  if (argc < 3)
    showUsageAndExit();

  string test_fn(argv[1]);
  string model_fn(argv[2]);
  string output_fn(argc < 4 ? "" : argv[3]);

  mat data, labels;
  getDataAndLabels(test_fn, data, labels);
  showSummary(data, labels);

  // Make predictions
  dnn_predicts(data, labels, model_fn, output_fn);

  return 0;
}

void dnn_predicts(mat& data, mat& labels, string model_fn, string output_fn) {

  FILE* fid = output_fn.empty() ? stdout : fopen(output_fn.c_str(), "w");

  zeroOneLabels(labels);

  DNN dnn(model_fn);
  evaluate(dnn, data, labels);

  if (fid != stdout)
    fclose(stdout);
}

void showUsageAndExit() {
  printf("Usage: dnn-predict [options] test_file model_file output_file\n");
  exit(-1);
}
