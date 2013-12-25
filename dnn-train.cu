#include <iostream>
#include <string>
#include <dnn.h>
#include <dnn-utility.h>
using namespace std;

void dnn_train(mat& data, mat& labels, string model_fn);
void getDataAndLabels(string train_fn, mat& data, mat& labels);

void showUsageAndExit() {
  printf("Usage: dnn-train [options] training_set_file [model_file]\n");
  exit(-1);
}

int main (int argc, char* argv[]) {

  if (argc < 2)
    showUsageAndExit();

  string train_fn(argv[1]);
  string model_fn = (argc < 3) ? (train_fn + ".model") : argv[2];

  mat data, labels;

  getDataAndLabels(train_fn, data, labels);

  showSummary(data, labels);

  dnn_train(data, labels, model_fn);

  return 0;
}

void dnn_train(mat& data, mat& labels, string model_fn) {

  zeroOneLabels(labels);

  size_t input_dim = data.getCols();
  size_t output_dim = labels.getCols();

  vector<size_t> dims(4);
  dims[0] = input_dim;
  dims[1] = 512;
  dims[2] = 512;
  dims.back() = output_dim;

  vector<float> coeff(data.getRows());

  DNN dnn(dims);
  vector<mat> O(dnn.getNLayer());
  std::vector<mat> gradient;

  int nIteration = 262144;
  for (int itr=0; itr<nIteration; ++itr) {
    dnn.feedForward(data, &O);
    size_t nError = zeroOneError(O.back(), labels);
    showAccuracy(nError, labels.size());
    if (nError == 0) break;

    mat error = O.back() - labels;

    dnn.getEmptyGradient(gradient);
    dnn.backPropagate(error, O, gradient, coeff);
    dnn.updateParameters(gradient, 5 * 1e-3);
  }

  dnn.save(model_fn);
}
