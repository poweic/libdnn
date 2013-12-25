#include <iostream>
#include <string>
#include <dnn.h>
using namespace std;

void dnn_train(string train_fn, string label_fn, string model_fn);
void print(const std::vector<mat>& vm);
void evaluate(DNN& dnn, mat& X, mat& y);
void zeroOneLabels(const mat& label);
size_t zeroOneError(const mat& predict, const mat& label);
void showAccuracy(size_t nError, size_t nTotal);

int main (int argc, char* argv[]) {

  if (argc < 3)
    return -1;

  string train_fn(argv[1]);
  string label_fn(argv[2]);
  string model_fn = (argc < 4) ? (train_fn + ".model") : argv[3];

  dnn_train(train_fn, label_fn, model_fn);

  return 0;
}

void dnn_train(string train_fn, string label_fn, string model_fn) {

  mat data(train_fn);
  mat label(label_fn);

  zeroOneLabels(label);

  size_t input_dim  = data.getCols();
  size_t output_dim = label.getCols();
  size_t nData	    = data.getRows();

  printf("---------------------------------------------\n");
  printf("  Number of input feature (data) %10lu \n", nData);
  printf("  Dimension of  input feature    %10lu \n", input_dim);
  printf("  Dimension of output feature    %10lu \n", output_dim);
  printf("---------------------------------------------\n");

  vector<size_t> dims(10);
  dims[0] = input_dim;
  dims[1] = 1024;
  dims[2] = 1024;
  dims[3] = 512;
  dims[4] = 512;
  dims[5] = 256;
  dims[6] = 256;
  dims[7] = 128;
  dims[8] = 64;
  dims[9] = 32;
  dims.back() = output_dim;

  vector<float> coeff(data.getRows());

  DNN dnn(dims);
  vector<mat> O(dnn.getNLayer());
  std::vector<mat> gradient;

  int nIteration = 262144;
  for (int itr=0; itr<nIteration; ++itr) {
    dnn.feedForward(data, &O);
    size_t nError = zeroOneError(O.back(), label);
    showAccuracy(nError, label.size());
    if (nError == 0) break;

    mat error = O.back() - label;

    dnn.getEmptyGradient(gradient);
    dnn.backPropagate(error, O, gradient, coeff);
    dnn.updateParameters(gradient, 5 * 1e-3);
  }

  dnn.save(model_fn);
}

void showAccuracy(size_t nError, size_t nTotal) {
  size_t nCorr = nTotal - nError;
  printf("Accuracy = %.2f%% ( %lu / %lu ) \n", (float) nCorr / nTotal * 100, nCorr, nTotal);
}

void dnn_predicts() {
  // DNN dnn("dnn.model");
  // evaluate(dnn, data, label);
  // return;
}

void zeroOneLabels(const mat& label) {
  thrust::device_ptr<float> dptr(label.getData());
  thrust::host_vector<float> y(dptr, dptr + label.size());

  map<float, bool> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[y[i]] = true;

  size_t nClasses = classes.size();
  assert(nClasses == 2);

  thrust::replace(dptr, dptr + label.size(), -1, 0);
}

size_t zeroOneError(const mat& predict, const mat& label) {
  assert(predict.size() == label.size());

  size_t L = label.size();
  thrust::device_ptr<float> l_ptr(label.getData());
  thrust::device_ptr<float> p_ptr(predict.getData());

  thrust::device_vector<float> p_vec(L);
  thrust::transform(p_ptr, p_ptr + L, p_vec.begin(), func::to_zero_one<float>());

  float nError = thrust::inner_product(p_vec.begin(), p_vec.end(), l_ptr, 0.0, thrust::plus<float>(), thrust::not_equal_to<float>());
  return nError;
}

void evaluate(DNN& dnn, mat& X, mat& y) {
  vector<mat> O(dnn.getNLayer());
  dnn.feedForward(X, &O);

  size_t nError = zeroOneError(O.back(), y);
  showAccuracy(nError, y.size());
}

void print(const std::vector<mat>& vm) {
  for (size_t i=0; i<vm.size(); ++i) {
    printf("rows = %lu, cols = %lu\n", vm[i].getRows(), vm[i].getCols());
    vm[i].print();
  }
}
