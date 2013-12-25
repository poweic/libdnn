#include <dnn-utility.h>

void evaluate(DNN& dnn, mat& X, mat& y) {
  vector<mat> O(dnn.getNLayer());
  dnn.feedForward(X, &O);

  size_t nError = zeroOneError(O.back(), y);
  showAccuracy(nError, y.size());
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

void print(const std::vector<mat>& vm) {
  for (size_t i=0; i<vm.size(); ++i) {
    printf("rows = %lu, cols = %lu\n", vm[i].getRows(), vm[i].getCols());
    vm[i].print();
  }
}

void showSummary(const mat& data, const mat& label) {
  size_t input_dim  = data.getCols();
  size_t output_dim = label.getCols();
  size_t nData	    = data.getRows();

  printf("---------------------------------------------\n");
  printf("  Number of input feature (data) %10lu \n", nData);
  printf("  Dimension of  input feature    %10lu \n", input_dim);
  printf("  Dimension of output feature    %10lu \n", output_dim);
  printf("---------------------------------------------\n");
}

void showAccuracy(size_t nError, size_t nTotal) {
  size_t nCorr = nTotal - nError;
  printf("Accuracy = %.2f%% ( %lu / %lu ) \n", (float) nCorr / nTotal * 100, nCorr, nTotal);
}

void getDataAndLabels(string train_fn, mat& data, mat& labels) {

  string tmpfn = getTempFilename();

  exec("cut -f 1 -d ':' " + train_fn + " > " + tmpfn);
  labels = mat(tmpfn);

  exec("cut -f 2 -d ':' " + train_fn + " > " + tmpfn);
  data	 = mat(tmpfn);

  exec("rm " + tmpfn);
}

string getTempFilename() {
  std::stringstream ss;
  time_t seconds;
  time(&seconds);
  ss << seconds;
  return ".tmp." + ss.str();
}

void exec(string command) {
  system(command.c_str());
}
