#include <iostream>
#include <string>
#include <color.h>
#include <dnn.h>
using namespace std;

void dnn_test();
void print(const std::vector<mat>& vm);

int main (int argc, char* argv[]) {

  dnn_test();
  return 0;
}

void dnn_test() {

  vector<string> files;
  ext::load(files, "data/autism.txt");

  mat data(files[0]);
  mat target(files[1]);
  //mat data("data/data.mat");
  //mat target("data/target.mat");
  
  size_t input_dim  = data.getCols();
  size_t output_dim = target.getCols();
  size_t nData	    = data.getRows();

  printf("---------------------------------------------\n");
  printf("  Number of input feature (data) %10lu \n", nData);
  printf("  Dimension of  input feature    %10lu \n", input_dim);
  printf("  Dimension of output feature    %10lu \n", output_dim);
  printf("---------------------------------------------\n");

  vector<size_t> dims(4);
  dims[0] = input_dim; dims[1] = 256; dims[2] = 256; dims[3] = output_dim;

  vector<float> coeff = ext::randn<float>(data.getRows());

  DNN dnn(dims);
  // DNN dnn("dnn.model");
  vector<mat> O(dnn.getNLayer());
  std::vector<mat> gradient;

  for (int itr=0; itr<2; ++itr) {
    cout << "iteration " << itr << endl;
    dnn.feedForward(data, &O);

    // print(O);

    mat error = target - O.back();
    for (size_t i=0; i<error.getRows(); ++i)
      for (size_t j=0; j<error.getCols(); ++j)
	error[i][j] = 0.5 * pow(error[i][j], 2.0);

    dnn.getEmptyGradient(gradient);
    dnn.backPropagate(error, O, gradient, coeff);
    dnn.updateParameters(gradient, 1e-4);
  }

  dnn.save("dnn.model");
}

void print(const std::vector<mat>& vm) {
  for (size_t i=0; i<vm.size(); ++i) {
    printf("rows = %lu, cols = %lu\n", vm[i].getRows(), vm[i].getCols());
    vm[i].print(3);
  }
}
