#include <cnn.h>

void cnn_playground() {
  mat data = rand(32, 16);

  matlog(data);

  int N = 7;
  vector<mat> kernels(N);
  for (int i=0; i<kernels.size(); ++i) {
    kernels[i] = rand(3, 3);
    matlog(kernels[i]);
  }

  mat x = data;
  for (int i=0; i<kernels.size(); ++i) {
    x = convn(x, kernels[i], "valid");
    matlog(x);
  }
}

int main() {

  // test_downsample();
  // cnn_playground();
  
  return 0;
}
