#include <cnn.h>

int main() {

  vector<mat> data(3);
  for (size_t i=0; i<data.size(); ++i)
    data[i] = randn(100, 100);

  ConvolutionalLayer cl(3, 8, 5, 5);
  cl.status();

  vector<mat> outputs;
  cl.feedForward(outputs, data);

  for (const auto& o : outputs) {
    o.print();
    cout << endl;
  }

  return 0;
}
