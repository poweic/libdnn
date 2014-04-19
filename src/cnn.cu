#include <cnn.h>

ConvolutionalLayer::ConvolutionalLayer(size_t n, size_t m, size_t h, size_t w) {
  if (w == -1)
    w = h;

  assert(n > 0 && m > 0 && h > 0 && w > 0);

  _bias.resize(n);

  _kernels.resize(n);
  for (int i=0; i<n; ++i) {
    _kernels[i].assign(m, rand(h, w));
    _bias[i] = 0;
  }
}

void ConvolutionalLayer::feedForward(vector<mat>& fouts, const vector<mat>& fins) {

  int nInputs  = getNumInputMaps(),
      nOutputs = getNumOutputMaps();

  if (fins.size() != nInputs)
    throw std::runtime_error("\33[31m[Error]\33[0m Number of inputs maps ("
	+ to_string(fins.size()) + ") does not match number of kernels ("
	+ to_string(nInputs) + ").");

  fouts.resize(nOutputs);

  for (int j=0; j<nOutputs; j++) {
    Size s = get_convn_size(fins[0], _kernels[0][j], "valid");
    fouts[j].resize(s.m, s.n);
    for (int i=0; i<nInputs; ++i)
      fouts[j] += convn(fins[i], _kernels[i][j], "valid");
    fouts[j] = sigmoid(fouts[j] + _bias[j]);
  }
}

// NOTE: in MATLAB
// xcorr2 stands for 2D cross-correlation
// (I don't know why MATLAB does not have "xcorrn" for n-dimensional xcorr)
// The following operation are theoretically equivalent:
// (with only some trivial numerical error)
// (1)  convn(x, rot180(h)) == xcorr2(x, h)
//     xcorr2(x, rot180(h)) ==  convn(x, h)
// (2) convn(rot180(x), h) == rot180(convn(x, rot180(h)))
//     ^
//     |_____ which is obviously faster

void ConvolutionalLayer::backPropagate(vector<mat>& errors, const vector<mat>& fins,
    const vector<mat>& fouts, float learning_rate) {

  // FIXME How to backPropagate a BATCH of images AT A TIME.

  // In the following codes, the iteration index i and j stands for
  // i : # of input  features. i = 0 ~ M, where M = fin.size()
  // j : # of output features. j = 0 ~ N, where N = fouts.size() = errors.size()
  int M = fins.size(),
      N = fouts.size();

  // Compute delta from errors over the derivatives of sigmoid function.
  vector<mat> deltas(N);
  for (size_t j=0; j<N; ++j)
    deltas[j] = fouts[j] & ( 1.0f - fouts[j] ) & errors[j];

  // Update kernels with learning rate
  for (size_t j=0; j<N; ++j) {
    for (size_t i=0; i<M; ++i)
      _kernels[i][j] += convn(rot180(fins[i]), deltas[j], "valid") * learning_rate;

    _bias[j] += sum_all(deltas[j]) * learning_rate;
  }

  // Feed Backward
  Size s = get_convn_size(errors[0], rot180(_kernels[0][0]), "full");
  vector<mat> err(M);
  for (size_t i=0; i<M; ++i) {
    err[i].resize(s.m, s.n);

    for (size_t j=0; j<N; ++j)
      err[i] += convn(errors[j], rot180(_kernels[j][i]), "full");
  }
}

void ConvolutionalLayer::status() const {

  printf("+--------------+---------------+--------------+---------------+\n");
  printf("| # input maps | # output maps | kernel width | kernel height |\n");
  printf("+--------------+---------------+--------------+---------------+\n");
  printf("|      %-5lu   |       %-5lu   |      %-5lu   |       %-5lu   |\n",
      getNumInputMaps(), getNumOutputMaps(), getKernelWidth(), getKernelHeight());
  printf("+--------------+---------------+--------------+---------------+\n");

}

size_t ConvolutionalLayer::getKernelWidth() const {
  return _kernels[0][0].getCols();
}

size_t ConvolutionalLayer::getKernelHeight() const {
  return _kernels[0][0].getRows();
}

size_t ConvolutionalLayer::getNumInputMaps() const {
  return _kernels.size();
}

size_t ConvolutionalLayer::getNumOutputMaps() const {
  return _kernels[0].size();
}

SubSamplingLayer::SubSamplingLayer(size_t scale): _scale(scale) {
}

void SubSamplingLayer::feedForward(vector<mat>& fouts, const vector<mat>& fins) {
  fouts.resize(fins.size());

  for (size_t i=0; i<fins.size(); ++i)
    fouts[i] = downsample(fins[i], _scale);
}

void SubSamplingLayer::backPropagate(vector<mat>& errors, const vector<mat>& fins,
    const vector<mat>& fouts, float learning_rate) {

  // FIXME A downsampling followed by upsampling may NOT give the same dimension.
  for (size_t i=0; i<errors.size(); ++i)
    errors[i] = upsample(errors[i], _scale);
}
