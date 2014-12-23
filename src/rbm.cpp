// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <rbm.h>
#include <cstdlib>

ostream& operator << (ostream& os, const UNIT_TYPE& type) {
  switch (type) {
    case GAUSSIAN: os << "Gaussian"; break;
    case BERNOULLI: os << "Bernoulli"; break;
  }
  return os;
}

/* \brief class StackedRbm
 *
 * */

StackedRbm::StackedRbm(const vector<size_t>& dims)
  : _dims(dims), _max_epoch(128), _slope_thres(0.05), _learning_rate(0.1),
  _initial_momentum(0.5), _final_momentum(0.9), _l2_penalty(0.0002) {

    _weights.resize(_dims.size() - 1);
}

void StackedRbm::init() {
  for (size_t i=0; i<_weights.size(); ++i) {
    size_t in = _dims[i] + 1,
	   out = _dims[i + 1] + 1;

    _weights[i] = randn(in, out) * sqrt(0.1 / out);
    _weights[i] = ~_weights[i];
  }
}

void StackedRbm::setParams(size_t max_epoch, float slope_thres, float learning_rate,
    float initial_momentum, float final_momentum, float l2_penalty) {

  _max_epoch = max_epoch;
  _slope_thres = slope_thres;
  _learning_rate = learning_rate;
  _initial_momentum = initial_momentum;
  _final_momentum = final_momentum;
  _l2_penalty = l2_penalty;
}

void StackedRbm::printParams() const {
  printf("_max_epoch = %lu\n", _max_epoch);
  printf("_slope_thres = %f\n", _slope_thres);
  printf("_learning_rate = %f\n", _learning_rate);
  printf("_initial_momentum = %f\n", _initial_momentum);
  printf("_final_momentum = %f\n", _final_momentum);
  printf("_l2_penalty = %f\n", _l2_penalty);
}

void StackedRbm::train(DataSet& data, UNIT_TYPE vis_type) {

  // FIXME For NOW, hidden units are only allowed to be Bernoulli
  UNIT_TYPE hid_type = BERNOULLI;

  // If vis_type is Bernoulli, make sure the visible units have values in the
  // range [0, 1]. If the values scatter over a wide range and has a Gaussian
  // distribution, make sure the values are normalized to 0 mean and 1 standard
  // deviation before going into this function.
  /*if (vis_type == BERNOULLI)
    assert(ext::max(data.getX()) <= 1 && ext::min(data.getX()) >= 0);*/
  
  for (size_t i=0; i<_weights.size(); ++i) {
    this->rbm_train(data, i, vis_type, hid_type);

    vis_type = hid_type;
    hid_type = BERNOULLI;
  }
}

void StackedRbm::up_propagate(const mat& W, const mat& visible, mat& hidden, UNIT_TYPE type) {
  hidden = W * visible;

  if (type == BERNOULLI)
    hidden = sigmoid(hidden);
  
  hidden = add_bias(hidden);
}

void StackedRbm::down_propagate(const mat& W, mat& visible, const mat& hidden, UNIT_TYPE type) {
  visible = ~W * hidden;

  if (type == BERNOULLI)
    visible = sigmoid(visible);

  visible = add_bias(visible);
}

void StackedRbm::antiWeightExplosion(mat& W, const mat& v1, const mat& v2, float &learning_rate) {
  float v1_avg_std = calcAverageStandardDeviation(v1),
	v2_avg_std = calcAverageStandardDeviation(v2),
	std_ratio = v2_avg_std / v1_avg_std;

  assert(std_ratio != 0);

  if (std_ratio > 2) {
    // printf("\33[34m[Info]\33[0m W and learning_rate shrinked to prevent weights to explode!!\n");
    W /= std_ratio;
    learning_rate *= 0.9;
  }
}

float StackedRbm::getReconstructionError(DataSet& data, const mat& W,
    UNIT_TYPE vis_type, UNIT_TYPE hid_type, int layer) {

  float r_error = 0;

  const size_t batch_size = 1024;
  size_t nData = data.size();

  Batches batches(batch_size, nData);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {

    // v1 is input data, v2 is reconstructed data
    mat v1, v2, h1;

    v1 = getBatchData(data, itr, layer);
    v1 = add_bias(v1);

    // Up propagation
    up_propagate(W, v1, h1, hid_type);

    // Sampling
    sample(h1, hid_type);
    h1 = add_bias(h1);

    // Down propagation
    down_propagate(W, v2, h1, vis_type);

    r_error += pow(nrm2(v1 - v2), 2.0f);
  }

  r_error = sqrt(r_error) / nData;

  data.rewind();

  return r_error;
}

float StackedRbm::getFreeEnergy(const mat& visible, const mat& W) {
  int N = visible.getCols();
  mat hidden = W * visible;

  mat va(1, N);
  memcpy2D(va, hidden, hidden.getRows() - 1, 0, 1, N, 0, 0);
  //CCE(cudaMemcpy(va.getData(), hidden.getData() + hidden.size() - N, sizeof(float) * N, cudaMemcpyDeviceToDevice));

  hidden = add_bias(hidden, -1000.0f);

  log1pexp(hidden);

  // mat e = hidden * mat(hidden.getCols(), 1, 1) + va;
  mat e = mat(1, hidden.getRows(), 1) * hidden + va;

  return -sum_all(e) / N;
}

float StackedRbm::getFreeEnergyGap(DataSet& data, size_t batch_size, const mat& W, int layer) {

  size_t nData = data.size();
  Batches batches(batch_size, nData);
  Batches::iterator ii = batches.begin();

  float fe1 = getFreeEnergy(getBatchData(data, ii, layer), W),
	fe2 = getFreeEnergy(getBatchData(data, ii+1, layer), W);

  data.rewind();

  return abs(fe1 - fe2);
}

mat StackedRbm::getBatchData(DataSet& data, const Batches::iterator& itr, int layer) {
  mat x = (mat) data[itr].x;
  x = ~x;
  for (int i=0; i<layer; ++i)
    x = sigmoid(_weights[i] * x);
  return x;
}

void StackedRbm::rbm_train(DataSet& data, int layer, UNIT_TYPE vis_type, UNIT_TYPE hid_type) {

  clog << "Training \33[34m" << vis_type << " - " << hid_type << "\33[0m RBM ..." << endl;

  // Note: The learning rate of Gaussian RBM needs to be about one or two
  // orders of magnitude smaller than when using binary visible units.
  // Otherwise value will explode very quickly and get NaN.
  // [cf. A Practical Guide to Training Restricted Boltzmann Machines]

  float lr = _learning_rate;

  if (vis_type == GAUSSIAN) lr *= 0.01;
  if (hid_type == GAUSSIAN) lr *= 0.01;

  size_t nData = data.size();
  size_t batch_size = nData < 1024 ? (nData / 10) : 1024;

  mat &W = _weights[layer];
  mat dW(W.getRows(), W.getCols(), 0);

  const size_t minEpoch = 5;

  vector<float> errors;

  float initialSlope = 0;

  ProgressBar pBar;
  if (_max_epoch != 0)
    pBar = ProgressBar("( | Δ free energy | = ...     , reconstruction error = ...      )");

  perf::Timer timer;
  timer.start();

  size_t epoch;
  for (epoch=0; epoch < _max_epoch; ++epoch) {

    float momentum = (epoch <= 5) ? _initial_momentum : _final_momentum;

    Batches batches(batch_size, nData);
    for (Batches::iterator itr = batches.begin() + 1; itr != batches.end(); ++itr) {
      // The first batch is kept as held-out set for validation. Therefore
      // itr starts from begin() + 1 rather than begin().

      mat v1, v2, h1, h2;

      v1 = getBatchData(data, itr, layer);
      v1 = add_bias(v1);

      // Up propagation
      up_propagate(W, v1, h1, hid_type);

      // Calculate positive
      mat positive = h1 * ~v1;

      // Sampling
      sample(h1, hid_type);
      h1 = add_bias(h1);

      // Down-and-Up propagation
      down_propagate(W, v2, h1, vis_type);
      up_propagate(W, v2, h2, hid_type);

      // Calculate negative
      mat negative = h2 * ~v2;

      // Prevent weight explosion (cf. "kaldi-trunk/src/nnet/nnet-rbm.cc")
      antiWeightExplosion(W, v1, v2, lr);

      dW = dW * momentum				// momentum
	+ (lr / batch_size) * (positive - negative)	// gradient of CD
	- (lr * _l2_penalty) * W;			// gradient of L2-penalty

      W += dW;
    }

    float fe_gap = getFreeEnergyGap(data, batch_size, W, layer);
    float error = getReconstructionError(data, W, vis_type, hid_type, layer);
    errors.push_back(error);

    if (epoch == minEpoch)
      initialSlope = getSlope(errors, minEpoch);

    if (epoch > minEpoch) {
      float ratio = abs(getSlope(errors, minEpoch) / initialSlope);
      float percentage = (epoch == _max_epoch - 1) ? 1.0 : std::min(1.0f, _slope_thres / ratio);

      char status[100];
      sprintf(status, "( | Δ free energy | = %.2e, reconstruction error = %.2e )", fe_gap, error);

      pBar.refresh(percentage, status);

      if (ratio < _slope_thres)
	break;
    }
  }

  float t_end = timer.getTime();
  printf("Average magnitude of elements in weight W = %.7f\n", nrm2(W) / sqrt(W.size()));
  printf("# of epoch = %lu. Average time for each epoch = %f (seconds)\n\n", epoch, t_end / epoch / 1000);
}

void StackedRbm::save(const string& fn) {

  ofstream fout(fn.c_str());

  if (!fout.is_open())
    throw std::runtime_error("Cannot open file: \"" + fn + "\"");

  FeatureTransform *affine, *activation;
  size_t dim;

  for (size_t i=0; i<_weights.size(); ++i) {

    if (_weights[i].size() > 0)
      affine = new AffineTransform(_weights[i]);
    else
      affine = new AffineTransform(_dims[i], _dims[i+1]);
    fout << affine;

    dim = affine->getOutputDimension();

    if (i < _weights.size() - 1)
      activation = new Sigmoid(dim, dim);
    else
      activation = new Softmax(dim, dim);

    fout << activation;

    delete affine;
    delete activation;
  }

  fout.close();

  printf("\33[34m[Info]\33[0m Model saved to \33[32m%s\33[0m\n", fn.c_str());
}

vector<size_t> StackedRbm::parseDimensions(
    size_t input_dim, 
    const string& hidden_structure, 
    size_t output_dim) {

  // ===========================================================================
  // Initialize hidden structure
  vector<size_t> dims = splitAsInt(hidden_structure, '-');
  dims.insert(dims.begin(), input_dim);
  dims.push_back((size_t) output_dim);

  printf("\n");
  printf("\33[32m Start RBM pre-training with following hidden structure:\33[0m\n");
  printf("\33[34m [   input  layer  ]\33[0m %lu\n", dims[0]);
  for (size_t i=1; i<dims.size()-1; ++i)
    printf("\33[34m [ hidden layer #%-2lu]\33[0m %lu\n", i, dims[i]);
  printf("\33[34m [   output layer  ]\33[0m %lu\n\n", dims.back());
  // ===========================================================================

  return dims;
}

// Calculuate standard deviation of each dimension of x.
// After that, average over all standard deviations.
float calcAverageStandardDeviation(const mat& x) {
  size_t rows = x.getRows(),
	 cols = x.getCols();

  mat x_minus_mean = x - (mat(rows, rows, 1) * x) / rows;
  mat sum_of_squares = mat(1, rows, 1) * (x_minus_mean & x_minus_mean);

  hmat squares(1, cols);
  CCE(cudaMemcpy(squares.getData(), sum_of_squares.getData(), sizeof(float) * squares.size(), cudaMemcpyDeviceToHost));

  int N = (rows == 1) ? 1 : rows - 1;

  float s = 0;
  for (size_t i=0; i<cols; ++i)
    s += sqrt(squares[i] / N);
  s /= cols;

  return s;
}

float getSlope(const vector<float> &seq, size_t N) {
  vector<float> x(N);
  for (size_t i=0; i<N; ++i)
    x[i] = N - 1 - i;

  vector<float> y(N);
  for (size_t i=seq.size() - N; i<seq.size(); ++i)
    y[i - (seq.size() - N)] = seq[i];

  float m, c;
  linearRegression(x, y, &m, &c);

  return m;
}

float getAsymptoticBound(const vector<float> &error, size_t epoch, size_t maxEpoch, size_t N) {
  vector<float> x(N);
  for (size_t i=0; i<N; ++i)
    x[i] = epoch - (N - 1 - i);

  vector<float> y(N);
  for (size_t i=error.size() - N; i<error.size(); ++i)
    y[i - (error.size() - N)] = error[i];

  float m, c;
  linearRegression(x, y, &m, &c);

  return m * (float) maxEpoch + c;
}
