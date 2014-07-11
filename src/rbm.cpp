#include <rbm.h>
#include <cstdlib>

const float StackedRbmTrainer::initial_momentum = 0.5;
const float StackedRbmTrainer::final_momentum = 0.9;
const float StackedRbmTrainer::L2_penalty = 0.0002;

ostream& operator << (ostream& os, const UNIT_TYPE& type) {
  switch (type) {
    case GAUSSIAN: os << "Gaussian"; break;
    case BERNOULLI: os << "Bernoulli"; break;
  }
  return os;
}

void StackedRbmTrainer::train(DataSet& data) {

  _weights.resize(_dims.size() - 1);

  size_t nData = data.size();

  // FIXME For NOW, hidden units are only allowed to be Bernoulli
  UNIT_TYPE vis_type = _vis_type, hid_type = BERNOULLI;

  // If vis_type is Bernoulli, make sure the visible units have values in the
  // range [0, 1]. If the values scatter over a wide range and has a Gaussian
  // distribution, make sure the values are normalized to 0 mean and 1 standard
  // deviation before going into this function.
  /*if (vis_type == BERNOULLI)
    assert(ext::max(data.getX()) <= 1 && ext::min(data.getX()) >= 0);*/
  
  for (size_t i=0; i<_weights.size(); ++i) {

    _weights[i].resize(_dims[i] + 1, _dims[i + 1] + 1);

    this->rbm_train(data, i, vis_type, hid_type);

    vis_type = hid_type;
    hid_type = BERNOULLI;
  }
}

void StackedRbmTrainer::up_propagate(const mat& W, const mat& visible, mat& hidden, UNIT_TYPE type) {
  hidden = visible * W;

  if (type == BERNOULLI)
    hidden = sigmoid(hidden);
  
  fill_bias(hidden);
}

void StackedRbmTrainer::down_propagate(const mat& W, mat& visible, const mat& hidden, UNIT_TYPE type) {
  visible = hidden * ~W;

  if (type == BERNOULLI)
    visible = sigmoid(visible);

  fill_bias(visible);
}

void StackedRbmTrainer::antiWeightExplosion(mat& W, const mat& v1, const mat& v2, float &learning_rate) {
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

float StackedRbmTrainer::getReconstructionError(DataSet& data, const mat& W
    , UNIT_TYPE vis_type, UNIT_TYPE hid_type, int layer) {

  float r_error = 0;

  const size_t batch_size = 1024;
  size_t nData = data.size();

  Batches batches(batch_size, nData);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {

    // v1 is input data, v2 is reconstructed data
    mat v1, v2, h1;

    v1 = getBatchData(data, itr, layer);
    // v1 = data.getX(*itr);
    fill_bias(v1);

    // Up propagation
    up_propagate(W, v1, h1, hid_type);

    // Sampling
    sample(h1, hid_type);

    // Down propagation
    down_propagate(W, v2, h1, vis_type);

    r_error += pow(nrm2(v1 - v2), 2.0f);
  }

  r_error = sqrt(r_error) / nData;

  data.getDataStream().rewind();

  return r_error;
}

float StackedRbmTrainer::getFreeEnergy(const mat& visible, const mat& W) {
  int N = visible.getRows();
  mat hidden = visible * W;

  mat va(N, 1);
  CCE(cudaMemcpy(va.getData(),
	hidden.getData() + hidden.size() - N,
	sizeof(float) * N, cudaMemcpyDeviceToDevice));

  fillLastColumnWith(hidden, -1000.0f);

  log1pexp(hidden);

  mat e = hidden * mat(hidden.getCols(), 1, 1) + va;
  mat sum_of_e = mat(1, N, 1) * e;

  float free_energy = 0;
  CCE(cudaMemcpy(&free_energy, sum_of_e.getData(), sizeof(float), cudaMemcpyDeviceToHost));

  free_energy = - free_energy / N;

  return free_energy;
}

float StackedRbmTrainer::getFreeEnergyGap(DataSet& data, size_t batch_size, const mat& W, int layer) {

  size_t nData = data.size();
  Batches batches(batch_size, nData);
  Batches::iterator ii = batches.begin();

  float fe1 = getFreeEnergy(getBatchData(data, ii, layer), W),
	fe2 = getFreeEnergy(getBatchData(data, ii+1, layer), W);

  data.getDataStream().rewind();

  return abs(fe1 - fe2);
}

mat StackedRbmTrainer::getBatchData(DataSet& data, const Batches::iterator& itr, int layer) {
  mat x = (mat) data[itr].x;
  for (int i=0; i<layer; ++i) 
    x = sigmoid(x * _weights[i]);
  return x;
}

void StackedRbmTrainer::rbm_train(DataSet& data, int layer, UNIT_TYPE vis_type, UNIT_TYPE hid_type) {

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

  size_t m = _dims[layer] + 1,
	 n = _dims[layer + 1] + 1;

  W = randn(m, n) * sqrt(0.1 / n);
  mat dW(m, n, 0);

  size_t minEpoch = 5, maxEpoch = 32;

  std::vector<float> errors;

  float initialSlope = 0;

  ProgressBar pBar("( | Δ free energy | = ...     , reconstruction error = ...      )");

  perf::Timer timer;
  timer.start();

  size_t epoch;
  for (epoch=0; epoch < maxEpoch; ++epoch) {

    float momentum = (epoch <= 5) ? initial_momentum : final_momentum;

    Batches batches(batch_size, nData);
    for (Batches::iterator itr = batches.begin() + 1; itr != batches.end(); ++itr) {
      // The first batch is kept as held-out set for validation. Therefore
      // itr starts from begin() + 1 rather than begin().

      mat v1, v2, h1, h2;

      v1 = getBatchData(data, itr, layer);
      // v1 = data.getX(*itr);
      fill_bias(v1);

      // Up propagation
      up_propagate(W, v1, h1, hid_type);

      // Calculate positive
      mat positive = ~v1 * h1;

      // Sampling
      sample(h1, hid_type);

      // Down-and-Up propagation
      down_propagate(W, v2, h1, vis_type);
      up_propagate(W, v2, h2, hid_type);

      // Calculate negative
      mat negative = ~v2 * h2;

      // Prevent weight explosion (cf. "kaldi-trunk/src/nnet/nnet-rbm.cc")
      antiWeightExplosion(W, v1, v2, lr);

      dW = dW * momentum				// momentum
	+ (lr / batch_size) * (positive - negative)	// gradient of CD
	- (lr * L2_penalty) * W;			// gradient of L2-penalty

      W += dW;
    }

    float fe_gap = getFreeEnergyGap(data, batch_size, W, layer);
    float error = getReconstructionError(data, W, vis_type, hid_type, layer);
    errors.push_back(error);

    if (epoch == minEpoch)
      initialSlope = getSlope(errors, minEpoch);

    if (epoch > minEpoch) {
      float ratio = abs(getSlope(errors, minEpoch) / initialSlope);
      float percentage = (epoch == maxEpoch - 1) ? 1.0 : std::min(1.0f, _slopeThres / ratio);

      char status[100];
      sprintf(status, "( | Δ free energy | = %.2e, reconstruction error = %.2e )", fe_gap, error);

      pBar.refresh(percentage, status);

      if (ratio < _slopeThres)
	break;
    }
  }

  float t_end = timer.getTime();
  printf("Average magnitude of elements in weight W = %.7f\n", nrm2(W) / sqrt(W.size()));
  printf("# of epoch = %lu, average time for each epoch = %f\n", epoch, t_end / epoch);
}

void StackedRbmTrainer::save(const string& fn) {
  FILE* fid = fopen(fn.c_str(), "w");

  if (!fid)
    throw std::runtime_error("Cannot open file: \"" + fn + "\"");

  FeatureTransform *affine, *activation;
  size_t dim;

  for (size_t i=0; i<_weights.size() - 1; ++i) {
    affine = new AffineTransform(_weights[i]);
    affine->write(fid);

    dim = affine->getOutputDimension();
    activation = new Sigmoid(dim, dim);
    activation->write(fid);

    delete affine;
    delete activation;
  }

  affine = new AffineTransform(_weights.back());
  affine->write(fid);

  dim = affine->getOutputDimension();
  activation = new Softmax(dim, dim);
  activation->write(fid);

  delete affine;
  delete activation;

  fclose(fid);
}

// Show a dialogue and ask user for the output dimension
size_t getOutputDimension() {
  string userInput = "";

  while (!is_number(userInput)) {
    printf("\33[33m Since RBM is a kind of UNSUPERVISED pre-training. "
	   "Please enter how many nodes you want in the output layer.\33[0m "
	   "[      ]\b\b\b\b\b");
    cin >> userInput;
  }

  return atoi(userInput.c_str());
}

std::vector<size_t> getDimensionsForRBM(
    size_t input_dim, 
    const string& hidden_structure, 
    size_t output_dim) {

  // ===========================================================================
  // Initialize hidden structure
  std::vector<size_t> dims = splitAsInt(hidden_structure, '-');
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


float getSlope(const std::vector<float> &seq, size_t N) {
  std::vector<float> x(N);
  for (size_t i=0; i<N; ++i)
    x[i] = N - 1 - i;

  std::vector<float> y(N);
  for (size_t i=seq.size() - N; i<seq.size(); ++i)
    y[i - (seq.size() - N)] = seq[i];

  float m, c;
  linearRegression(x, y, &m, &c);

  return m;
}

float getAsymptoticBound(const std::vector<float> &error, size_t epoch, size_t maxEpoch, size_t N) {
  std::vector<float> x(N);
  for (size_t i=0; i<N; ++i)
    x[i] = epoch - (N - 1 - i);

  std::vector<float> y(N);
  for (size_t i=error.size() - N; i<error.size(); ++i)
    y[i - (error.size() - N)] = error[i];

  float m, c;
  linearRegression(x, y, &m, &c);

  return m * (float) maxEpoch + c;
}

/*mat sum(mat& m, size_t dimension = 1) {
  if (dimension == 1)
    return mat(1, m.getRows(), 1) * m;
  else
    return m * mat(m.getCols(), 1, 1);
} */
