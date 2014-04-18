#include <rbm.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdlib>
#define fill_bias(x) { fillLastColumnWith(x, 1.0f); }

ostream& operator << (ostream& os, const RBM_UNIT_TYPE& type) {
  switch (type) {
    case GAUSSIAN: os << "Gaussian"; break;
    case BERNOULLI: os << "Bernoulli"; break;
  }
  return os;
}

hmat batchFeedForwarding(const hmat& X, const mat& w) {
  printf("Start feedforwarding (in batch)... \n");
  perf::Timer timer;
  timer.start();

  size_t nData = X.getCols();

  // TODO Crashed when # of input (i.e. nData) is very large.
  // Even the original data (say 351 x 500,000) can be fitted into memory
  // , a hidden layer of width 2048 would need (2048 x 500,000) memory,
  // which will casue OUT OF MEMORY.
  printf("Allocating host matrix of size %lu ...", w.getCols() * nData);
  hmat Y(w.getCols(), nData);
  printf("[Done]\n");

  Batches batches(2048, nData);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    mat fin  = getBatchData(X, *itr);
    mat fout = sigmoid(fin * w);
    fill_bias(fout);

    size_t offset = fout.getCols() * itr->offset,
	   nBytes = sizeof(float) * fout.size();

    fout = ~fout;
    CCE(cudaMemcpy(Y.getData() + offset, fout.getData(), nBytes, cudaMemcpyDeviceToHost));
  }

  timer.elapsed();
  return Y;
}

std::vector<mat> initStackedRBM(DataSet& data, const std::vector<size_t>& dims,
    float slopeThres, RBM_UNIT_TYPE type, float learning_rate) {

  std::vector<mat> weights(dims.size() - 1);

  size_t nData = data.size();

  // FIXME For NOW, hidden units are only allowed to be Bernoulli
  RBM_UNIT_TYPE vis_type = type, hid_type = BERNOULLI;

  // If vis_type is Bernoulli, make sure the visible units have values in the
  // range [0, 1]. If the values scatter over a wide range and has a Gaussian
  // distribution, make sure the values are normalized to 0 mean and 1 standard
  // deviation before going into this function.
  if (vis_type == BERNOULLI)
    assert(ext::max(data.getX()) <= 1 && ext::min(data.getX()) >= 0);
  
  hmat X = data.getX();
  for (size_t i=0; i<weights.size(); ++i) {

    weights[i] = rbmTrain(X, dims[i + 1], slopeThres, vis_type, hid_type, learning_rate);

    vis_type = hid_type;
    hid_type = BERNOULLI;

    X = batchFeedForwarding(X, weights[i]);
  }

  return weights;
}

__global__ void setupCuRandState( curandState * state, unsigned long seed ) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  curand_init ( seed, x, 0, &state[x] );
}

inline __device__ void gaussian(float& x, curandState* state) {
  x += curand_normal(state);
}

inline __device__ void bernoulli(float& x, curandState* state) {
  x = (float) (x >= curand_uniform(state));
}

typedef __device__ void (*Operation)(float&, curandState*);

template <Operation op>
__global__ void sampling_kernel(float* const data, curandState* globalState, unsigned int rows, unsigned int cols) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  if (x >= cols || y >= rows)
    return;

  int i = x * rows + y;
  int j = tx * blockDim.y + ty;
  op(data[i], globalState +j);
  // data[i] = data[i] + curand_normal(globalState + j);
  __syncthreads();
}

class CURAND_STATE {
public:
  CURAND_STATE(unsigned seed = unsigned(time(NULL)), int N = 32): _states(NULL) {
    cudaMalloc ( &_states, N * N * sizeof( curandState ) );
    setupCuRandState <<< 1, N * N >>> ( _states, seed );
  }

  curandState* get() const { return _states; }

  ~CURAND_STATE() {
    cudaFree(_states);
  }

private:
  curandState* _states;
};

void sample(mat &prob, RBM_UNIT_TYPE type) {
  static CURAND_STATE state;

  const size_t N = 32;
  dim3 threads(N, N);
  dim3 grid;
  grid.x = (unsigned int) ceil((float) prob.getCols() / N);
  grid.y = (unsigned int) ceil((float) prob.getRows() / N);

  switch (type) {
    case GAUSSIAN:
      sampling_kernel<gaussian><<< grid, threads >>>(prob.getData(), state.get(), prob.getRows(), prob.getCols());
      break;
    case BERNOULLI:
      sampling_kernel<bernoulli><<< grid, threads >>>(prob.getData(), state.get(), prob.getRows(), prob.getCols());
      break;
  }

  CCE(cudaDeviceSynchronize());
  fill_bias(prob);
}

void up_propagate(const mat& W, const mat& visible, mat& hidden, RBM_UNIT_TYPE type) {
  hidden = visible * W;

  if (type == BERNOULLI)
    hidden = sigmoid(hidden);
  
  fill_bias(hidden);
}

void down_propagate(const mat& W, mat& visible, const mat& hidden, RBM_UNIT_TYPE type) {
  visible = hidden * ~W;

  if (type == BERNOULLI)
    visible = sigmoid(visible);

  fill_bias(visible);
}

// Calculuate standard deviation of each dimension of x.
// After that, average over all standard deviations.
float calcAverageStandardDeviation(const mat& x) {
  size_t rows = x.getRows(),
	 cols = x.getCols();

  mat x_minus_mean = x - ((mat(rows, rows) += 1) * x) / rows;
  mat sum_of_squares = (mat(1, rows) += 1) * (x_minus_mean & x_minus_mean);

  hmat squares(1, cols);
  CCE(cudaMemcpy(squares.getData(), sum_of_squares.getData(), sizeof(float) * squares.size(), cudaMemcpyDeviceToHost));

  int N = (rows == 1) ? 1 : rows - 1;

  float s = 0;
  for (size_t i=0; i<cols; ++i)
    s += sqrt(squares[i] / N);
  s /= cols;

  return s;
}

void antiWeightExplosion(mat& W, const mat& v1, const mat& v2, float &learning_rate) {
  float v1_avg_std = calcAverageStandardDeviation(v1),
	v2_avg_std = calcAverageStandardDeviation(v2),
	std_ratio = v2_avg_std / v1_avg_std;

  assert(std_ratio != 0);

  if (std_ratio > 2) {
    printf("\33[34m[Info]\33[0m W and learning_rate shrinked to prevent weights to explode!!\n");
    W /= std_ratio;
    learning_rate *= 0.9;
  }
}

float getReconstructionError(const hmat& data, const mat& W, RBM_UNIT_TYPE vis_type, RBM_UNIT_TYPE hid_type) {

  float r_error = 0;

  const size_t batch_size = 1024;
  size_t nData = data.getCols();

  Batches batches(batch_size, nData);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {

    // v1 is input data, v2 is reconstructed data
    mat v1, v2, h1;

    v1 = getBatchData(data, *itr);
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

  return r_error;
}

float getFreeEnergy(const mat& visible, const mat& W) {
  int N = visible.getRows();
  mat hidden = visible * W;

  mat va(N, 1);
  CCE(cudaMemcpy(va.getData(),
	hidden.getData() + hidden.size() - N,
	sizeof(float) * N, cudaMemcpyDeviceToDevice));

  fillLastColumnWith(hidden, -1000.0f);

  transform(hidden, func::log_of_one_plus_exp<float>());

  mat e = hidden * (mat(hidden.getCols(), 1) += 1) + va;
  mat sum_of_e = (mat(1, N) += 1) * e;

  float free_energy = 0;
  CCE(cudaMemcpy(&free_energy, sum_of_e.getData(), sizeof(float), cudaMemcpyDeviceToHost));

  free_energy = - free_energy / N;

  return free_energy;
}

float getFreeEnergyGap(const hmat& data, size_t batch_size, const mat& W) {

  size_t nData = data.getCols();
  Batches batches(batch_size, nData);
  Batches::iterator ii = batches.begin();

  float fe1 = getFreeEnergy(getBatchData(data, *(ii    )), W),
	fe2 = getFreeEnergy(getBatchData(data, *(ii + 1)), W);

  return abs(fe1 - fe2);
}

mat rbmTrain(const hmat& data, size_t nHiddenUnits, float threshold, RBM_UNIT_TYPE vis_type, RBM_UNIT_TYPE hid_type, float learning_rate) {

  // Note: The learning rate of Gaussian RBM needs to be about one or two
  // orders of magnitude smaller than when using binary visible units.
  // Otherwise value will explode very quickly and get NaN.
  // [cf. A Practical Guide to Training Restricted Boltzmann Machines]
  // FIXME 為什麼當data量增加到某個程度時，相同的learning-rate會導致 W 爆掉??
  // 照理說應該不會影響。因為data增加應該只相當於多跑幾個epoch，怎麼會讓 W 爆掉??
  if (vis_type == GAUSSIAN) learning_rate *= 0.01;
  if (hid_type == GAUSSIAN) learning_rate *= 0.01;
  cout << "Training \33[34m" << vis_type << " - " << hid_type << "\33[0m RBM ..." << endl;

  const float initial_momentum = 0.5, final_momentum = 0.9, L2_penalty = 0.0002;

  size_t batch_size = 1024;
  size_t input_dim = data.getRows();
  size_t nData = data.getCols();

  mat W(input_dim, nHiddenUnits + 1);
  mat dW(W.getRows(), W.getCols());
  ext::randn(W, 0, 0.1 / W.getCols());

  size_t minEpoch = 5, maxEpoch = 64;

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

      v1 = getBatchData(data, *itr);
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
      antiWeightExplosion(W, v1, v2, learning_rate);

      dW = dW * momentum					// momentum
	+ (learning_rate / batch_size) * (positive - negative)	// gradient of CD
	- (learning_rate * L2_penalty) * W;			// gradient of L2-penalty

      W += dW;
    }

    float fe_gap = getFreeEnergyGap(data, batch_size, W);
    float error = getReconstructionError(data, W, vis_type, hid_type);
    errors.push_back(error);

    if (epoch == minEpoch)
      initialSlope = getSlope(errors, minEpoch);

    if (epoch > minEpoch) {
      float ratio = abs(getSlope(errors, minEpoch) / initialSlope);
      float percentage = (epoch == maxEpoch - 1) ? 1.0 : std::min(1.0f, threshold / ratio);

      char status[100];
      sprintf(status, "( | Δ free energy | = %.2e, reconstruction error = %.2e )", fe_gap, error);

      pBar.refresh(percentage, status);

      if (ratio < threshold)
	break;
    }
  }

  float t_end = timer.getTime();
  printf("Average magnitude of elements in weight W = %.7f\n", nrm2(W) / sqrt(W.size()));
  printf("# of epoch = %lu, average time for each epoch = %f\n", epoch, t_end / epoch);
  
  return W;
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
    return (mat(1, m.getRows()) += 1) * m;
  else
    return m * (mat(m.getCols(), 1) += 1);
} */
