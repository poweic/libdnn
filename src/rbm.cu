#include <rbm.h>
#include <curand.h>
#include <curand_kernel.h>

__device__ float generate_rand(curandState* globalState) {
  curandState localState = *globalState;
  float RANDOM = curand_uniform( &localState );
  *globalState = localState;
  return RANDOM;
}

__global__ void setupCuRandState( curandState * state, size_t rows, size_t cols, unsigned long seed ) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  unsigned int i = x * rows + y;
  curand_init ( seed, i, 0, &state[i] );
}

__global__ void mat_rand(float* data, curandState* globalState, size_t rows, size_t cols) {

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  unsigned int i = x * rows + y;
  data[i] = generate_rand(globalState + i);
}

void fast_rand(mat& x) {
  size_t rows = x.getRows(),
	 cols = x.getCols();
  size_t N = rows * cols;

  const size_t BLOCK_DIM = 32;
  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  dim3 grid;
  grid.x = (unsigned int) ceil((float) cols / BLOCK_DIM);
  grid.y = (unsigned int) ceil((float) rows / BLOCK_DIM);

  static curandState* devStates = NULL;
  static size_t prevN = N;
  if (N > prevN) {
    prevN = N;
    cudaFree(devStates);
    devStates = NULL;
  }

  if (devStates == NULL) {
    cudaMalloc ( &devStates, N*sizeof( curandState ) );
    setupCuRandState <<< grid, threads >>> ( devStates, rows, cols, unsigned(time(NULL)) );
  }

  // setup seeds
  mat_rand <<< grid, threads >>> (x.getData(), devStates, rows, cols);
}

std::vector<mat> rbminit(DataSet& data, const std::vector<size_t> &dims, float slopeThres) {
  std::vector<mat> weights(dims.size() - 1);

  mat X = data.X;
  for (size_t i=0; i<weights.size(); ++i) {
    weights[i] = RBMinit(X, dims[i + 1], slopeThres);
    X = ext::sigmoid(X * weights[i]);
    fillLastColumnWith(X, 1.0f);
  }

  return weights;
}

__global__ void turnOnWithProbabilityKernel(float* const data, curandState* globalState, unsigned int rows, unsigned int cols) {

  // Matrix index
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  unsigned int i = x * rows + y;
  // data[i] = (float) data[i] > prob[i];
  data[i] = (float) (data[i] > generate_rand(globalState + i));
  __syncthreads();
}

void turnOnWithProbability(mat &y) {
  size_t rows = y.getRows();
  size_t cols = y.getCols();
  size_t N = rows * cols;

  const size_t BLOCK_DIM = 32;
  dim3 threads(BLOCK_DIM, BLOCK_DIM);
  dim3 grid;
  grid.x = (unsigned int) ceil((float) y.getCols() / BLOCK_DIM);
  grid.y = (unsigned int) ceil((float) y.getRows() / BLOCK_DIM);

  static curandState* devStates = NULL;
  static size_t prevN = N;
  if (N > prevN) {
    prevN = N;
    cudaFree(devStates);
    devStates = NULL;
  }

  if (devStates == NULL) {
    // printf("\n\nAllocate curandState of size %lu MBytes\n", N * sizeof(curandState) / 1024 / 1024);
    cudaMalloc ( &devStates, N * sizeof( curandState ) );
    setupCuRandState <<< grid, threads >>> ( devStates, rows, cols, unsigned(time(NULL)) );
  }

  turnOnWithProbabilityKernel<<< grid, threads >>>(y.getData(), devStates, y.getRows(), y.getCols());
  CCE(cudaDeviceSynchronize());
}

mat sum(mat& m, size_t dimension = 1) {
  if (dimension == 1)
    return (mat(1, m.getRows()) += 1) * m;
  else
    return m * (mat(m.getCols(), 1) += 1);
}

void linearRegression(const std::vector<float> &x, const std::vector<float>& y, float* const &m, float* const &c) {
  int n = x.size();
  double A=0.0,B=0.0,C=0.0,D=0.0;

  for (size_t i=0; i<n; ++i) {
    A += x[i];
    B += y[i];
    C += x[i]*x[i];
    D += x[i]*y[i];
  }

  *m = (n*D-A*B) / (n*C-A*A);
  *c = (B-(*m)*A) / n;
} 

float getSlope(const std::vector<float> &error, size_t N) {
  std::vector<float> x(N);
  for (size_t i=0; i<N; ++i)
    x[i] = N - 1 - i;

  std::vector<float> y(N);
  for (size_t i=error.size() - N; i<error.size(); ++i)
    y[i - (error.size() - N)] = error[i];

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

float calcStd(const std::vector<float> &error, size_t N) {
  float mean = 0;
  for (size_t i=error.size() - N; i < error.size(); ++i)
    mean += error[i];
  mean /= N;

  float std = 0;
  for (size_t i=error.size() - N; i < error.size(); ++i)
    std += pow(error[i] - mean, 2.0f);
  std = sqrt(std / (N-1));

  return std;
}

mat RBMinit(mat& data, size_t nHiddenUnits, float threshold) {
  // Make sure the visible units have values in the range [0, 1]
  float max = ext::max(data),
	min = ext::min(data);

  assert(max <= 1 && min >= 0);

  size_t input_dim = data.getCols();

  mat W(input_dim, nHiddenUnits + 1);
  ext::randn(W, 0, 0.1 / W.getCols());


  size_t batchSize = 128;
  size_t nData = data.getRows();

  size_t nBatch = nData / batchSize;
  size_t minEpoch = 5, maxEpoch = 1024;

  std::vector<float> errors;
  errors.reserve(maxEpoch);
  float learningRate = 1e-1;

  float initialSlope = 0;

  ProgressBar pBar("RBM init ( error = ...       , slope ratio = ...        )");
  for (size_t epoch=0; epoch < maxEpoch; ++epoch) {

    float error = 0;
    for (size_t i=0; i<nBatch; ++i) {
      mat v1(batchSize, input_dim);
      memcpy2D(v1, data, i * batchSize, 0, batchSize, input_dim, 0, 0);

      // Up Sampling
      mat h1 = ext::sigmoid(v1 * W);
      fillLastColumnWith(h1, 1.0f);
      turnOnWithProbability(h1);

      // Down-and-Up propagation
      mat v2 = ext::sigmoid(h1 * ~W);
      fillLastColumnWith(v2, 1.0f);
      turnOnWithProbability(v2);

      mat h2 = ext::sigmoid(v2 * W);
      fillLastColumnWith(h2, 1.0f);

      // Calculate Positive & Negative
      mat positive = ~v1 * h1;
      mat negative = ~v2 * h2;

      float lr = learningRate / batchSize;

      mat dW = lr * (positive - negative);

      W += dW;
      error += pow(nrm2(v1 - v2), 2.0f);
    }
    errors.push_back(sqrt(error) / ( nBatch * batchSize ) );

    if (epoch == minEpoch)
      initialSlope = getSlope(errors, minEpoch);

    if (epoch > minEpoch) {
      float ratio = abs(getSlope(errors, 5) / initialSlope);
      char status[100];
      sprintf(status, "RBM init ( error = %.4e, slope ratio = %.4e )", errors[epoch], ratio);
      pBar.refresh(std::min(1.0f, threshold / ratio), status);

      if (ratio < threshold)
	break;
    }
  }

  printf("\nAverage magnitude of element in weight W = %.7f\n", nrm2(W) / sqrt(W.size()));
  
  return W;
}

bool is_number(const std::string& s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

std::vector<size_t> getDimensionsForRBM(const DataSet& data, const string& structure) {

  string userInput = "";

  while (!is_number(userInput)) {
    printf("\33[33m Since RBM is a kind of UNSUPERVISED pre-training. "
	   "Please enter how many nodes you want in the output layer.\33[0m "
	   "[      ]\b\b\b\b\b");
    cin >> userInput;
  }

  size_t output_dim = atoi(userInput.c_str());

  // ===========================================================================
  // Initialize hidden structure
  size_t input_dim = data.X.getCols() - 1;
  std::vector<size_t> dims = splitAsInt(structure, '-');
  dims.insert(dims.begin(), input_dim);
  dims.push_back((size_t) output_dim);

  printf("\33[32m Start RBM pre-training with following hidden structure:\33[0m\n");
  printf("\33[34m [   input  layer  ]\33[0m %lu\n", dims[0]);
  for (size_t i=1; i<dims.size()-1; ++i)
    printf("\33[34m [ hidden layer #%-2lu]\33[0m %lu\n", i, dims[i]);
  printf("\33[34m [   output layer  ]\33[0m %lu\n\n", dims.back());
  // ===========================================================================

  return dims;
}
