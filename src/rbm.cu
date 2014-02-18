#include <rbm.h>
#include <curand.h>
#include <curand_kernel.h>

hmat batchFeedForwarding(const hmat& X, const mat& w) {
  size_t nData = X.getCols();

  hmat Y(w.getCols(), nData);
  Batches batches(2048, nData);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    mat fin  = getBatchData(X, *itr);
    mat fout = sigmoid(fin * w);
    fillLastColumnWith(fout, 1.0f);

    size_t offset = fout.getCols() * itr->offset,
	   nBytes = sizeof(float) * fout.size();

    fout = ~fout;
    CCE(cudaMemcpy(Y.getData() + offset, fout.getData(), nBytes, cudaMemcpyDeviceToHost));
  }
  return Y;
}

std::vector<mat> rbminit(DataSet& data, const std::vector<size_t>& dims, float slopeThres) {
  std::vector<mat> weights(dims.size() - 1);

  size_t nData = data.size();

  hmat X = data.getX();
  for (size_t i=0; i<weights.size(); ++i) {
    weights[i] = RBMinit(X, dims[i + 1], slopeThres);
    X = batchFeedForwarding(X, weights[i]);
  }

  return weights;
}

__device__ float generate_rand(curandState* globalState) {
  curandState localState = *globalState;
  float RANDOM = curand_uniform( &localState );
  *globalState = localState;
  return RANDOM;
}

__global__ void setupCuRandState( curandState * state, unsigned long seed ) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  curand_init ( seed, x, 0, &state[x] );
}

__global__ void turnOnWithProbabilityKernel(float* const data, curandState* globalState, unsigned int rows, unsigned int cols) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  if (x >= cols || y >= rows)
    return;

  int i = x * rows + y;
  int j = tx * blockDim.y + ty;
  data[i] = (float) (data[i] > generate_rand(globalState + j));
  __syncthreads();
}

void turnOnWithProbability(mat &y) {
  size_t rows = y.getRows();
  size_t cols = y.getCols();
  
  const size_t N = 32;
  dim3 threads(N, N);
  dim3 grid;
  grid.x = (unsigned int) ceil((float) y.getCols() / N);
  grid.y = (unsigned int) ceil((float) y.getRows() / N);

  static curandState* devStates = NULL;

  if (devStates == NULL) {
    cudaMalloc ( &devStates, N * N * sizeof( curandState ) );
    setupCuRandState <<< 1, N*N >>> ( devStates, unsigned(time(NULL)) );
  }

  turnOnWithProbabilityKernel<<< grid, threads >>>(y.getData(), devStates, y.getRows(), y.getCols());
  CCE(cudaDeviceSynchronize());
}

mat RBMinit(const hmat& data, size_t nHiddenUnits, float threshold) {
  // Make sure the visible units have values in the range [0, 1]
  float max = ext::max(data),
	min = ext::min(data);

  assert(max <= 1 && min >= 0);

  size_t batchSize = 128;
  size_t input_dim = data.getRows();
  size_t nData = data.getCols();

  mat W(input_dim, nHiddenUnits + 1);
  ext::randn(W, 0, 0.1 / W.getCols());

  size_t minEpoch = 5, maxEpoch = 1024;

  std::vector<float> errors;
  errors.reserve(maxEpoch);
  float learningRate = 1e-1;

  float initialSlope = 0;

  ProgressBar pBar("RBM init ( error = ...       , slope ratio = ...        )");

  perf::Timer timer;
  timer.start();
  size_t epoch;
  for (epoch=0; epoch < maxEpoch; ++epoch) {

    float error = 0;

    Batches batches(batchSize, nData);
    for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {

      mat v1 = getBatchData(data, *itr);

      // Up Sampling
      mat h1 = sigmoid(v1 * W);
      fillLastColumnWith(h1, 1.0f);
      turnOnWithProbability(h1);

      // Down-and-Up propagation
      mat v2 = sigmoid(h1 * ~W);
      fillLastColumnWith(v2, 1.0f);
      turnOnWithProbability(v2);

      mat h2 = sigmoid(v2 * W);
      fillLastColumnWith(h2, 1.0f);

      // Calculate Positive & Negative
      mat positive = ~v1 * h1;
      mat negative = ~v2 * h2;

      float lr = learningRate / batchSize;

      mat dW = lr * (positive - negative);

      W += dW;
      error += pow(nrm2(v1 - v2), 2.0f);
    }

    errors.push_back(sqrt(error) / nData );

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

  printf("Average magnitude of element in weight W = %.7f\n", nrm2(W) / sqrt(W.size()));
  float t_end = timer.getTime();
  printf("Average time for each epoch = %f\n", t_end / epoch);
  
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
  size_t input_dim = data.getInputDimension();
  std::vector<size_t> dims = splitAsInt(structure, '-');
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

/*mat sum(mat& m, size_t dimension = 1) {
  if (dimension == 1)
    return (mat(1, m.getRows()) += 1) * m;
  else
    return m * (mat(m.getCols(), 1) += 1);
} */
