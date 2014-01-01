#include <rbm.h>

std::vector<mat> rbminit(DataSet& data, const std::vector<size_t> &dims) {
  std::vector<mat> weights(dims.size() - 1);

  size_t nHiddenUnits = 7;
  mat X = data.X;
  for (size_t i=0; i<weights.size(); ++i) {
    weights[i] = RBMinit(X, dims[i + 1]);
    X = ext::sigmoid(X * weights[i]);
    fillLastColumnWith(X, 1.0f);
  }

  return weights;
}

__global__ void turnOnWithProbabilityKernel(float* const data, const float* const prob, unsigned int rows, unsigned int cols) {

  // Matrix index
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  unsigned int i = x * rows + y;
  data[i] = (float) data[i] > prob[i];
  __syncthreads();
}

void turnOnWithProbability(mat &y) {
  mat R(y.getRows(), y.getCols());
  ext::rand(R);

  const size_t N = 32;
  dim3 threads(N, N);
  dim3 grid;
  grid.x = (unsigned int) ceil((float) y.getCols() / N);
  grid.y = (unsigned int) ceil((float) y.getRows() / N);

  turnOnWithProbabilityKernel<<< grid, threads >>>(y.getData(), R.getData(), y.getRows(), y.getCols());
  CCE(cudaDeviceSynchronize());
}

mat sum(mat& m, size_t dimension = 1) {
  if (dimension == 1)
    return (mat(1, m.getRows()) += 1) * m;
  else
    return m * (mat(m.getCols(), 1) += 1);
}

mat RBMinit(mat& data, size_t nHiddenUnits) {

  size_t input_dim = data.getCols();

  mat W(input_dim, nHiddenUnits + 1);
  ext::randn(W, 0, 0.01);


  size_t batchSize = 32;
  size_t nData = data.getRows();

  size_t nBatch = nData / batchSize;
  size_t maxEpoch = 1024;

  ProgressBar pBar("RBM initialization");
  for (size_t epoch=0; epoch < maxEpoch; ++epoch) {
    pBar.refresh(epoch, maxEpoch);

    for (size_t i=0; i<nBatch; ++i) {
      mat x(batchSize, input_dim);
      memcpy2D(x, data, i * batchSize, 0, batchSize, input_dim, 0, 0);

      // Up Sampling
      mat y = ext::sigmoid(x * W);
      fillLastColumnWith(y, 1.0f);
      turnOnWithProbability(y);

      // Down-and-Up propagation
      mat x2 = ext::sigmoid(y * ~W);
      fillLastColumnWith(x2, 1.0f);

      mat y2 = ext::sigmoid(x2 * W);
      fillLastColumnWith(y2, 1.0f);

      // Calculate Positive & Negative
      mat positive = ~x  * y ;
      mat negative = ~x2 * y2;

      float learningRate = 0.001;
      float lr = learningRate / batchSize;

      mat dW = lr * (positive - negative);

      float ratio = nrm2(dW) / nrm2(W);
      if (ratio > 0.01)
	dW *= 0.01 / ratio;

      W += dW;

      /*float normDW = nrm2(dW);
      float normW = nrm2(W);
      printf("epoch #%4lu: norm(dW) / norm(W) = %.7f (%.4f / %.4f) \n", epoch, normDW / normW, normDW, normW);*/

    }
  }

  printf("Average magnitude of element in weight W = %.7f\n", nrm2(W) / W.size());
  
  return W;
}
