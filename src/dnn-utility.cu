#include <dnn-utility.h>

CURAND_STATE::CURAND_STATE(unsigned seed, int N): _states(NULL) {
  cudaMalloc ( &_states, N * N * sizeof( curandState ) );
  setupCuRandState <<< 1, N * N >>> ( _states, seed );
  CCE(cudaDeviceSynchronize());
}

curandState* CURAND_STATE::get() const {
  return _states;
}

CURAND_STATE::~CURAND_STATE() {
  cudaFree(_states);
}

__global__ void setupCuRandState( curandState * state, unsigned long seed ) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  curand_init ( seed, x, 0, &state[x] );
}

inline __device__ void get_curand_normal(float& x, curandState* state) {
  x = curand_normal(state);
}

inline __device__ void get_curand_uniform(float& x, curandState* state) {
  x = curand_uniform(state);
}

template <Operation op>
__global__ void rand_kernel(float* const data, curandState* globalState, unsigned int rows, unsigned int cols) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  if (x >= cols || y >= rows)
    return;

  int i = x * rows + y;
  int j = tx * blockDim.y + ty;
  op(data[i], globalState + j);
  __syncthreads();
}

mat randn(int m, int n) {

  // Use ext::randn (which is set to seed 0) to debug.
  mat x(m, n);
  ext::randn(x);
  return x;

  /*static CURAND_STATE state();

  mat x(m, n);

  ALLOCATE_GRIDS_AND_THREADS(m, n);
  rand_kernel<get_curand_normal><<<grids, threads>>>(x.getData(), state.get(), m, n);
  CCE(cudaDeviceSynchronize());

  return x;*/
}

mat rand(int m, int n) {

  // Use ext::rand (which is set to seed 0) to debug.
  mat x(m, n);
  ext::rand(x);
  return x;

  /*static CURAND_STATE state();

  mat x(m, n);

  ALLOCATE_GRIDS_AND_THREADS(m, n);
  rand_kernel<get_curand_uniform><<<grids, threads>>>(x.getData(), state.get(), m, n);
  CCE(cudaDeviceSynchronize());

  return x;*/
}


map<int, int> getLabelMapping(const hmat& labels) {
  map<int, int> classes;
  for (size_t i=0; i<labels.size(); ++i)
    classes[(int) labels[i]] = 1;

  int counter = 0;
  map<int, int>::iterator itr = classes.begin();
  for (; itr != classes.end(); ++itr)
    itr->second = ++counter;

  return classes;
}

namespace ext {

  void rescale(mat& data, float lower, float upper) {
    float min = ext::min(data);
    float max = ext::max(data);

    float ratio = (upper - lower) / (max - min);
    data = (data - min) * ratio + lower;
  }

  float max(const mat& v) {
    thrust::device_ptr<float> vPtr(v.getData());
    thrust::device_ptr<float> maxPtr = thrust::max_element(vPtr, vPtr + v.size());
    thrust::host_vector<float> hMaxPtr(maxPtr, maxPtr + 1);
    return hMaxPtr[0];
  }

  float min(const mat& v) {
    thrust::device_ptr<float> vPtr(v.getData());
    thrust::device_ptr<float> minPtr = thrust::min_element(vPtr, vPtr + v.size());
    thrust::host_vector<float> hMaxPtr(minPtr, minPtr + 1);
    return hMaxPtr[0];
  }

  float max(const hmat& v) {
    float* m = thrust::max_element(v.getData(), v.getData() + v.size());
    return *m;
  }

  float min(const hmat& v) {
    float* m = thrust::min_element(v.getData(), v.getData() + v.size());
    return *m;
  }
};

__global__ void dcrossentropy_kernel(float* error, float* const target, float* const output, unsigned int rows, unsigned int cols) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  if (x >= cols || y >= rows)
    return;

  int i = x * rows + y;

  // target[y] need to be 0-based 
  error[i] = output[i] - (float) (target[y] == x);

  __syncthreads();
}

void dCrossEntropy(mat& error, const mat &target, const mat& output) {

  assert(error.getRows() == output.getRows() && error.getCols() == output.getCols());

  ALLOCATE_GRIDS_AND_THREADS(error.getRows(), error.getCols());

  dcrossentropy_kernel<<< grids, threads >>>(
      error.getData(), target.getData(), output.getData(),
      error.getRows(), error.getCols());

  CCE(cudaDeviceSynchronize());
}

mat getError(const mat& target, const mat& output, ERROR_MEASURE errorMeasure) {

  mat error(output.getRows(), output.getCols());

  switch (errorMeasure) {
    case L2ERROR: 
      // FIXME
      /*error = output - target;
      error.reserve(error.getRows() * (error.getCols() + 1));
      error.resize(error.getRows(), error.getCols() + 1);*/

      break;
    case CROSS_ENTROPY:

      dCrossEntropy(error, target, output);

      break;
  }

  return error;
}

mat posteriorProb2Label(const mat& prob) {

  assert(prob.getCols() > 1);

  size_t rows = prob.getRows(),
	 cols = prob.getCols();

  float* h_prob = new float[prob.size()];
  float* h_labels  = new float[rows];
  CCE(cudaMemcpy(h_prob, prob.getData(), sizeof(float) * prob.size(), cudaMemcpyDeviceToHost));
  CCE(cudaDeviceSynchronize());

  for (size_t i=0; i<rows; ++i) {

    float max = -1e10;
    size_t maxIdx = 0;

    for (size_t j=0; j<cols; ++j) {
      if (h_prob[j * rows + i] > max) {
	max = h_prob[j * rows + i];
	maxIdx = j;
      }
    }

    h_labels[i] = maxIdx;
  }

  mat labels(h_labels, rows, 1);

  delete [] h_prob;
  delete [] h_labels;

  return labels;
}

vector<float> copyToHost(const mat& m) {
  vector<float> hm(m.size());
  thrust::device_ptr<float> dPtr(m.getData());
  thrust::copy(dPtr, dPtr + m.size(), hm.begin());
  return hm;
}

size_t countDifference(const mat& m1, const mat& m2) {
  assert(m1.size() == m2.size());

  size_t L = m1.size();
  thrust::device_ptr<float> ptr1(m1.getData());
  thrust::device_ptr<float> ptr2(m2.getData());

  size_t nDiff = thrust::inner_product(ptr1, ptr1 + L, ptr2, 0.0, thrust::plus<float>(), thrust::not_equal_to<float>());
  return nDiff;
}


size_t zeroOneError(const mat& prob, const mat& label, ERROR_MEASURE errorMeasure) {
  assert(prob.getRows() == label.getRows());
  assert(label.getCols() == 1);

  size_t nError = 0;

  if (errorMeasure == L2ERROR) {
    // nError = countDifference(label, prob);
  }
  else {
    mat L = posteriorProb2Label(prob);
    nError = countDifference(L, label);
  }

  return nError;
}

device_matrix<float> log(const device_matrix<float>& x) {
  return transform(x, func::log<float>());
}
