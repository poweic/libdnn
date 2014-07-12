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

inline __device__ void sample_gaussian(float& x, curandState* state) {
  x += curand_normal(state);
}

inline __device__ void sample_bernoulli(float& x, curandState* state) {
  x = (float) (x >= curand_uniform(state));
}

template <Operation op>
__global__ void element_wise_curand_kernel(float* const data, curandState* globalState, unsigned int rows, unsigned int cols) {
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

void sample(mat &prob, UNIT_TYPE type) {
  static CURAND_STATE state;

  ALLOCATE_GRIDS_AND_THREADS(prob.getRows(), prob.getCols());

  switch (type) {
    case GAUSSIAN:
      element_wise_curand_kernel<sample_gaussian><<< grids, threads >>>(prob.getData(), state.get(), prob.getRows(), prob.getCols());
      break;
    case BERNOULLI:
      element_wise_curand_kernel<sample_bernoulli><<< grids, threads >>>(prob.getData(), state.get(), prob.getRows(), prob.getCols());
      break;
  }

  CCE(cudaDeviceSynchronize());
  fill_bias(prob);
}

mat randn(int m, int n) {

#ifdef DEBUG
  // Use ext::randn (which is set to seed 0) to debug.
  mat x(m, n);
  ext::randn(x);
  return x;
#else
  static CURAND_STATE state;

  mat x(m, n);

  ALLOCATE_GRIDS_AND_THREADS(m, n);
  element_wise_curand_kernel<get_curand_normal><<<grids, threads>>>(x.getData(), state.get(), m, n);
  CCE(cudaDeviceSynchronize());

  return x;
#endif
}

mat rand(int m, int n) {

#ifdef DEBUG
  // Use ext::rand (which is set to seed 0) to debug.
  mat x(m, n);
  ext::rand(x);
  return x;
#else
  static CURAND_STATE state;

  mat x(m, n);

  ALLOCATE_GRIDS_AND_THREADS(m, n);
  element_wise_curand_kernel<get_curand_uniform><<<grids, threads>>>(x.getData(), state.get(), m, n);
  CCE(cudaDeviceSynchronize());

  return x;
#endif
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

template <typename T>
device_matrix<T> MaxPerRow(device_matrix<T>& A) {
  device_matrix<T> rmax(A.getRows(), 1);
  device_matrix<T> At = ~A;

  // allocate storage for per-row results and indices
  thrust::device_vector<T> row_indices(A.getRows());
  thrust::device_vector<T> row_results(A.getRows());

  // compute row sums by summing values with equal row indices
  thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(A.getCols())),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(A.getCols())) + A.size(),
     thrust::device_ptr<T>(At.getData()),
     row_indices.begin(),
     thrust::device_ptr<T>(rmax.getData()),
     thrust::equal_to<T>(),
     thrust::maximum<T>());

  return rmax;
}

template <typename T>
__global__ void substract_max_per_row_kernel(T* const A, T* const rmax, unsigned int rows, unsigned int cols) {
  // Matrix index
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  A[x * rows + y] -= rmax[y];
}

template <typename T>
void SubstractMaxPerRow(device_matrix<T>& x) {
  device_matrix<T> rmax = MaxPerRow(x);

  ALLOCATE_GRIDS_AND_THREADS(x.getRows(), x.getCols());
  substract_max_per_row_kernel<float><<< grids, threads >>>
    (x.getData(), rmax.getData(), x.getRows(), x.getCols());

  CCE(cudaDeviceSynchronize());
}

template <typename T>
void fillLastColumnWith(device_matrix<T>& A, const T value) {
  thrust::device_ptr<T> ptr(A.getData());
  thrust::fill(ptr + A.size() - A.getRows(), ptr + A.size(), value);
}

template <typename T>
device_matrix<T> operator & (const device_matrix<T>& A, const device_matrix<T>& B) {
  assert(A.getRows() == B.getRows() && A.getCols() == B.getCols());

  device_matrix<T> C(A.getRows(), A.getCols());

  thrust::device_ptr<T> aPtr(A.getData());
  thrust::device_ptr<T> bPtr(B.getData());
  thrust::device_ptr<T> cPtr(C.getData());

  thrust::transform(aPtr, aPtr + A.size(), bPtr, cPtr, thrust::multiplies<T>());

  return C;
}

template <typename T>
device_matrix<T> log(const device_matrix<T>& x) {
  return transform(x, func::log<T>());
}

template <typename T>
device_matrix<T> log1pexp(const device_matrix<T>& x) {
  return transform(x, func::log_of_one_plus_exp<T>());
}

template <typename T>
device_matrix<T> sigmoid(const device_matrix<T>& x) {
  return transform(x, func::sigmoid<T>());
}

template <typename T>
device_matrix<T> softmax(const device_matrix<T>& x) {
  mat x2(x);
  x2.resize(x2.getRows(), x2.getCols() - 1);
  SubstractMaxPerRow(x2);

  mat p(x2.getRows(), x2.getCols());

  thrust::device_ptr<T> xPtr(x2.getData());
  thrust::device_ptr<T> pPtr(p.getData());
  thrust::transform(xPtr, xPtr + x2.size(), pPtr, func::exp<T>());

  mat sumOfProb = p * mat(p.getCols(), p.getCols(), 1);

  mat y(p.getRows(), p.getCols() + 1);
  thrust::device_ptr<T> yPtr(y.getData());
  thrust::device_ptr<T> sPtr(sumOfProb.getData());
  thrust::transform(pPtr, pPtr + p.size(), sPtr, yPtr, thrust::divides<T>());

  return y;
}

/* \brief Explicit instantiation definition of template functions
 */

#define register_device_matrix_utility(T) \
  template device_matrix<T> operator &<T> (const device_matrix<T>& A, const device_matrix<T>& B); \
  template void fillLastColumnWith<T>(device_matrix<T>& A, const T value); \
  template device_matrix<T> log<T>(const device_matrix<T>& x); \
  template device_matrix<T> log1pexp<T>(const device_matrix<T>& x); \
  template device_matrix<T> sigmoid<T>(const device_matrix<T>& x); \
  template device_matrix<T> softmax<T>(const device_matrix<T>& x); \
  template device_matrix<T> MaxPerRow<T>(device_matrix<T>& A); \
  template void SubstractMaxPerRow<T>(device_matrix<T>& x);

register_device_matrix_utility(float);
