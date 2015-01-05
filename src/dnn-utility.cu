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

#include <dnn-utility.h>
using namespace std;

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

  ALLOCATE_GRIDS_AND_THREADS(prob.getCols(), prob.getRows());

  switch (type) {
    case GAUSSIAN:
      element_wise_curand_kernel<sample_gaussian><<< grids, threads >>>(prob.getData(), state.get(), prob.getRows(), prob.getCols());
      break;
    case BERNOULLI:
      element_wise_curand_kernel<sample_bernoulli><<< grids, threads >>>(prob.getData(), state.get(), prob.getRows(), prob.getCols());
      break;
  }

  CCE(cudaDeviceSynchronize());
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

  ALLOCATE_GRIDS_AND_THREADS(n, m);
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

  ALLOCATE_GRIDS_AND_THREADS(n, m);
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

__global__ void compute_error_kernel(float* error, float* const target,
    float* const output, unsigned int rows, unsigned int cols) {

  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  if (x >= rows || y >= cols)
    return;

  int i = y * rows + x;

  // target[y] need to be 0-based 
  error[i] = output[i] - (float) (target[y] == x);

  __syncthreads();
}

mat getError(const mat& target, const mat& output, ERROR_MEASURE errorMeasure) {

  mat error(output.getRows(), output.getCols());

  switch (errorMeasure) {

    case L2ERROR: 
      // FIXME
      // error = ~output - target;
      // error = ~error;
      break;

    case CROSS_ENTROPY:

      ALLOCATE_GRIDS_AND_THREADS(error.getRows(), error.getCols());

      compute_error_kernel<<< grids, threads >>>(
	  error.getData(), target.getData(), output.getData(),
	  error.getRows(), error.getCols());

      CCE(cudaDeviceSynchronize());

      break;
  }

  return error;
}

mat posteriorProb2Label(const mat& prob) {

  assert(prob.getCols() > 1);

  size_t rows = prob.getRows(),
	 cols = prob.getCols();

  hmat h_prob(prob);
  hmat h_labels(1, cols);

  for (size_t j=0; j<cols; ++j) {

    float max = -1e10;
    size_t maxIdx = 0;

    for (size_t i=0; i<rows; ++i) {
      if (h_prob(i, j) > max) {
	max = h_prob(i, j);
	maxIdx = i;
      }
    }

    h_labels[j] = maxIdx;
  }

  return h_labels;
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


size_t zeroOneError(const mat& prob, const mat& label) {
  assert(prob.getCols() == label.getRows());
  assert(label.getCols() == 1);

  mat L = posteriorProb2Label(prob);

  return countDifference(L, label);
}

template <typename T>
device_matrix<T> MaxPerRow(const device_matrix<T>& A) {

  device_matrix<T> At(~A);
  device_matrix<T> rmax(At.getCols(), 1);

  // allocate storage for per-row results and indices
  thrust::device_vector<T> row_indices(At.getCols());

  // Originally, it compute row sums (thrust::plus) by summing values with equal
  // row indices. I replace thrust::plus with thrust::maximum and get rowmax
  thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(At.getRows())),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(At.getRows())) + A.size(),
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

  ALLOCATE_GRIDS_AND_THREADS(x.getCols(), x.getRows());
  substract_max_per_row_kernel<float><<< grids, threads >>>
    (x.getData(), rmax.getData(), x.getRows(), x.getCols());

  CCE(cudaDeviceSynchronize());
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
device_matrix<T>& operator &= (device_matrix<T>& A, const device_matrix<T>& B) {
  A = A & B;
  return A;
}

template <typename T>
device_matrix<T> exp(const device_matrix<T>& x) {
  return transform(x, func::exp<T>());
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
device_matrix<T> d_sigmoid(const device_matrix<T>& x) {
  return transform(x, func::d_sigmoid<T>());
}

template <typename T>
device_matrix<T> tanh(const device_matrix<T>& x) {
  return transform(x, func::hyperbolic_tangent<T>());
}

template <typename T>
device_matrix<T> d_tanh(const device_matrix<T>& x) {
  return transform(x, func::d_hyperbolic_tangent<T>());
}

template <typename T>
device_matrix<T> relu(const device_matrix<T>& x) {
  return transform(x, func::max<T>(0.0f));
}

template <typename T>
device_matrix<T> is_greater(const device_matrix<T>& x, const T value) {
  return transform(x, func::greater<T>(value));
}

template <typename T>
device_matrix<T> softmax(const device_matrix<T>& x_t) {
  mat x(~x_t);
  x.resize(x.getRows(), x.getCols() - 1);
  SubstractMaxPerRow(x);

  x = exp(x);
  thrust::device_ptr<T> xPtr(x.getData());

  mat sum = x * mat(x.getCols(), x.getCols(), 1);
  mat y(x.getRows(), x.getCols() + 1);

  thrust::transform(xPtr, xPtr + x.size(),
      thrust::device_ptr<T>(sum.getData()),
      thrust::device_ptr<T>(y.getData()),
      thrust::divides<T>());

  return ~y;
}

/* ! \brief Sum all the elements in a matrix.
 * \fn sum_all(const device_matrix<T>& x)
 * \param x matrix x to be sum
 * return the result in host memory.
 */
template <typename T>
T sum_all(const device_matrix<T>& x) {
  /*int r = x.getRows(),
      c = x.getCols();

  mat d_s = mat(1, r, 1) * x * mat(c, 1, 1);
  return hmat(d_s)[0];*/

  thrust::device_ptr<T> ptr(x.getData());
  return thrust::reduce(ptr, ptr + x.size());
}

/* \brief Explicit instantiation definition of template functions
 */

#define register_device_matrix_utility(T) \
  template device_matrix<T> operator &<T> (const device_matrix<T>& A, const device_matrix<T>& B); \
  template device_matrix<T>& operator &=<T> (device_matrix<T>& A, const device_matrix<T>& B); \
  template device_matrix<T> add_bias<T>(const device_matrix<T>& A, const T value, bool add_new_column); \
  template device_matrix<T> exp<T>(const device_matrix<T>& x); \
  template device_matrix<T> log<T>(const device_matrix<T>& x); \
  template device_matrix<T> log1pexp<T>(const device_matrix<T>& x); \
  template device_matrix<T> sigmoid<T>(const device_matrix<T>& x); \
  template device_matrix<T> d_sigmoid<T>(const device_matrix<T>& x); \
  template device_matrix<T> tanh<T>(const device_matrix<T>& x); \
  template device_matrix<T> d_tanh<T>(const device_matrix<T>& x); \
  template device_matrix<T> softmax<T>(const device_matrix<T>& x); \
  template device_matrix<T> relu(const device_matrix<T>& x); \
  template device_matrix<T> is_greater(const device_matrix<T>& x, const T value); \
  template device_matrix<T> MaxPerRow<T>(const device_matrix<T>& A); \
  template T sum_all<T>(const device_matrix<T>& A); \
  template void SubstractMaxPerRow<T>(device_matrix<T>& x);

register_device_matrix_utility(float);
