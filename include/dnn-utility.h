#ifndef _DNN_UTILITY_H_
#define _DNN_UTILITY_H_

#include <math_ext.h>
#include <utility.h>
#include <map>

#ifdef __CUDACC__
  #include <device_math.h>
  #include <device_arithmetic.h>
  #define WHERE thrust

  #define NV_DEVICE_WARP_SIZE 32

  #define ALLOCATE_GRIDS_AND_THREADS(rows, cols) \
    dim3 grids( ceil( (float) cols / NV_DEVICE_WARP_SIZE), ceil( (float) rows / NV_DEVICE_WARP_SIZE)); \
    dim3 threads(NV_DEVICE_WARP_SIZE, NV_DEVICE_WARP_SIZE);

#endif

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <host_matrix.h>

#include <curand.h>
#include <curand_kernel.h>

typedef host_matrix<float> hmat;

map<int, int> getLabelMapping(const hmat& labels);

mat getError(const mat& target, const mat& output, ERROR_MEASURE errorMeasure);
mat posteriorProb2Label(const mat& prob);

size_t zeroOneError(const mat& predict, const mat& label, ERROR_MEASURE errorMeasure);
// mat& calcError(const mat& output, const mat& trainY, size_t offset = 0, size_t nData = 0);

class CURAND_STATE {
public:
  CURAND_STATE(unsigned seed = unsigned(time(NULL)), int N = 32);
  curandState* get() const;

  ~CURAND_STATE();

private:
  curandState* _states;
};

typedef __device__ void (*Operation)(float&, curandState*);
__global__ void setupCuRandState( curandState * state, unsigned long seed );

mat randn(int m, int n);
mat rand(int m, int n);

vector<float> copyToHost(const mat& m);
size_t countDifference(const mat& m1, const mat& m2);

namespace ext {
  void rescale(mat& data, float lower, float upper);

  float max(const mat& v);
  float min(const mat& v);

  float max(const hmat& v);
  float min(const hmat& v);
};

template <typename T>
bool hasNAN(const host_matrix<T>& x) {

  for (int i=0; i<x.getRows(); ++i)
    for (int j=0; j<x.getCols(); ++j)
      if (x(i, j) != x(i, j))
	return true;

  return false;
}

/*! \brief Copy a block memory.
 * Copy a block (of size h by w) of memory from src to dest.
 * Both the number of rows of src and dest must be greater or equal to h.
 * Both the number of cols of src and dest must be greater or equal to w.
 *
 * \param dest	destination device matrix.
 * \param src	source device matrix.
 * \param r0	source row id. (0-based)
 * \param c0	source column id. (0-based)
 * \param h	height of the block of memory to be copied.
 * \param w	width of the block of memory to be copied.
 * \param r1	destination row id. (0-based) i.e. the position to paste.
 * \param c1	destination column id. (0-based) i.e. the position to paste.
 * */
template <typename T>
void memcpy2D(device_matrix<T>& dest, const device_matrix<T>& src,
    size_t r0, size_t c0, size_t h, size_t w, size_t r1, size_t c1) {

  device_matrix<float>::cublas_geam(
      CUBLAS_OP_N, CUBLAS_OP_N,
      h, w,
      1.0, src.getData() + c0 * src.getRows() + r0, src.getRows(),
      0.0, dest.getData(), dest.getRows(),
      dest.getData() + c1 * dest.getRows() + r1, dest.getRows());
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

device_matrix<float> log(const device_matrix<float>& x);

template <typename T>
device_matrix<T> sigmoid(const device_matrix<T>& x) {
  return transform(x, func::sigmoid<T>());
}

template <typename T, typename UnaryFunction>
device_matrix<T> transform(const device_matrix<T>& x, UnaryFunction op) {
  device_matrix<T> s(x.getRows(), x.getCols());

  thrust::device_ptr<T> xPtr(x.getData());
  thrust::device_ptr<T> sPtr(s.getData());

  thrust::transform(xPtr, xPtr + x.size(), sPtr, op);

  return s;
}

#endif // _DNN_UTILITY_H_
