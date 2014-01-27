#ifndef _DNN_UTILITY_H_
#define _DNN_UTILITY_H_

#include <arithmetic.h>
#include <math_ext.h>
#include <utility.h>
#include <map>

#ifdef __CUDACC__
  #include <device_math.h>
  #include <device_arithmetic.h>
  #define WHERE thrust
#endif

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

map<int, int> getLabelMapping(const mat& labels);
bool isLabeled(const mat& labels);

mat label2PosteriorProb(const mat& labels);
mat posteriorProb2Label(const mat& prob);

size_t zeroOneError(const mat& predict, const mat& label, ERROR_MEASURE errorMeasure);
mat& calcError(const mat& output, const mat& trainY, size_t offset = 0, size_t nData = 0);

vector<float> copyToHost(const mat& m);
size_t countDifference(const mat& m1, const mat& m2);

namespace ext {
  void rescale(mat& data, float lower, float upper);

  float max(const mat& v);
  float min(const mat& v);

  template <typename T>
  device_matrix<T> sigmoid(const device_matrix<T>& x) {
    device_matrix<T> s(x.getRows(), x.getCols());

    thrust::device_ptr<T> xPtr(x.getData());
    thrust::device_ptr<T> sPtr(s.getData());

    thrust::transform(xPtr, xPtr + x.size(), sPtr, func::sigmoid<float>());

    return s;
  }
};

template <typename T>
void memcpy2D(device_matrix<T>& dest, const device_matrix<T>& src, size_t r0, size_t c0, size_t h, size_t w, size_t r1, size_t c1) {

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

#endif // _DNN_UTILITY_H_
