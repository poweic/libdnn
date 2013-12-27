#ifndef __DNN_H_
#define __DNN_H_

#define mylog(x) { cout << #x << " = " << x << endl; }

#include <arithmetic.h>
#include <math_ext.h>

#ifndef __CUDACC__

  #include <arithmetic.h>
  #include <matrix_math.h>
  #include <matrix.h>
  typedef Matrix2D<float> mat;
  typedef std::vector<float> vec;
  #define WHERE std

#else

  #include <device_matrix.h>
  #include <device_math.h>
  #include <device_arithmetic.h>
  
  #include <thrust/transform_reduce.h>
  #include <thrust/functional.h>
  #include <thrust/host_vector.h>
  #include <thrust/device_vector.h>
  #include <thrust/inner_product.h>
  typedef device_matrix<float> mat;
  typedef thrust::device_vector<float> vec;

  #define WHERE thrust

namespace ext {
  template <typename T>
  device_matrix<T> b_sigmoid(const device_matrix<T>& x) {
    device_matrix<T> s(x.getRows(), x.getCols() + 1);
    
    thrust::device_ptr<T> xPtr(x.getData());
    thrust::device_ptr<T> sPtr(s.getData());

    // Leave last column in s untouched
    thrust::transform(xPtr, xPtr + x.size(), sPtr, func::sigmoid<float>());

    // Fill last column in s with 1.0
    thrust::fill(sPtr + s.size() - s.getRows(), sPtr + s.size(), (float) 1.0);

    return s;
  }

  template <typename T>
  device_matrix<T> sigmoid(const device_matrix<T>& x) {
    device_matrix<T> s(x.getRows(), x.getCols());

    thrust::device_ptr<T> xPtr(x.getData());
    thrust::device_ptr<T> sPtr(s.getData());

    thrust::transform(xPtr, xPtr + x.size(), sPtr, func::sigmoid<float>());

    return s;
  }
}

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
device_matrix<T> add_bias(const device_matrix<T>& A) {
  device_matrix<T> B(A.getRows(), A.getCols() + 1);

  B += 1.0;

  device_matrix<T>::cublas_geam(
      CUBLAS_OP_N, CUBLAS_OP_N,
      A.getRows(), A.getCols(),
      1.0, A.getData(), A.getRows(),
      0.0, B.getData(), B.getRows(),
      B.getData(), B.getRows()
  );

  return B;
}

#endif

#define dsigma(x) ((x) & ((float) 1.0 - (x)))

class DataSet {
public:
  DataSet(mat& dataX, mat& dataY):
    X(dataX), y(dataY), nData(dataX.getRows()) {}

  mat& X, &y;
  size_t nData;
};

enum ERROR_MEASURE {
  L2ERROR,  /* for binary-classification only */
  CROSS_ENTROPY
};


class DNN {
public:
  DNN();
  DNN(string fn);
  DNN(const std::vector<size_t>& dims);
  DNN(const DNN& source);
  DNN& operator = (DNN rhs);

  void feedForward(const mat& x, std::vector<mat>* hidden_output, size_t offset = 0, size_t batchSize = 0);

  void backPropagate(mat& delta, std::vector<mat>& hidden_output, std::vector<mat>& gradient);

  void updateParameters(std::vector<mat>& gradient, float learning_rate = 1e-3);

  size_t getNLayer() const;
  size_t getDepth() const;
  void getEmptyGradient(std::vector<mat>& g) const;

  void _read(FILE* fid);
  void read(string fn);
  void save(string fn) const;
  void print() const;

  void train(const DataSet& train, const DataSet& valid, size_t batchSize, ERROR_MEASURE err);

  std::vector<mat>& getWeights();
  const std::vector<mat>& getWeights() const;
  std::vector<size_t>& getDims();
  const std::vector<size_t>& getDims() const;

  friend void swap(DNN& lhs, DNN& rhs);

private:
  std::vector<size_t> _dims;
  std::vector<mat> _weights;
};

void swap(DNN& lhs, DNN& rhs);

template <typename T>
vector<T> add_bias(const vector<T>& v) {
  vector<T> vb(v.size() + 1);
  WHERE::copy(v.begin(), v.end(), vb.begin());
  vb.back() = 1.0;
  return vb;
}

template <typename T>
void remove_bias(vector<T>& v) {
  v.pop_back();
}

template <typename T>
Matrix2D<T> add_bias(const Matrix2D<T>& A) {
  Matrix2D<T> B(A.getRows(), A.getCols() + 1);

  for (size_t i=0; i<B.getRows(); ++i) {
    for (size_t j=0; j<B.getCols(); ++j)
      B[i][j] = A[i][j];
    B[i][B.getCols()] = 1;
  }
  return B;
}

template <typename T>
void remove_bias(Matrix2D<T>& A) {
  Matrix2D<T> B(A.getRows(), A.getCols() - 1);

  for (size_t i=0; i<B.getRows(); ++i)
    for (size_t j=0; j<B.getCols(); ++j)
      B[i][j] = A[i][j];

  A = B;
}

mat l2error(mat& targets, mat& predicts);

void print(const thrust::host_vector<float>& hv);
void print(const mat& m);
void print(const thrust::device_vector<float>& dv);

#endif  // __DNN_H_
