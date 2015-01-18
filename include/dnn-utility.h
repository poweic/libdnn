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
    dim3 grids( ceil( (float) rows / NV_DEVICE_WARP_SIZE), ceil( (float) cols / NV_DEVICE_WARP_SIZE)); \
    dim3 threads(NV_DEVICE_WARP_SIZE, NV_DEVICE_WARP_SIZE);

#endif

#define assert_nan(x) { if (hasNAN(x)) \
  throw std::runtime_error(RED_ERROR + #x " has NaN"); }

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

#include <host_matrix.h>

#include <curand.h>
#include <curand_kernel.h>

typedef host_matrix<float> hmat;

std::map<int, int> getLabelMapping(const hmat& labels);

mat getError(const mat& target, const mat& output, ERROR_MEASURE errorMeasure);
mat posteriorProb2Label(const mat& prob);

size_t zeroOneError(const mat& predict, const mat& label);

class CURAND_STATE {
public:
  CURAND_STATE(unsigned seed = unsigned(time(NULL)), int N = 32);
  curandState* get() const;

  ~CURAND_STATE();

private:
  curandState* _states;
};

typedef void (*Operation)(float&, curandState*);
__global__ void setupCuRandState( curandState * state, unsigned long seed );

enum UNIT_TYPE {
  BERNOULLI = 1,
  GAUSSIAN = 2
};

void sample(mat &prob, UNIT_TYPE type);

mat randn(int m, int n);
mat rand(int m, int n);

inline mat zeros(int m, int n) { return mat(m, n, 0); }
inline mat ones(int m, int n) { return mat(m, n, 1); }

std::vector<float> copyToHost(const mat& m);
size_t countDifference(const mat& m1, const mat& m2);

namespace ext {
  void rescale(mat& data, float lower, float upper);

  float max(const mat& v);
  float min(const mat& v);

  float max(const hmat& v);
  float min(const hmat& v);
};

template <typename T>
void SubstractMaxPerRow(device_matrix<T>& x);

template <typename T>
device_matrix<T> MaxPerRow(const device_matrix<T>& A);

template <typename T>
bool hasNAN(const device_matrix<T>& x) {

  host_matrix<T> h_x(x);

  for (int i=0; i<h_x.getRows(); ++i)
    for (int j=0; j<h_x.getCols(); ++j)
      if (h_x(i, j) != h_x(i, j))
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

  device_matrix<T>::cublas_geam(
      CUBLAS_OP_N, CUBLAS_OP_N,
      h, w,
      1.0, src.getData() + c0 * src.getRows() + r0, src.getRows(),
      0.0, dest.getData(), dest.getRows(),
      dest.getData() + c1 * dest.getRows() + r1, dest.getRows());
}

/*template <typename T>
device_matrix<T> vercat(const std::vector<device_matrix<T> >& matrices, 
    bool reserve = false) {

  size_t rows = matrices[0].getRows(),
	 cols = matrices[0].getCols();

  int R = matrices.size() * rows;
  int C = cols;

  if (reserve) R += 1;

  mat result(R, C);
  for (size_t i=0; i<matrices.size(); ++i)
    memcpy2D<T>(result, matrices[i], 0, 0, rows, cols, i*rows, 0);
  return result;
}*/

/*! \brief vertical split (the inverse function of vercat)
 *  versplit will ignore any pre-reserved row in the input matrix (i.e. big)
 *  Because block_rows is the quotient of two integer, which is always floor.
 *  (see C++ 6.5.5 Multiplicative operators)
 * \param big		the input matrix
 * \param n_sub_matrix	split into how many sub matrices vertically.
 * \param block_rows    # of rows in each sub matrices
 *
 * */
template <typename T>
std::vector<device_matrix<T> > versplit(const device_matrix<T>& big,
    size_t n_sub_matrix, size_t block_rows) {

  std::vector<device_matrix<T> > blocks(n_sub_matrix);
  for (size_t i=0; i<n_sub_matrix; ++i) {
    blocks[i].resize(block_rows, big.getCols());
    memcpy2D(blocks[i], big, i * block_rows, 0, block_rows, big.getCols(), 0, 0);
  }

  return blocks;
}

template <typename T>
device_matrix<T> add_bias(const device_matrix<T>& A, const T value = 1,
    bool add_new_column = false) {

  if (add_new_column) {
    device_matrix<T> B(A.getRows() + 1, A.getCols(), 1.0f);
    memcpy2D<T>(B, A, 0, 0, A.getRows(), A.getCols(), 0, 0);
    return B;
  }
  else {
    device_matrix<T> B(A);
    device_matrix<T> bias(1, B.getCols(), value);
    memcpy2D<T>(B, bias, 0, 0, 1, B.getCols(), B.getRows() - 1, 0);
    return B;
  }
}

template <typename T>
device_matrix<T> remove_bias(const device_matrix<T>& x) {
  device_matrix<T> y(x.getRows() - 1, x.getCols());
  memcpy2D<T>(y, x, 0, 0, x.getRows() - 1, x.getCols(), 0, 0);
  return y;
}

// convert a linear index to a row index
template <typename T>
struct linear_index_to_col_index: public thrust::unary_function<T, T> {
  T rows; // number of rows

  __host__ __device__ linear_index_to_col_index(T rows) : rows(rows) {}

  __host__ __device__ T operator() (T i) { return i % rows; }
};

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T, T> {
  T rows; // number of rows

  __host__ __device__ linear_index_to_row_index(T rows) : rows(rows) {}

  __host__ __device__ T operator() (T i) { return i / rows; }
};

namespace func {
  template <typename T>
  struct max {
    const T a;
    max(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T& x) const { return x > a ? x : a; }
  };

  template <typename T>
  struct greater {
    const T a;
    greater(T _a) : a(_a) {}
    __host__ __device__ T operator()(const T& x) const { return x > a ? 1 : 0; }
  };
  
  template <typename T>
  struct d_sigmoid {
    d_sigmoid() {}
    __host__ __device__ T operator()(const T& y) const {
      return (1 - y) * y;
    }
  };

  template <typename T>
  struct hyperbolic_tangent {
    hyperbolic_tangent() {}
    __host__ __device__ T operator()(const T& x) const {
      if (x > 9) return 1;
      if (x < -9) return 0;

      T a = expf(2 * x);
      return (a - 1) / (a + 1);
    }
  };

  // Though the derivative of tanh(x) is sech^2(x), it's faster to compute if we
  // use: 1 - y^2, where y is tanh(x) and already computed before.
  template <typename T>
  struct d_hyperbolic_tangent {
    d_hyperbolic_tangent () {}
    __host__ __device__ T operator()(const T& y) const {
      return 1 - y * y;
    }
  };

};


template <typename T>
device_matrix<T> operator & (const device_matrix<T>& A, const device_matrix<T>& B);

template <typename T>
device_matrix<T>& operator &= (device_matrix<T>& A, const device_matrix<T>& B);

template <typename T> device_matrix<T> exp(const device_matrix<T>& x);

template <typename T> device_matrix<T> log(const device_matrix<T>& x);

template <typename T> device_matrix<T> log1pexp(const device_matrix<T>& x);

template <typename T> device_matrix<T> sigmoid(const device_matrix<T>& x);

template <typename T> device_matrix<T> d_sigmoid(const device_matrix<T>& x);

template <typename T> device_matrix<T> tanh(const device_matrix<T>& x);

template <typename T> device_matrix<T> d_tanh(const device_matrix<T>& x);

template <typename T> device_matrix<T> softmax(const device_matrix<T>& x);

template <typename T> device_matrix<T> relu(const device_matrix<T>& x);

template <typename T> device_matrix<T> is_greater(const device_matrix<T>& x, const T value);

template <typename T> T sum_all(const device_matrix<T>& x);

template <typename T, typename UnaryFunction>
device_matrix<T> transform(const device_matrix<T>& x, UnaryFunction op) {
  device_matrix<T> s(x.getRows(), x.getCols());

  thrust::device_ptr<T> xPtr(x.getData());
  thrust::device_ptr<T> sPtr(s.getData());

  thrust::transform(xPtr, xPtr + x.size(), sPtr, op);

  return s;
}

#endif // _DNN_UTILITY_H_
