#ifndef __DNN_H_
#define __DNN_H_

#include <arithmetic.h>

#ifndef __CUDACC__

  #include <arithmetic.h>
  #include <math_ext.h>
  #include <matrix.h>
  typedef Matrix2D<float> mat;
  typedef std::vector<float> vec;
  #define WHERE std

#else

  #include <device_matrix.h>
  #include <device_math_ext.h>
  #include <device_arithmetic.h>
  
  #include <thrust/transform_reduce.h>
  #include <thrust/functional.h>
  #include <thrust/host_vector.h>
  #include <thrust/device_vector.h>
  typedef device_matrix<float> mat;
  typedef thrust::device_vector<float> vec;
  #define WHERE thrust

#endif

#define dsigma(x) ((x) & ((float) 1.0 - (x)))

class DNN {
public:
  DNN();
  DNN(string fn);
  DNN(const std::vector<size_t>& dims);
  DNN(const DNN& source);
  DNN& operator = (DNN rhs);

  void randInit();
  void feedForward(const mat& x, std::vector<mat>* hidden_output);

  void backPropagate(vec& p, std::vector<vec>& hidden_output, std::vector<mat>& gradient);
  void backPropagate(mat& p, std::vector<mat>& hidden_output, std::vector<mat>& gradient, const vec& coeff);

  void updateParameters(std::vector<mat>& gradient, float learning_rate = 1e-3);

  size_t getNLayer() const;
  size_t getDepth() const;
  void getEmptyGradient(std::vector<mat>& g) const;

  void read(string fn);
  void save(string fn) const;
  void print() const;

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

#define HIDDEN_OUTPUT_ALIASING(O, x, y, z, w) \
std::vector<vec>& x	= O.hox; \
std::vector<vec>& y	= O.hoy; \
vec& z		= O.hoz; \
std::vector<vec>& w	= O.hod;

#define GRADIENT_REF(g, g1, g2, g3, g4) \
std::vector<mat>& g1	= g.grad1; \
std::vector<mat>& g2 = g.grad2; \
vec& g3		= g.grad3; \
std::vector<mat>& g4 = g.grad4;

#define GRADIENT_CONST_REF(g, g1, g2, g3, g4) \
const std::vector<mat>& g1	= g.grad1; \
const std::vector<mat>& g2 = g.grad2; \
const vec& g3		= g.grad3; \
const std::vector<mat>& g4 = g.grad4;

class HIDDEN_OUTPUT {
  public:
    std::vector<vec> hox;
    std::vector<vec> hoy;
    vec hoz;
    std::vector<vec> hod;
};

void swap(HIDDEN_OUTPUT& lhs, HIDDEN_OUTPUT& rhs);

class GRADIENT {
  public:
    std::vector<mat> grad1;
    std::vector<mat> grad2;
    vec grad3;
    std::vector<mat> grad4;
};

void swap(GRADIENT& lhs, GRADIENT& rhs);

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

#endif  // __DNN_H_
