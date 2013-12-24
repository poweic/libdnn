#include <matrix.h>

namespace ext {
  template <typename T>
  T sum(const Matrix2D<T>& m) {
    T s = 0;
    for (size_t i=0; i<m.getRows(); ++i)
      for (size_t j=0; j<m.getCols(); ++j)
	s += m[i][j];

    return s;
  }

  template <typename T>
  void rand(Matrix2D<T>& m) {

    for (size_t i=0; i<m.getRows(); ++i)
      for (size_t j=0; j<m.getCols(); ++j)
	m[i][j] = rand01<T>();
  }

  template <typename T>
  void randn(Matrix2D<T>& m) {

    for (size_t i=0; i<m.getRows(); ++i)
      for (size_t j=0; j<m.getCols(); ++j)
	m[i][j] = randn<T>(0, 1);
  }

  template <typename T>
  Matrix2D<T> sigmoid(const Matrix2D<T>& x) {
    Matrix2D<T> s(x.getRows(), x.getCols());

    for (size_t i=0; i<x.getRows(); ++i)
      std::transform(x[i], x[i] + x.getCols(), s[i], func::sigmoid<T>());

    return s;
  }

  template <typename T>
  Matrix2D<T> b_sigmoid(const Matrix2D<T>& x) {
    Matrix2D<T> s(x.getRows(), x.getCols() + 1);


    for (size_t i=0; i<x.getRows(); ++i) {
      std::transform(x[i], x[i] + x.getCols(), s[i], func::sigmoid<T>());
      s[i][x.getCols()] = 1.0;
    }

    return s;
  }
};
