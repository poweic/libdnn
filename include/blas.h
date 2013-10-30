#ifndef __VECTOR_BLAS_H__
#define __VECTOR_BLAS_H__

#include <utility.h>
#include <vector>
#include <matrix.h>
#include <functional>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#define ASSERT_NOT_SCALAR(T) {static_assert(std::is_scalar<T>::value, "val must be scalar");} 
#else
#define ASSERT_NOT_SCALAR(T) {}
#endif

using namespace std;

// =====================================
// ===== Matrix - Vector Operators =====
// =====================================
template <typename T>
Matrix2D<T> operator * (const vector<T>& col_vector, const vector<T>& row_vector) {

  Matrix2D<T> m(col_vector.size(), row_vector.size());

  foreach (i, col_vector)
    foreach (j, row_vector)
      m[i][j] = col_vector[i] * row_vector[j];

  return m;
}

template <typename T>
vector<T> operator & (const vector<T>& x, const vector<T>& y) {
  assert(x.size() == y.size());
  vector<T> z(x.size());
  std::transform (x.begin(), x.end(), y.begin(), z.begin(), std::multiplies<float>());
  return z;
}

template <typename T>
vector<T> operator * (const Matrix2D<T>& A, const vector<T>& col_vector) {
  assert(A.getCols() == col_vector.size());

  vector<T> y(A.getRows());
  size_t cols = A.getCols();

  foreach (i, y) {
    for (size_t j=0; j<cols; ++j)
      y[i] += col_vector[j] * A[i][j];
  }

  return y;
}

template <typename T>
vector<T> operator * (const vector<T>& row_vector, const Matrix2D<T>& A) {
  assert(row_vector.size() == A.getRows());

  vector<T> y(A.getCols());
  size_t rows = A.getRows();

  foreach (i, y) {
    for (size_t j=0; j<rows; ++j)
      y[i] += row_vector[j] * A[j][i];
  }

  return y;
}

#define VECTOR std::vector
#define WHERE std
#include <functional.inl>
#include <blas.inl>
#undef VECTOR
#undef WHERE

template <typename T>
Matrix2D<T> operator & (const Matrix2D<T>& A, const Matrix2D<T>& B) {
  assert(A.getRows() == B.getRows() && A.getCols() == B.getCols());

  Matrix2D<T> C(A.getRows(), A.getCols());

  range (i, A.getRows())
    range (j, A.getCols())
      C[i][j] = A[i][j] * B[i][j];

  return C;
}

template <typename T>
Matrix2D<T> operator & (const Matrix2D<T>& A, const vector<T>& v) {
  assert(A.getRows() == v.size());

  size_t rows = A.getRows();
  size_t cols = A.getCols();
  Matrix2D<T> result(rows, cols);
  for (size_t i=0; i<rows; ++i)
    std::transform(A[i], A[i] + cols, result[i], func::ax<T>(v[i]));

  return result;
}

template <typename T>
Matrix2D<T> operator - (T c, Matrix2D<T> m) {

  Matrix2D<T> result(m.getRows(), m.getCols());

  range (i, m.getRows())
    range (j, m.getCols())
      result[i][j] = c - m[i][j];
  return result;
}

// ===================================
void blas_testing_examples();

#endif // __VECTOR_BLAS_H__
