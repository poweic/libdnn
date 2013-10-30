#ifndef __DEVICE_BLAS_H_
#define __DEVICE_BLAS_H_

#include <device_matrix.h>

// ====================================
// ===== Vector Utility Functions =====
// ====================================
template <typename T>
T norm(const thrust::host_vector<T>& v) {
  return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), func::square<T>(), 0, thrust::plus<T>()) );
}

template <typename T>
T norm(const thrust::device_vector<T>& v) {
  return std::sqrt( thrust::transform_reduce(v.begin(), v.end(), func::square<T>(), 0, thrust::plus<T>()) );
}

template <typename T>
void print(const thrust::host_vector<T>& v, int precision = 4) {
  std::vector<T> stl_v(v.begin(), v.end());
  ::print(stl_v);
}

template <typename T>
void print(const thrust::device_vector<T>& v, int precision = 4) {
  thrust::host_vector<T> hv(v);
  print(hv, precision);
}

// =====================================
// ===== Matrix - Vector Operators =====
// =====================================

#define VECTOR thrust::device_vector
#define MATRIX device_matrix

/*template <typename T>
bool hasNAN(const MATRIX<T>& m) {
  Matrix2D<T> hm(m);
  for (size_t i=0; i < hm.getRows(); ++i )
    for (size_t j=0; j < hm.getCols(); ++j)
      if (hm[i][j] != hm[i][j])
	return true;
  return false;
}

template <typename T>
bool hasNAN(const VECTOR<T>& v) {
  thrust::host_vector<T> hv(v);
  for (size_t i=0; i < hv.size(); ++i)
    if (hv[i] != hv[i])
      return true;
  return false;
}*/

template <typename T>
MATRIX<T> operator * (const VECTOR<T>& col_vector, const VECTOR<T>& row_vector) {
  size_t m = col_vector.size();
  size_t n = row_vector.size();
  MATRIX<T> result(m, n);
  size_t k = 1;

  // Treat device_vector as an 1 by N matrix
  const T* cv = thrust::raw_pointer_cast(col_vector.data());
  const T* rv = thrust::raw_pointer_cast(row_vector.data());

  float alpha = 1.0, beta = 0.0;

  int lda = m;
  int ldb = 1;
  int ldc = m;

  cublasStatus_t status;
  status = cublasSgemm(dmat::_handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, cv, lda, rv, ldb, &beta, result.getData(), ldc);

  CCE(status);

  return result;
}

template <typename T>
VECTOR<T> operator & (const VECTOR<T>& x, const VECTOR<T>& y) {
  assert(x.size() == y.size());
  VECTOR<T> z(x.size());
  thrust::transform(x.begin(), x.end(), y.begin(), z.begin(), thrust::multiplies<T>());
  return z;
}

template <typename T>
VECTOR<T> operator * (const MATRIX<T>& m, const VECTOR<T>& v) {
  assert(m._cols == v.size());
  VECTOR<T> result(m._rows);

  float alpha = 1.0, beta = 0.0;
  int lda = m._rows;

  cublasStatus_t status;
  status = cublasSgemv(MATRIX<T>::_handle.get(), CUBLAS_OP_N, m._rows, m._cols, &alpha, m._data, lda, thrust::raw_pointer_cast(v.data()), STRIDE, &beta, thrust::raw_pointer_cast(result.data()), STRIDE);
  CCE(status);

  return result;
}

template <typename T>
VECTOR<T> operator * (const VECTOR<T>& v, const MATRIX<T>& m) {
  assert(v.size() == m._rows); 
  VECTOR<T> result(m._cols);

  float alpha = 1.0, beta = 0.0;
  int lda = m._rows;

  cublasStatus_t status;
  status = cublasSgemv(MATRIX<T>::_handle.get(), CUBLAS_OP_T, m._rows, m._cols, &alpha, m._data, lda, thrust::raw_pointer_cast(v.data()), STRIDE, &beta, thrust::raw_pointer_cast(result.data()), STRIDE);
  CCE(status);

  return result;
}

#undef VECTOR
#undef MATRIX

#define VECTOR thrust::device_vector
#define WHERE thrust
#include <functional.inl>
#include <blas.inl>
#undef VECTOR
#undef WHERE

#define VECTOR thrust::device_vector
#define MATRIX device_matrix

template <typename T>
MATRIX<T> operator & (const VECTOR<T>& v, const MATRIX<T>& m) {

  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  // FIXME !!!!!!!!!!!!!! THIS IS FUCKING SLOW !!!!!!!!!!!!!!!!! FIXME
  // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  assert(v.size() == m.getCols());
  MATRIX<T> result(m);

  size_t rows = m.getRows();
  size_t cols = m.getCols();

  for (size_t i=0; i<cols; ++i) {
    thrust::device_ptr<T> ptr(m.getData() + rows * i);
    thrust::device_ptr<T> ptr2(result.getData() + rows * i);

    VECTOR<T> cv(ptr, ptr + rows);
    VECTOR<T> cv2(ptr2, ptr2 + rows);

    thrust::transform(
	m.getData() + rows * i,
	m.getData() + rows * (i + 1),
	result.getData() + rows * i,
	func::ax<T>(v[i]) );
  }

  return result;
}

#undef VECTOR
#undef MATRIX
// #define VECTOR thrust::device_vector
// #define WHERE thrust
// // =====================================
// // ===== Multiplication Assignment =====
// // =====================================
// 
// template <typename T, typename U>
// VECTOR<T>& operator *= (VECTOR<T> &v, U val) {
//   WHERE::transform(v.begin(), v.end(), v.begin(), func::ax<T>(val));
//   return v;
// }
// 
// // [1 2 3] * 10 ==> [10 20 30]
// template <typename T, typename U>
// VECTOR<T> operator * (VECTOR<T> v, U val) {
//   return (v *= val);
// }
// 
// // 10 * [1 2 3] ==> [10 20 30]
// template <typename T, typename U>
// VECTOR<T> operator * (U val, VECTOR<T> v) {
//   return (v *= val);
// }
// // ===========================
// // ===== vector / scalar =====
// // ===========================
// template <typename T, typename U>
// VECTOR<T>& operator /= (VECTOR<T> &v, U val) {
//   v *= (T) (1) / val;
//   return v;
// }
// 
// // [10 20 30] / 10 ==> [1 2 3]
// template <typename T, typename U>
// VECTOR<T> operator / (VECTOR<T> v, U val) {
//   return (v /= val);
// }
// 
// // =================================
// // ======= scalar ./ vector ========
// // =================================
// // 10 / [1 2 5] ==> [10/1 10/2 10/5]
// template <typename T, typename U>
// VECTOR<T> operator / (U val, VECTOR<T> v) {
//   WHERE::transform(v.begin(), v.end(), v.begin(), func::adx<T>(val));
//   return v;
// }
// 
// template <typename T, typename U>
// VECTOR<T>& operator += (VECTOR<T> &v, U val) {
//   WHERE::transform(v.begin(), v.end(), v.begin(), func::apx<T>(val));
//   return v;
// }
// 
// // ===========================
// // ===== vector + scalar =====
// // ===========================
// // [1 2 3 4] + 5 ==> [6 7 8 9]
// template <typename T, typename U>
// VECTOR<T> operator + (VECTOR<T> v, U val) {
//   return (v += val);
// }
// 
// // ===========================
// // ===== scalar + VECTOR =====
// // ===========================
// // [1 2 3 4] + 5 ==> [6 7 8 9]
// template <typename T, typename U>
// VECTOR<T> operator + (U val, VECTOR<T> v) {
//   return (v += val);
// }
// 
// // =============================
// // ====== vector + vector ======
// // =============================
// // [1 2 3] + [2 3 4] ==> [3 5 7]
// template <typename T>
// VECTOR<T>& operator += (VECTOR<T> &v1, const VECTOR<T> &v2) {
//   WHERE::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), WHERE::plus<T>());
//   return v1;
// }
// template <typename T>
// VECTOR<T> operator + (VECTOR<T> v1, const VECTOR<T> &v2) {
//   return (v1 += v2);
// }
// 
// // ===========================
// // ===== vector - scalar =====
// // ===========================
// 
// template <typename T, typename U>
// VECTOR<T>& operator -= (VECTOR<T> &v, U val) {
//   v += -((T) val);
//   return v;
// }
// 
// // [1 2 3 4] - 1 ==> [0 1 2 3]
// template <typename T, typename U>
// VECTOR<T> operator - (VECTOR<T> v1, U val) {
//   return (v1 -= val);
// }
// 
// // ===========================
// // ===== scalar - vector =====
// // ===========================
// // 5 - [1 2 3 4] ==> [4 3 2 1]
// template <typename T, typename U>
// VECTOR<T> operator - (U val, VECTOR<T> v) {
//   WHERE::transform(v.begin(), v.end(), v.begin(), func::amx<T>(val));
//   return v;
// }
// 
// // =============================
// // ====== vector - vector ======
// // =============================
// // [2 3 4] - [1 2 3] ==> [1 1 1]
// template <typename T>
// VECTOR<T>& operator -= (VECTOR<T> &v1, const VECTOR<T> &v2) {
//   WHERE::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), WHERE::minus<T>());
//   return v1;
// }
// 
// template <typename T>
// VECTOR<T> operator - (VECTOR<T> v1, const VECTOR<T> &v2) {
//   return (v1 -= v2);
// }
// 
// #undef VECTOR
// #undef WHERE

#endif // __DEVICE_BLAS_H_
