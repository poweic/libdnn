#ifndef __VECTOR_BLAS_H__
#define __VECTOR_BLAS_H__

#include <vector>
#include <algorithm>
#include <functional>

#ifdef __GXX_EXPERIMENTAL_CXX0X__
#define ASSERT_NOT_SCALAR(T) {static_assert(std::is_scalar<T>::value, "val must be scalar");} 
#else
#define ASSERT_NOT_SCALAR(T) {}
#endif

using namespace std;

template <typename T>
vector<T> operator & (const vector<T>& x, const vector<T>& y) {
  assert(x.size() == y.size());
  vector<T> z(x.size());
  std::transform (x.begin(), x.end(), y.begin(), z.begin(), std::multiplies<float>());
  return z;
}

#define VECTOR std::vector
#define WHERE std
#include <functional.inl>
#include <operators.inl>
#undef VECTOR
#undef WHERE

#endif // __VECTOR_BLAS_H__
