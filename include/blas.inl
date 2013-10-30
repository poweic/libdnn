// =====================================
// ===== Multiplication Assignment =====
// =====================================
template <typename T, typename U>
VECTOR<T>& operator *= (VECTOR<T> &v, U val) {
  ASSERT_NOT_SCALAR(U);
  WHERE::transform(v.begin(), v.end(), v.begin(), func::ax<T>(val));
  return v;
}

// [1 2 3] * 10 ==> [10 20 30]
template <typename T, typename U>
VECTOR<T> operator * (VECTOR<T> v, U val) {
  ASSERT_NOT_SCALAR(U);
  return (v *= val);
}

// 10 * [1 2 3] ==> [10 20 30]
template <typename T, typename U>
VECTOR<T> operator * (U val, VECTOR<T> v) {
  ASSERT_NOT_SCALAR(U);
  return (v *= val);
}
// ===========================
// ===== vector / scalar =====
// ===========================
template <typename T, typename U>
VECTOR<T>& operator /= (VECTOR<T> &v, U val) {
  ASSERT_NOT_SCALAR(U);
  v *= (T) (1) / val;
  return v;
}

// [10 20 30] / 10 ==> [1 2 3]
template <typename T, typename U>
VECTOR<T> operator / (VECTOR<T> v, U val) {
  ASSERT_NOT_SCALAR(U);
  return (v /= val);
}

// =================================
// ======= scalar ./ vector ========
// =================================
// 10 / [1 2 5] ==> [10/1 10/2 10/5]
template <typename T, typename U>
VECTOR<T> operator / (U val, VECTOR<T> v) {
  ASSERT_NOT_SCALAR(U);
  WHERE::transform(v.begin(), v.end(), v.begin(), func::adx<T>(val));
  return v;
}

template <typename T, typename U>
VECTOR<T>& operator += (VECTOR<T> &v, U val) {
  ASSERT_NOT_SCALAR(U);
  WHERE::transform(v.begin(), v.end(), v.begin(), func::apx<T>(val));
  return v;
}

// ===========================
// ===== vector + scalar =====
// ===========================
// [1 2 3 4] + 5 ==> [6 7 8 9]
template <typename T, typename U>
VECTOR<T> operator + (VECTOR<T> v, U val) {
  ASSERT_NOT_SCALAR(U);
  return (v += val);
}

// ===========================
// ===== scalar + vector =====
// ===========================
// [1 2 3 4] + 5 ==> [6 7 8 9]
template <typename T, typename U>
VECTOR<T> operator + (U val, VECTOR<T> v) {
  ASSERT_NOT_SCALAR(U);
  return (v += val);
}

// ===========================
// ===== vector - scalar =====
// ===========================

template <typename T, typename U>
VECTOR<T>& operator -= (VECTOR<T> &v, U val) {
  ASSERT_NOT_SCALAR(U);
  v += -((T) val);
  return v;
}

// [1 2 3 4] - 1 ==> [0 1 2 3]
template <typename T, typename U>
VECTOR<T> operator - (VECTOR<T> v1, U val) {
  ASSERT_NOT_SCALAR(U);
  return (v1 -= val);
}

// ===========================
// ===== scalar - vector =====
// ===========================
// 5 - [1 2 3 4] ==> [4 3 2 1]
template <typename T, typename U>
VECTOR<T> operator - (U val, VECTOR<T> v) {
  ASSERT_NOT_SCALAR(U);
  WHERE::transform(v.begin(), v.end(), v.begin(), func::amx<T>(val));
  return v;
}

// =============================
// ====== vector + vector ======
// =============================
// [1 2 3] + [2 3 4] ==> [3 5 7]
template <typename T>
VECTOR<T>& operator += (VECTOR<T> &v1, const VECTOR<T> &v2) {
  assert(v1.size() == v2.size());
  WHERE::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), WHERE::plus<T>());
  return v1;
}

template <typename T>
VECTOR<T> operator + (VECTOR<T> v1, const VECTOR<T> &v2) {
  return (v1 += v2);
}

// =============================
// ====== vector - vector ======
// =============================
// [2 3 4] - [1 2 3] ==> [1 1 1]
template <typename T>
VECTOR<T>& operator -= (VECTOR<T> &v1, const VECTOR<T> &v2) {
  assert(v1.size() == v2.size());
  WHERE::transform(v1.begin(), v1.end(), v2.begin(), v1.begin(), WHERE::minus<T>());
  return v1;
}

template <typename T>
VECTOR<T> operator - (VECTOR<T> v1, const VECTOR<T> &v2) {
  return (v1 -= v2);
}
