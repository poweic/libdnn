#ifndef __DEVICE_MATH_EXT_H_
#define __DEVICE_MATH_EXT_H_

#include <math_ext.h>
#include <device_matrix.h>

namespace ext {
  template <typename T>
  vector<T> toStlVector(const thrust::device_vector<T>& v) {
    vector<T> stl_vector(v.size());
    thrust::copy(v.begin(), v.end(), stl_vector.begin());
    return stl_vector;
  }

  template <typename T>
  thrust::device_vector<T> toDeviceVector(const vector<T>& v) {
    return thrust::device_vector<T>(v.begin(), v.end());
  }

  // ========================
  // ===== Save as File =====
  // ========================
  template <typename T>
  void save(const thrust::device_vector<T>& v, string filename) {
    ext::save(toStlVector(v), filename);
  }

  // ==========================
  // ===== Load from File =====
  // ==========================
  template <typename T>
  void load(thrust::device_vector<T>& v, string filename) {
    vector<T> hv;
    ext::load<T>(hv, filename);
    v = thrust::device_vector<T>(hv.begin(), hv.end());
  }
  // =================================
  // ===== Summation over Vector =====
  // =================================
  template <typename T>
  T sum(const thrust::device_vector<T>& v) {
    return thrust::reduce(v.begin(), v.end());
  }

  template <typename T>
  T sum(const device_matrix<T>& m) {
    return thrust::reduce(m.getData(), m.getData() + m.size(), (T) 0, thrust::plus<T>());
  }

  // ================
  // ===== Rand =====
  // ================
  template <typename T>
  void rand(device_matrix<T>& m) {
    Matrix2D<T> h_m(m.getRows(), m.getCols());
    rand(h_m);
    m = device_matrix<T>(h_m);
  }

  template <typename T>
  void randn(device_matrix<T>& m) {
    Matrix2D<T> h_m(m.getRows(), m.getCols());
    ext::randn(h_m);
    m = device_matrix<T>(h_m);
  }
  // ===================
  // ===== SoftMax =====
  // ===================
  template <typename T>
  thrust::device_vector<T> softmax(const thrust::device_vector<T>& x) {
    thrust::device_vector<T> s(x.size());

    thrust::transform(x.begin(), x.end(), s.begin(), func::exp<T>());

    T denominator = 1.0 / ext::sum(s);
    thrust::transform(s.begin(), s.end(), s.begin(), func::ax<T>(denominator));

    return s;
  }

  // ============================
  // ===== Sigmoid Function =====
  // ============================
  template <typename T>
  thrust::device_vector<T> sigmoid(const thrust::device_vector<T>& x) {
    thrust::device_vector<T> s(x.size());
    thrust::transform(x.begin(), x.end(), s.begin(), func::sigmoid<T>());
    return s;
  }

  // ================================
  // ===== Biased after Sigmoid =====
  // ================================
  template <typename T>
  thrust::device_vector<T> b_sigmoid(const thrust::device_vector<T>& x) {
    thrust::device_vector<T> s(x.size() + 1);
    thrust::transform(x.begin(), x.end(), s.begin(), func::sigmoid<T>());
    s.back() = 1.0;
    return s;
  }
};

#endif // __DEVICE_MATH_EXT_H_
