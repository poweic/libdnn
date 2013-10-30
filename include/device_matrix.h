#ifndef __DEVICE_MATRIX_H__
#define __DEVICE_MATRIX_H__

#include <matrix.h>
#include <cassert>
#include <string>

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/* Includes, cuda */
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <helper_cuda.h>
using namespace std;

#define CCE(x) checkCudaErrors(x)

#define host_matrix Matrix2D
#define STRIDE (sizeof(T) / sizeof(float))

class CUBLAS_HANDLE {
public:
  CUBLAS_HANDLE()  { CCE(cublasCreate(&_handle)); }
  ~CUBLAS_HANDLE() { CCE(cublasDestroy(_handle)); }

  cublasHandle_t& get() { return _handle; }
private:
  cublasHandle_t _handle;
};

template <typename T>
class device_matrix {
public:
  // default constructor 
  device_matrix();

  device_matrix(size_t r, size_t c);

  // Load from file. Ex: *.mat in text-form
  device_matrix(const string& filename);

  // Copy Constructor 
  device_matrix(const device_matrix<T>& source);

  // Constructor from Host Matrix
  device_matrix(const host_matrix<T>& h_matrix);

  ~device_matrix();

  // ===========================
  // ===== Other Functions =====
  // ===========================
  
  // ===== Addition =====
  // device_matrix<T>& operator += (T val) { return *this; } 
  // device_matrix<T> operator + (T val) const { return *this; }
  
  device_matrix<T>& operator += (const device_matrix<T>& rhs);
  device_matrix<T> operator + (const device_matrix<T>& rhs) const;

  // ===== Substraction =====
  // device_matrix<T>& operator -= (T val) { return *this; }
  // device_matrix<T> operator - (T val) const { return *this; }
  
  device_matrix<T>& operator -= (const device_matrix<T>& rhs);
  device_matrix<T> operator - (const device_matrix<T>& rhs) const;

  // ===== Division =====
  device_matrix<T>& operator /= (T alpha);
  device_matrix<T> operator / (T alpha) const;
  
  // ===== Matrix-scalar Multiplication =====
  device_matrix<T>& operator *= (T alpha);
  device_matrix<T> operator * (T alpha) const;
  // ===== Matrix-Vector Multiplication =====
  template <typename S>
  friend thrust::device_vector<S> operator * (const thrust::device_vector<S>& rhs, const device_matrix<S>& m);
  template <typename S>
  friend thrust::device_vector<S> operator * (const device_matrix<S>& m, const thrust::device_vector<S>& rhs);
  // ===== Matrix-Matrix Multiplication =====
  device_matrix<T>& operator *= (const device_matrix<T>& rhs);
  device_matrix<T> operator * (const device_matrix<T>& rhs) const;

  operator host_matrix<T>() const;

  template <typename S>
  friend void swap(device_matrix<S>& lhs, device_matrix<S>& rhs);

  template <typename S>
  friend S L1_NORM(const device_matrix<S>& A, const device_matrix<S>& B);

  friend void sgemm(const device_matrix<float>& A, const device_matrix<float>& B, device_matrix<float>& C, float alpha, float beta);

  friend void sgeam(const device_matrix<float>& A, const device_matrix<float>& B, device_matrix<float>& C, float alpha, float beta);

  // Operator Assignment:
  // call copy constructor first, and swap with the temp variable
  device_matrix<T>& operator = (device_matrix<T> rhs);

  void _init();
  void resize(size_t r, size_t c);
  void print(size_t precision = 5) const ;

  void fillwith(T val) {
    thrust::device_ptr<T> ptr(_data);
    thrust::fill(ptr, ptr + this->size(), val);
  }

  size_t size() const { return _rows * _cols; }
  size_t getRows() const { return _rows; }
  size_t getCols() const { return _cols; }
  T* getData() const { return _data; }
  void saveas(const string& filename) const;

  static CUBLAS_HANDLE _handle;

private:

  size_t _rows;
  size_t _cols;
  T* _data;
};

template <typename T>
void swap(device_matrix<T>& lhs, device_matrix<T>& rhs) {
  using std::swap;
  swap(lhs._rows, rhs._rows);
  swap(lhs._cols, rhs._cols);
  swap(lhs._data, rhs._data);
}

// In a class template, when performing implicit instantiation, the 
// members are instantiated on demand. Since the code does not use the
// static member, it's not even instantiated in the whole application.
template <typename T>
CUBLAS_HANDLE device_matrix<T>::_handle;

typedef device_matrix<float> dmat;
void sgemm(const dmat& A, const dmat& B, dmat& C, float alpha = 1.0, float beta = 0.0);
void sgeam(const dmat& A, const dmat& B, dmat& C, float alpha = 1.0, float beta = 1.0);
float snrm2(const dmat& A);


template <typename T>
device_matrix<T> operator * (T alpha, const device_matrix<T>& m) {
  return m * alpha;
}

template <typename T>
T L1_NORM(const device_matrix<T>& A, const device_matrix<T>& B) {
  return matsum(A - B);
}

#endif // __DEVICE_MATRIX_H__
