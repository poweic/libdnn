#ifndef _DNN_UTILITY_H_
#define _DNN_UTILITY_H_

#include <limits>
#include <cstdio>

#include <arithmetic.h>
#include <math_ext.h>
#include <perf.h>

#ifndef PAUSE
#define PAUSE { printf("Press Enter key to continue..."); fgetc(stdin); }
#endif

#ifndef matlog
#define matlog(x) { cout << "\33[34m" << #x << "\33[0m = [" << endl; x.print(); cout << "];" << endl; }
#endif
#define dsigma(x) ((x) & ((float) 1.0 - (x)))
#define mylog(x) { cout << #x << " = " << x << endl; }

#define float_min std::numeric_limits<float>::min()
#define float_max std::numeric_limits<float>::max()

#include <device_matrix.h>

#ifdef __CUDACC__
  #include <device_math.h>
  #include <device_arithmetic.h>
  #define WHERE thrust
#endif

#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/inner_product.h>

typedef device_matrix<float> mat;

enum ERROR_MEASURE {
  L2ERROR,  /* for binary-classification only */
  CROSS_ENTROPY
};

void playground();


map<int, int> getLabelMapping(const mat& labels);

mat label2PosteriorProb(const mat& labels);
mat posteriorProb2Label(const mat& prob);

size_t zeroOneError(const mat& predict, const mat& label, ERROR_MEASURE errorMeasure);
mat& calcError(const mat& output, const mat& trainY, size_t offset = 0, size_t nData = 0);

vector<float> copyToHost(const mat& m);
size_t countDifference(const mat& m1, const mat& m2);

void print(const std::vector<mat>& vm);

void showAccuracy(size_t nError, size_t nTotal);

void getDataAndLabels(string train_fn, mat& data, mat& labels);

bool isFileSparse(string train_fn);

string getTempFilename();
void exec(string command);
float str2float(const string &s);
vector<string> split(const string &s, char delim);
vector<string>& split(const string &s, char delim, vector<string>& elems);
vector<size_t> splitAsInt(const string &s, char delim);

std::vector<size_t> randshuf(size_t N);

bool isLabeled(const mat& labels);

mat rowSum(mat& m);

namespace ext {
  void rescale(mat& data, float lower, float upper);

  float max(const mat& v);
  float min(const mat& v);

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

  template <typename T>
  device_matrix<T> softmax(const device_matrix<T>& x) {
    // TODO
    // Do the softmax
    device_matrix<T> s(x.getRows(), x.getCols());

    thrust::device_ptr<T> xPtr(x.getData());
    thrust::device_ptr<T> sPtr(s.getData());

    thrust::transform(xPtr, xPtr + x.size(), sPtr, func::sigmoid<float>());

    return s;
  }
};

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


#endif // _DNN_UTILITY_H_
