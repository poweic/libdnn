#ifndef __DNN_H_
#define __DNN_H_

#include <dnn-utility.h>
#include <feature-transform.h>

class DNN {
public:
  DNN();
  DNN(string fn);
  DNN(const std::vector<size_t>& dims);
  DNN(const DNN& source);
  DNN& operator = (DNN rhs);
  ~DNN();

  void feedForward(const DataSet& data, std::vector<mat>& O, size_t offset = 0, size_t batchSize = 0);
  void backPropagate(const DataSet& data, std::vector<mat>& O, mat& error, size_t offset = 0, size_t batchSize = 0);

  void updateParameters(float learning_rate = 1e-3);
  mat getError(const mat& target, const mat& output, size_t offset, size_t batchSize, ERROR_MEASURE errorMeasure);

  size_t getNLayer() const;
  size_t getDepth() const;
  void getEmptyGradient(std::vector<mat>& g) const;

  void _read(FILE* fid);
  void read(string fn);
  void save(string fn) const;
  void print() const;

  void train(const DataSet& train, const DataSet& valid, size_t batchSize, ERROR_MEASURE err);
  mat predict(const DataSet& test);

  friend void swap(DNN& lhs, DNN& rhs);

private:
  std::vector<AffineTransform*> _transforms;
  std::vector<size_t> _dims;
};

void swap(DNN& lhs, DNN& rhs);

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

mat l2error(mat& targets, mat& predicts);

void print(const thrust::host_vector<float>& hv);
void print(const mat& m);
void print(const thrust::device_vector<float>& dv);

#endif  // __DNN_H_
