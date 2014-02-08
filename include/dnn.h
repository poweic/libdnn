#ifndef __DNN_H_
#define __DNN_H_

#include <dnn-utility.h>
#include <dataset.h>
#include <feature-transform.h>
#include <config.h>
#include <utility.h>

// sigmoid mapping
//    x     sigmoid(x) percentage
// -4.5951    0.01	   1%
// -3.8918    0.02         2%
// -2.9444    0.05         5%
// -2.1972    0.10        10%
// -1.3863    0.20        20%
//    0       0.50        50%
//  4.5951    0.80        20%
//  3.8918    0.90        10%
//  2.9444    0.95         5%
//  2.1972    0.98         2%
//  1.3863    0.99         1%

class DNN {
public:
  DNN();
  DNN(string fn);
  DNN(const Config& config);
  DNN(const DNN& source);
  ~DNN();

  DNN& operator = (DNN rhs);

  void init(const std::vector<size_t>& dims);
  void init(const std::vector<mat>& weights);

  void feedForward(mat& output, const mat& fin);
  void backPropagate(mat& error, const mat& fin, const mat& fout);

  void update(float learning_rate);
  mat getError(const mat& target, const mat& output, size_t offset, size_t batchSize, ERROR_MEASURE errorMeasure);

  void setConfig(const Config& config);
  size_t getNLayer() const;
  void getEmptyGradient(std::vector<mat>& g) const;

  Config getConfig() const;
  void adjustLearningRate(float trainAcc);

  void _read(FILE* fid);
  void read(string fn);
  void save(string fn) const;
  void print() const;
  
  mat predict(const DataSet& test);

  friend void swap(DNN& lhs, DNN& rhs);

private:
  std::vector<FeatureTransform*> _transforms;

  /* Hidden Outputs: outputs of each hidden layers
   * The first element in the std::vector (i.e. _houts[0])
   * is the output of first hidden layer. 
   * ( Note: this means no input data will be kept in _houts. )
   * ( Also, no output data will be kept in _houts. )
   * */
  std::vector<mat> _houts;
  Config _config;
};

void swap(DNN& lhs, DNN& rhs);

template <typename T>
void remove_bias(vector<T>& v) {
  v.pop_back();
}

void print(const thrust::host_vector<float>& hv);
void print(const thrust::device_vector<float>& dv);

#endif  // __DNN_H_
