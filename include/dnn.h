#ifndef __DNN_H_
#define __DNN_H_

#include <dnn-utility.h>
#include <dataset.h>
#include <feature-transform.h>
#include <config.h>
#include <utility.h>

class DNN {
public:
  DNN();
  DNN(string fn);
  DNN(const Config& config);
  DNN(const DNN& source);
  ~DNN();

  DNN& operator = (DNN rhs);

  void init(const std::vector<mat>& weights);

  mat feedForward(const mat& fin) const;
  void feedForward(mat& output, const mat& fin);
  void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

  void setConfig(const Config& config);
  size_t getNLayer() const;

  Config getConfig() const;
  void adjustLearningRate(float trainAcc);

  void status() const;

  void read(string fn);
  void save(string fn) const;

  std::vector<FeatureTransform*>& getTransforms();
  const std::vector<FeatureTransform*>& getTransforms() const;

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
#endif  // __DNN_H_
