#ifndef _FEATURE_TRANSFORM_H_
#define _FEATURE_TRANSFORM_H_

#include <dnn-utility.h>

string toString(std::vector<float> data, size_t rows, size_t cols);

class FeatureTransform {
public:
  FeatureTransform(const FeatureTransform& source);
  FeatureTransform(size_t rows, size_t cols, float variance);

  virtual FeatureTransform* clone() const = 0;
  virtual string toString() const = 0;
  virtual void feedForward(mat& fout, const mat& fin) = 0;
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) = 0;

  virtual void feedBackward(mat& error, const mat& delta);

  size_t getInputDimension() const;
  size_t getOutputDimension() const;
  void print() const;

protected:
  FeatureTransform(const mat& w);

  mat _w;

private:
  virtual FeatureTransform& operator = (const FeatureTransform& rhs) {}
};

class Sigmoid : public FeatureTransform {
public:
  Sigmoid(const mat& w);
  Sigmoid(const Sigmoid& src);
  Sigmoid(size_t rows, size_t cols, float variance);

  virtual Sigmoid* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

private:
  virtual Sigmoid& operator = (Sigmoid rhs) {}
};

class Softmax : public FeatureTransform {
public:
  Softmax(const mat& w);
  Softmax(size_t rows, size_t cols, float variance);
  Softmax(const Softmax& src);

  virtual Softmax* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

private:
  virtual Softmax& operator = (Softmax rhs) {}
};

#endif // _FEATURE_TRANSFORM_H_
