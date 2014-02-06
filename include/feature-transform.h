#ifndef _FEATURE_TRANSFORM_H_
#define _FEATURE_TRANSFORM_H_

#include <dnn-utility.h>

void playground();

class FeatureTransform {
public:
  FeatureTransform(const FeatureTransform& source);
  FeatureTransform(size_t rows, size_t cols, float variance);

  virtual FeatureTransform* clone() const = 0;
  virtual string toString() const = 0;
  virtual void feedForward(mat& fout, const mat& fin, size_t offset, size_t nData) = 0;
  virtual void backPropagate(const mat& fin, const mat& fout, mat& error) = 0;

  mat& getW();
  mat& getDw();
  const mat& getW() const;
  const mat& getDw() const;

  void update(float learning_rate);

protected:
  FeatureTransform(const mat& w);

  mat _w;
  mat _dw;

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
  virtual void feedForward(mat& fout, const mat& fin, size_t offset, size_t nData);
  virtual void backPropagate(const mat& fin, const mat& fout, mat& error);

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
  virtual void feedForward(mat& fout, const mat& fin, size_t offset, size_t nData);
  virtual void backPropagate(const mat& fin, const mat& fout, mat& error);

private:
  virtual Softmax& operator = (Softmax rhs) {}
};

#endif // _FEATURE_TRANSFORM_H_
