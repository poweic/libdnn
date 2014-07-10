#ifndef _FEATURE_TRANSFORM_H_
#define _FEATURE_TRANSFORM_H_

#include <dnn-utility.h>

// string toString(std::vector<float> data, size_t rows, size_t cols);

class FeatureTransform {
public:
  FeatureTransform() { }
  FeatureTransform(size_t input_dim, size_t output_dim);

  virtual FeatureTransform* clone() const = 0;
  virtual string toString() const = 0;
  virtual void feedForward(mat& fout, const mat& fin) = 0;
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) = 0;

  virtual size_t getInputDimension() const { return _input_dim; }
  virtual size_t getOutputDimension() const { return _output_dim; }

  virtual void read(FILE* fid) = 0;
  virtual void write(FILE* fid) const = 0;

  static FeatureTransform* create(FILE* fid);

protected:
  size_t _input_dim;
  size_t _output_dim;

private:
  virtual FeatureTransform& operator = (const FeatureTransform& rhs) { return *this; };
};

class AffineTransform : public FeatureTransform {
public:
  AffineTransform(size_t input_dim, size_t output_dim);
  AffineTransform(const mat& w);
  AffineTransform(FILE* fid);

  virtual void read(FILE* fid);
  virtual void write(FILE* fid) const;

  virtual AffineTransform* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

private:
  /* \brief _w is the "augmented" matrix (include bias term)
   *
   */
  mat _w;
};

class Activation : public FeatureTransform {
public:
  Activation();
  Activation(size_t input_dim, size_t output_dim);

  virtual void read(FILE* fid);
  virtual void write(FILE* fid) const;
};

class Sigmoid : public Activation {
public:
  Sigmoid(size_t input_dim, size_t output_dim);
  Sigmoid(FILE* fid);

  virtual Sigmoid* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

private:
  virtual Sigmoid& operator = (const Sigmoid& rhs) { return *this; }
};

class Softmax : public Activation {
public:
  Softmax(size_t input_dim, size_t output_dim);
  Softmax(FILE* fid);

  virtual Softmax* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

private:
  virtual Softmax& operator = (const Softmax& rhs) { return *this; }
};

#endif // _FEATURE_TRANSFORM_H_
