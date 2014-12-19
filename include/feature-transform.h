#ifndef _FEATURE_TRANSFORM_H_
#define _FEATURE_TRANSFORM_H_

#include <dnn-utility.h>
#include <tools/rapidxml-1.13/rapidxml_utils.hpp>
using namespace rapidxml;

class FeatureTransform {
public:
  FeatureTransform() { }
  FeatureTransform(size_t input_dim, size_t output_dim);

  virtual void read(xml_node<> *node);
  virtual void read(istream& is) = 0;
  virtual void write(ostream& os) const = 0;

  virtual FeatureTransform* clone() const = 0;
  virtual string toString() const = 0;

  virtual void feedForward(mat& fout, const mat& fin) = 0;
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) = 0;

  virtual size_t getInputDimension() const { return _input_dim; }
  virtual size_t getOutputDimension() const { return _output_dim; }

  virtual void status() const = 0;

  friend ostream& operator << (ostream& os, FeatureTransform* ft);
  friend istream& operator >> (istream& is, FeatureTransform* &ft);

  enum Type {
    Affine,
    Sigmoid,
    Softmax,
    Dropout,
    Convolution,
    SubSample
  };

  static Type token2type(string token);
  static std::map<Type, string> type2token;

protected:
  size_t _input_dim;
  size_t _output_dim;
};

bool isXmlFormat(istream& is);

ostream& operator << (ostream& os, FeatureTransform* ft);
istream& operator >> (istream& is, FeatureTransform* &ft);

class AffineTransform : public FeatureTransform {
public:
  AffineTransform() {}
  AffineTransform(size_t input_dim, size_t output_dim);
  AffineTransform(const mat& w);
  AffineTransform(istream& is);

  virtual void read(xml_node<> *node);
  virtual void read(istream& is);
  virtual void write(ostream& os) const;

  virtual AffineTransform* clone() const;
  virtual string toString() const;

  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

  virtual void status() const;

  mat& get_w();
  mat const& get_w() const;

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

  virtual void read(xml_node<> *node);
  virtual void read(istream& is);
  virtual void write(ostream& os) const;

  virtual void status() const;

  void dropout(mat& fout);
};

class Sigmoid : public Activation {
public:
  Sigmoid() {}
  Sigmoid(size_t input_dim, size_t output_dim);
  Sigmoid(istream& is);

  virtual Sigmoid* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);
};

class Softmax : public Activation {
public:
  Softmax() {}
  Softmax(size_t input_dim, size_t output_dim);
  Softmax(istream& is);

  virtual Softmax* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);
};

class Dropout : public Activation {
public:
  Dropout();
  Dropout(size_t input_dim, size_t output_dim);
  Dropout(istream& is);

  virtual void read(xml_node<> *node);
  virtual void write(ostream& os) const;

  virtual Dropout* clone() const;
  virtual string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

  void setDropout(bool flag) { _dropout = flag; }

private:
  /* \brief _dropout_ratio means how many values will be turned off. (in %)
   */
  float _dropout_ratio;
  bool _dropout;
  mat _dropout_mask;
};

#endif // _FEATURE_TRANSFORM_H_
