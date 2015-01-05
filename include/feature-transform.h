#ifndef _FEATURE_TRANSFORM_H_
#define _FEATURE_TRANSFORM_H_

#include <dnn-utility.h>
#include <cnn-utility.h>
#include <tools/rapidxml-1.13/rapidxml_utils.hpp>
using namespace rapidxml;

class FeatureTransform {
public:
  FeatureTransform() { }
  FeatureTransform(size_t input_dim, size_t output_dim);
  virtual ~FeatureTransform() {}

  virtual void read(xml_node<> *node);
  virtual void write(std::ostream& os) const = 0;

  virtual FeatureTransform* clone() const = 0;
  virtual std::string toString() const = 0;

  virtual void feedForward(mat& fout, const mat& fin) = 0;
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) = 0;

  virtual size_t getInputDimension() const { return _input_dim; }
  virtual size_t getOutputDimension() const { return _output_dim; }

  virtual size_t getNumParams() const { return 0; }

  friend std::ostream& operator << (std::ostream& os, FeatureTransform* ft);

  enum Type {
    Affine,
    Sigmoid,
    Tanh,
    ReLU,
    Softplus,
    Softmax,
    Dropout,
    Convolution,
    SubSample
  };

  static Type token2type(std::string token);
  static std::map<Type, std::string> type2token;

protected:
  size_t _input_dim;
  size_t _output_dim;
};

bool isXmlFormat(std::istream& is);
float GetNormalizedInitCoeff(size_t fan_in, size_t fan_out,
    FeatureTransform::Type type);

std::ostream& operator << (std::ostream& os, FeatureTransform* ft);
std::istream& operator >> (std::istream& is, FeatureTransform* &ft);

class AffineTransform : public FeatureTransform {
public:
  AffineTransform() {}
  AffineTransform(size_t input_dim, size_t output_dim);
  AffineTransform(const mat& w);

  virtual void read(xml_node<> *node);
  virtual void write(std::ostream& os) const;

  virtual AffineTransform* clone() const;
  virtual std::string toString() const;

  virtual void feedForward(mat& fout, const mat& fin);
  virtual void feedBackward(mat& error, const mat& delta);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);

  virtual size_t getNumParams() const;

  void set_w(const mat& w);
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
  virtual void write(std::ostream& os) const;

  void dropout(mat& fout);
};

class Sigmoid : public Activation {
public:
  Sigmoid() {}
  Sigmoid(size_t input_dim, size_t output_dim);

  virtual Sigmoid* clone() const;
  virtual std::string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);
};

class Tanh : public Activation {
public:
  Tanh() {}
  Tanh(size_t input_dim, size_t output_dim);

  virtual Tanh* clone() const;
  virtual std::string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);
};

class ReLU : public Activation {
public:
  ReLU() {}
  ReLU(size_t input_dim, size_t output_dim);

  virtual ReLU* clone() const;
  virtual std::string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);
};

class Softplus : public Activation {
public:
  Softplus() {}
  Softplus(size_t input_dim, size_t output_dim);

  virtual Softplus* clone() const;
  virtual std::string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);
};

class Softmax : public Activation {
public:
  Softmax() {}
  Softmax(size_t input_dim, size_t output_dim);

  virtual Softmax* clone() const;
  virtual std::string toString() const;
  virtual void feedForward(mat& fout, const mat& fin);
  virtual void backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate);
};

class Dropout : public Activation {
public:
  Dropout();
  Dropout(size_t input_dim, size_t output_dim);

  virtual void read(xml_node<> *node);
  virtual void write(std::ostream& os) const;

  virtual Dropout* clone() const;
  virtual std::string toString() const;
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

/* ! Multiple Input Multiple Output (MIMO)
 *   Feature transformation
 * */
class MIMOFeatureTransform : public FeatureTransform {
public:

  MIMOFeatureTransform() {}
  MIMOFeatureTransform(size_t n_input_maps, size_t n_output_maps);
  virtual ~MIMOFeatureTransform() {}

  virtual void read(xml_node<> *node);
  virtual void write(std::ostream& os) const;

  virtual MIMOFeatureTransform* clone() const = 0;
  virtual std::string toString() const = 0;

  virtual void feedForward(mat& fouts, const mat& fins) = 0;
  virtual void feedBackward(mat& errors, const mat& deltas) = 0;
  virtual void backPropagate(mat& errors, const mat& fins, const mat& fouts, float learning_rate) = 0;

  virtual size_t getInputDimension() const = 0;
  virtual size_t getOutputDimension() const = 0;

  friend std::ostream& operator << (std::ostream& os, const MIMOFeatureTransform *ft);

  void set_input_img_size(const SIZE& s);
  virtual SIZE get_input_img_size() const;
  virtual SIZE get_output_img_size() const = 0;

  size_t getNumInputMaps() const;
  size_t getNumOutputMaps() const;

protected:
  SIZE _input_img_size;
  size_t _n_input_maps;
  size_t _n_output_maps;
};

std::ostream& operator << (std::ostream& os, const MIMOFeatureTransform *ft);

class ConvolutionalLayer : public MIMOFeatureTransform {

public:

  ConvolutionalLayer() {}

  /* ! \brief A constructor
   * \param n number of input feature maps
   * \param m number of output feature maps
   * \param h height of convolutional kernels
   * \param w width of convolutional kernels. w = h if not w is not provided.
   * Creates and initialize n x m kernels, each of the size h x w.
   * */
  ConvolutionalLayer(size_t n, size_t m, int h, int w = -1);

  virtual ConvolutionalLayer* clone() const;
  virtual std::string toString() const;

  virtual void read(xml_node<> *node);
  virtual void write(std::ostream& os) const;

  virtual void feedForward(mat& fouts, const mat& fins);
  virtual void feedBackward(mat& errors, const mat& deltas);
  virtual void backPropagate(mat& errors, const mat& fins, const mat& fouts, float learning_rate);

  void update_kernel(const mat& fin, const mat& delta);
  void update_bias(const mat& delta);

  virtual size_t getInputDimension() const;
  virtual size_t getOutputDimension() const;

  virtual SIZE get_output_img_size() const;

  virtual size_t getNumParams() const;

  SIZE get_kernel_size() const;
  size_t getKernelWidth() const;
  size_t getKernelHeight() const;

private:
  std::vector<std::vector<mat> > _kernels;
  std::vector<float> _bias;
};

class SubSamplingLayer : public MIMOFeatureTransform {
public:

  SubSamplingLayer() {}

  SubSamplingLayer(size_t n, size_t m, size_t scale);

  virtual SubSamplingLayer* clone() const;
  virtual std::string toString() const;

  virtual void read(xml_node<> *node);
  virtual void write(std::ostream& os) const;

  virtual void feedForward(mat& fouts, const mat& fins);
  virtual void feedBackward(mat& errors, const mat& deltas);
  virtual void backPropagate(mat& errors, const mat& fins, const mat& fouts, float learning_rate);

  virtual size_t getInputDimension() const;
  virtual size_t getOutputDimension() const;

  virtual SIZE get_output_img_size() const;

  size_t getScale() const;

private:
  size_t  _scale;
};

#endif // _FEATURE_TRANSFORM_H_
