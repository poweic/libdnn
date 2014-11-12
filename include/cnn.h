#include <utility.h>
#include <cnn-utility.h>
#include <tools/rapidxml-1.13/rapidxml_utils.hpp>
using namespace rapidxml;

/* ! Multiple Input Multiple Output (MIMO)
 *   Feature transformation
 * */
class MIMOFeatureTransform {
public:

  MIMOFeatureTransform() {}
  MIMOFeatureTransform(size_t n_input_maps, size_t n_output_maps);

  virtual void read(xml_node<> *node);
  virtual void write(ostream& os) const;

  virtual MIMOFeatureTransform* clone() const = 0;
  virtual string toString() const = 0;

  virtual void feedForward(vector<mat>& fouts, const vector<mat>& fins) = 0;
  virtual void feedBackward(vector<mat>& errors, const vector<mat>& deltas) = 0;

  virtual void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate) = 0;

  virtual SIZE get_output_img_size() const = 0;

  virtual void status() const = 0;

  friend ostream& operator << (ostream& os, const MIMOFeatureTransform *ft);

  void set_input_img_size(const SIZE& s);
  SIZE get_input_img_size() const;

  size_t getNumInputMaps() const;
  size_t getNumOutputMaps() const;

protected:
  SIZE _input_img_size;
  size_t _n_input_maps;
  size_t _n_output_maps;
};

ostream& operator << (ostream& os, const MIMOFeatureTransform *ft);

class CNN {
public:

  CNN();
  CNN(const string& model_fn);
  ~CNN();

  void init(const string &structure, SIZE img_size);

  void feedForward(mat& fout, const mat& fin);
  void backPropagate(mat& error, const mat& fin, const mat& fout,
      float learning_rate);

  void feedBackward(mat& error, const mat& delta);

  void read(const string &fn);
  void save(const string &fn) const;

  size_t getInputDimension() const;
  size_t getOutputDimension() const;

  void status() const;

  friend ostream& operator << (ostream& os, const CNN& cnn);

private:

  std::vector<MIMOFeatureTransform*> _transforms;
  std::vector<vector<mat> > _houts;
};

ostream& operator << (ostream& os, const CNN& cnn);

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

  virtual void read(xml_node<> *node);
  virtual void write(ostream& os) const;

  virtual ConvolutionalLayer* clone() const;
  virtual string toString() const;

  virtual void feedForward(vector<mat>& fouts, const vector<mat>& fins);
  virtual void feedBackward(vector<mat>& errors, const vector<mat>& deltas);

  virtual void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate);

  virtual SIZE get_output_img_size() const;

  virtual void status() const;

  size_t getKernelWidth() const;
  size_t getKernelHeight() const;

private:
  vector<vector<mat> > _kernels;
  vector<float> _bias;
};

class SubSamplingLayer : public MIMOFeatureTransform {
public:

  SubSamplingLayer() {}

  SubSamplingLayer(size_t n, size_t m, size_t scale);

  virtual void read(xml_node<> *node);
  virtual void write(ostream& os) const;

  virtual SubSamplingLayer* clone() const;
  virtual string toString() const;

  virtual void feedForward(vector<mat>& fouts, const vector<mat>& fins);
  virtual void feedBackward(vector<mat>& errors, const vector<mat>& deltas);

  virtual void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate);

  virtual SIZE get_output_img_size() const;

  virtual void status() const;

  size_t getScale() const;

private:
  size_t _scale;
};
