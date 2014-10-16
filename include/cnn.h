#include <utility.h>
#include <cnn-utility.h>

/* ! Multiple Input Multiple Output (MIMO)
 *   Feature transformation
 * */
class MIMOFeatureTransform {
public:

  MIMOFeatureTransform(size_t n_input_maps, size_t n_output_maps);

  virtual void feedForward(vector<mat>& fouts, const vector<mat>& fins) = 0;
  virtual void feedBackward(vector<mat>& errors, const vector<mat>& deltas) = 0;

  virtual void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate) = 0;

  friend ostream& operator << (ostream& os, const MIMOFeatureTransform *ft);

  void set_input_img_size(const SIZE& s);
  SIZE get_input_img_size() const;
  virtual SIZE get_output_img_size() const = 0;

  size_t getNumInputMaps() const;
  size_t getNumOutputMaps() const;

  virtual void status() const = 0;

  void write(FILE* fid) const;
  void read(FILE* fid);
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

  void feedForward(mat& fout, const mat& fin);
  void backPropagate(mat& error, const mat& fin, const mat& fout,
      float learning_rate);

  void feedBackward(mat& error, const mat& delta);

  void init(const string &structure, SIZE img_size);
  void read(const string &fn);
  void save(const string &fn) const;

  size_t getInputDimension() const;
  size_t getOutputDimension() const;

  void status() const;

private:

  std::vector<MIMOFeatureTransform*> _transforms;
  std::vector<vector<mat> > _houts;
};

class ConvolutionalLayer : public MIMOFeatureTransform {

public:

  /* ! \brief A constructor
   * \param n number of input feature maps
   * \param m number of output feature maps
   * \param h height of convolutional kernels
   * \param w width of convolutional kernels. w = h if not w is not provided.
   * Creates and initialize n x m kernels, each of the size h x w.
   * */
  ConvolutionalLayer(size_t n, size_t m, int h, int w = -1);

  void feedForward(vector<mat>& fouts, const vector<mat>& fins);
  void feedBackward(vector<mat>& errors, const vector<mat>& deltas);

  void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate);

  void status() const;

  size_t getKernelWidth() const;
  size_t getKernelHeight() const;

  virtual SIZE get_output_img_size() const;

private:
  vector<vector<mat> > _kernels;
  vector<float> _bias;
};

class SubSamplingLayer : public MIMOFeatureTransform {
public:
  SubSamplingLayer(size_t n, size_t m, size_t scale);

  void status() const;

  size_t getScale() const;

  virtual SIZE get_output_img_size() const;

  void feedForward(vector<mat>& fouts, const vector<mat>& fins);
  void feedBackward(vector<mat>& errors, const vector<mat>& deltas);

  void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate);

private:
  size_t _scale;
};
