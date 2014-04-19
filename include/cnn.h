#include <cnn-utility.h>

class ConvolutionalLayer {

public:

  /* ! \brief A constructor
   * \param n number of input feature maps
   * \param m number of output feature maps
   * \param h height of convolutional kernels
   * \param w width of convolutional kernels. w = h if not w is not provided.
   * Creates and initialize n x m kernels, each of the size h x w.
   * */
  ConvolutionalLayer(size_t n, size_t m, size_t h, size_t w = -1);

  void feedForward(vector<mat>& fouts, const vector<mat>& fins);

  void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate);

  void status() const;

  size_t getKernelWidth() const;

  size_t getKernelHeight() const;

  size_t getNumInputMaps() const;

  size_t getNumOutputMaps() const;

private:
  vector<vector<mat> > _kernels;
  vector<float> _bias;
};

class SubSamplingLayer {
public:
  SubSamplingLayer(size_t scale);

  void feedForward(vector<mat>& fouts, const vector<mat>& fins);

  void backPropagate(vector<mat>& errors, const vector<mat>& fins,
      const vector<mat>& fouts, float learning_rate);

private:
  size_t _scale;
};
