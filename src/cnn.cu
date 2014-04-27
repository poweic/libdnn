#include <cnn.h>
#define matslog(x) { for (int i=0; i<x.size(); ++i) { printf(#x"[%d] = [\n", i); x[i].print(); printf("]\n"); } }

/*! 
 * Implementation of CNN goes here.
 */

CNN::CNN(): _transforms() {

}

CNN::CNN(const string& model_fn) : _transforms() {
  this->read(model_fn);
}

CNN::~CNN() {
  for (size_t i=0; i<_transforms.size(); ++i)
    delete _transforms[i];
}

void CNN::feedForward(mat& fout, const mat& fin) {

  // First 1st layer of CNN MUST have only 1 input feature map
  vector<mat> fins(1);

  // Transpose the input feature (fin) so that rows = feature dimension, cols =
  // the number of data in a single batch.
  // FIXME the last column in fin is the bias needed ONLY by DNN, not by CNN.
  fins[0].resize(fin.getRows(), fin.getCols() - 1);
  memcpy2D(fins[0], fin, 0, 0, fin.getRows(), fin.getCols() - 1, 0, 0);
  fins[0] = ~fins[0];

  // FIXME SubSamplingLayer does NOT need temporary buffer.
  // MAYBE just reserve those for ConvolutionalLayer.
  _houts.resize(_transforms.size());

  _transforms[0]->feedForward(_houts[0], fins);

  for (size_t i=1; i<_transforms.size(); ++i)
    _transforms[i]->feedForward(_houts[i], _houts[i-1]);

  // Concatenate
  fout = ~concat(_houts.back());

  // Reserve one more column for bias
  fout.reserve(fout.size() + fout.getRows());
  fout.resize(fout.getRows(), fout.getCols() + 1);
  fillLastColumnWith(fout, 1.0f);
}

void CNN::backPropagate(mat& error, const mat& fin, const mat& fout,
    float learning_rate) {

  // Remove last column, which is bias in DNN.
  mat _fout(fout), _error(error);
  _fout.resize(_fout.getRows(), _fout.getCols() - 1);
  _error.resize(_error.getRows(), _error.getCols() - 1);

  int N = _transforms.back()->getNumOutputMaps();
  vector<mat> fouts = de_concat(~_fout, N),
	      errors = de_concat(~_error, N);

  vector<mat> fins;
  fins.push_back(~fin);

  _transforms.back()->backPropagate(errors, _houts.back(), fouts, learning_rate);

  for (int i=_transforms.size() - 2; i >= 1; --i)
    _transforms[i]->backPropagate(errors, _houts[i-1], _houts[i], learning_rate);

  _transforms[0]->backPropagate(errors, fins, _houts[0], learning_rate);

  // Concatenate
  error = ~concat(errors);
}

void CNN::feedBackward(mat& error, const mat& delta) {
  // TODO
}

void CNN::init(const string &structure, SIZE img_size) {

  vector<string> layers = split(structure, '-');

  size_t nInputMaps = 1;

  for (size_t i=0; i<layers.size(); ++i) {

    if (layers[i].find("s") != string::npos) {
      size_t scale = str2int(layers[i].substr(0, layers[i].size() - 1));

      size_t nOutputMaps = nInputMaps;
      MIMOFeatureTransform* t = new SubSamplingLayer( nInputMaps, nOutputMaps, scale);
      t->set_input_img_size(img_size);
      _transforms.push_back(t);

      // Set the input img_size of next layer to be the output img_size of current layer.
      img_size = t->get_output_img_size();
    }
    else if (layers[i].find("x") != string::npos) {

      vector<string> dims = split(layers[i], 'x');

      size_t nOutputMaps   = str2int(dims[0]),
	     kernel_width  = str2int(dims[1]),
	     kernel_height = str2int(dims[2]);

      MIMOFeatureTransform* t =
	new ConvolutionalLayer( nInputMaps, nOutputMaps, kernel_height, kernel_width);

      t->set_input_img_size(img_size);

      _transforms.push_back(t);

      // Set the input img_size of next layer to be the output img_size of current layer.
      img_size = t->get_output_img_size();
      nInputMaps = nOutputMaps;
    }
    else
      throw std::runtime_error("\33[31m[Error]\33[0m No such type of layer. \""
	  + layers[i] + "\". Only convolutional/sub-sampling layer are allowed");

  }
}

void CNN::read(const string &fn) {
  // TODO

}

void CNN::save(const string &fn) const {
  // TODO
}

size_t CNN::getInputDimension() const { 
  if (_transforms.size() == 0)
    throw std::runtime_error(RED_ERROR + "CNN not initialized. Don't know input dimension yet.");

  SIZE s = _transforms[0]->get_input_img_size();
  int nInputs = _transforms[0]->getNumInputMaps();
  return nInputs * s.m * s.n;
}

size_t CNN::getOutputDimension() const { 

  if (_transforms.size() == 0)
    throw std::runtime_error(RED_ERROR + "CNN not initialized. Don't know output dimension yet.");

  SIZE s = _transforms.back()->get_output_img_size();
  int nOutputs = _transforms.back()->getNumOutputMaps();
  return nOutputs * s.m * s.n;
}

void CNN::status() const {

  printf("+--------------+---------------+--------------+---------------+\n");
  printf("| # input maps | # output maps | kernel width | kernel height |\n");

  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i]->status();

  printf("+-------------------------------------------------------------+\n");
}

/*! 
 * Implementation of ConvolutionalLayer goes here.
 */
ConvolutionalLayer::ConvolutionalLayer(size_t nInputs, size_t nOutputs, int h, int w)
  : MIMOFeatureTransform(nInputs, nOutputs) {
  if (w == -1)
    w = h;

  assert(nInputs > 0 && nOutputs > 0 && h > 0 && w > 0);

  static int counter = 0;

  printf("Initializing %lu x %lu kernels of size %d x %d\n", nInputs, nOutputs, h, w);

  size_t fan_in = nInputs * w * h,
	 fan_out = nOutputs * w * h;
  float coeff = 2 * sqrt(6.0f / (fan_in + fan_out));

  _kernels.resize(nInputs);
  for (size_t i=0; i<nInputs; ++i) {
    _kernels[i].resize(nOutputs);
    for (size_t j=0; j<nOutputs; ++j)
      _kernels[i][j] = (rand(h, w) - 0.5f) * coeff;
  }

  _bias.resize(nOutputs);
  for (size_t j=0; j<nOutputs; ++j)
    _bias[j] = 0;
}

/* FIXME If every element in fins is a single feature map, then only a data can
 *      be fed forward through this function.
 *      NOTE that fins.size()  == # of input feature maps
 *                             != # of data in a batch
 *
 *	To feed forward a whole batch in a single function:
 *                fins.size()  == # of input feature maps
 *		  fins[i].rows == map.rows x map.cols
 *		  fins[i].cols == # of data
 *
 *	That is fins.size() is still the # of input feature maps (, which is
 *      always inevitable). However, in the i-th element of fins (i.e. fins[i])
 *	, there're multiple input feature maps comes from multiple training data.
 * */

void ConvolutionalLayer::feedForward(vector<mat>& fouts, const vector<mat>& fins) {

  size_t nInputs  = getNumInputMaps(),
	 nOutputs = getNumOutputMaps();

  if (fins.size() != nInputs)
    throw std::runtime_error(RED_ERROR + "Number of inputs maps ( = "
	+ to_string(fins.size()) + ") does not match number of kernels ( = "
	+ to_string(nInputs) + ").");

  size_t batch_size = fins[0].getCols();

  SIZE s = get_output_img_size();

  if (fouts.size() != nOutputs)
    fouts.resize(nOutputs);

  for (size_t j=0; j<nOutputs; ++j)
    fouts[j].resize(s.m * s.n, batch_size, 0);

  for (size_t j=0; j<nOutputs; ++j) {
    for (size_t i=0; i<nInputs; ++i)
      fouts[j] += convn(fins[i], _kernels[i][j], _input_img_size, "valid_shm");
    fouts[j] = sigmoid(fouts[j] + _bias[j]);
  }
}

void ConvolutionalLayer::feedBackward(
    vector<mat>& errors, const vector<mat>& deltas) {

  // Since nInputs == nOutputs for subsampling layer, I just use N.
  size_t nInputs = getNumInputMaps(),
	 nOutputs = getNumOutputMaps();

  SIZE s = this->get_input_img_size();
  size_t batch_size = deltas[0].getCols();

  vector<vector<mat> > oImgs(nOutputs), iImgs(nInputs);
  for (size_t j=0; j<nOutputs; ++j)
    oImgs[j] = reshapeVectors2Images(deltas[j], this->get_output_img_size());

  for (size_t i=0; i<nInputs; ++i)
    iImgs[i].resize(batch_size);

  // FIXME beware that upsample may NOT be able to get back to original size
  for (size_t k=0; k<batch_size; ++k) {
    for (size_t i=0; i<nInputs; ++i) {
      iImgs[i][k].resize(s.m, s.n, 0);
      for (size_t j=0; j<nOutputs; ++j)
	iImgs[i][k] += convn(oImgs[j][k], rot180(_kernels[i][j]), "full");
    }
  }

  if (errors.size() != nInputs)
    errors.resize(nInputs);

  for (size_t i=0; i<nInputs; ++i)
    errors[i] = reshapeImages2Vectors(iImgs[i]);

}

// NOTE: in MATLAB
// xcorr2 stands for 2D cross-correlation
// (I don't know why MATLAB does not have "xcorrn" for n-dimensional xcorr)
// The following operation are theoretically equivalent:
// (with only some trivial numerical error)
// (1)  convn(x, rot180(h)) == xcorr2(x, h)
//     xcorr2(x, rot180(h)) ==  convn(x, h)
// (2) convn(rot180(x), h) == rot180(convn(x, rot180(h)))
//     ^
//     |_____ which is obviously faster

void ConvolutionalLayer::backPropagate(vector<mat>& errors, const vector<mat>& fins,
    const vector<mat>& fouts, float learning_rate) {

  size_t nInputs = getNumInputMaps(),
	 nOutputs = getNumOutputMaps();

  size_t batch_size = fins[0].getCols();

  // In the following codes, the iteration index i and j stands for
  // i : # of input  features. i = 0 ~ nInputs - 1 
  // j : # of output features. j = 0 ~ nOutputs - 1

  vector<mat> deltas(nOutputs);
  for (size_t j=0; j<nOutputs; ++j)
    deltas[j] = fouts[j] & ( 1.0f - fouts[j] ) & errors[j];

  this->feedBackward(errors, deltas);

  // iImgs represents the input images.
  // oImgs represents the output images. (Before sigmoid or any other activation function)
  vector<vector<mat> > iImgs(nInputs), oImgs(nOutputs);

  for (size_t i=0; i<nInputs; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i], _input_img_size);

  for (size_t j=0; j<nOutputs; ++j)
    oImgs[j] = reshapeVectors2Images(deltas[j], this->get_output_img_size());

  assert(learning_rate > 0);
  float lr = learning_rate / batch_size;

  // Update kernels with learning rate
  for (size_t k=0; k<batch_size; ++k) {
    for (size_t j=0; j<nOutputs; ++j) {

      for (size_t i=0; i<nInputs; ++i)
	_kernels[i][j] -= convn(rot180(iImgs[i][k]), oImgs[j][k], "valid") * lr;

      _bias[j] -= sum_all(oImgs[j][k]) * lr;
    }
  }
}

void ConvolutionalLayer::status() const {

  printf("+--------------+---------------+--------------+---------------+\n");
  printf("|      %-5lu   |       %-5lu   |      %-5lu   |       %-5lu   |\n",
      getNumInputMaps(), getNumOutputMaps(), getKernelWidth(), getKernelHeight());
}

size_t ConvolutionalLayer::getKernelWidth() const {
  return _kernels[0][0].getCols();
}

size_t ConvolutionalLayer::getKernelHeight() const {
  return _kernels[0][0].getRows();
}

/*size_t ConvolutionalLayer::getNumInputMaps() const {
  return _kernels.size();
}

size_t ConvolutionalLayer::getNumOutputMaps() const {
  return _kernels[0].size();
}*/

SubSamplingLayer::SubSamplingLayer(size_t m, size_t n, size_t scale)
  : MIMOFeatureTransform(m, n), _scale(scale) {
}

void SubSamplingLayer::status() const {
  printf("+-------------------------------------------------------------+\n");
  printf("|                Sub-Sampling Factor = %-4lu                   |\n", _scale);
}
  
size_t SubSamplingLayer::getScale() const {
  return _scale;
}

void SubSamplingLayer::feedForward(vector<mat>& fouts, const vector<mat>& fins) {

  // Since nInputs == nOutputs for subsampling layer, I just use N.
  size_t N = fins.size();

  if (fouts.size() != N)
    fouts.resize(N);

  for (size_t i=0; i<N; ++i)
    fouts[i] = downsample(fins[i], _scale, _input_img_size);
}

void SubSamplingLayer::feedBackward(
    vector<mat>& errors, const vector<mat>& deltas) {

  // Since nInputs == nOutputs for subsampling layer, I just use N.
  size_t N = deltas.size();
  size_t batch_size = deltas[0].getCols();

  vector<vector<mat> > oImgs(N), iImgs(N);
  for (size_t i=0; i<N; ++i)
    oImgs[i] = reshapeVectors2Images(deltas[i], this->get_output_img_size());

  // FIXME beware that upsample may NOT be able to get back to original size
  for (size_t i=0; i<N; ++i) {
    iImgs[i].resize(batch_size);
    for (size_t k=0; k<batch_size; ++k)
      iImgs[i][k] = upsample(oImgs[i][k], _input_img_size) / (_scale * _scale);
  }

  if (errors.size() != N)
    errors.resize(N);

  for (size_t j=0; j<N; ++j)
    errors[j] = reshapeImages2Vectors(iImgs[j]);
}

void SubSamplingLayer::backPropagate(vector<mat>& errors, const vector<mat>& fins,
    const vector<mat>& fouts, float learning_rate) {

  // Copy errors element by element to deltas
  vector<mat> deltas(errors);

  this->feedBackward(errors, deltas);
}
