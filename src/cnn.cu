#include <cnn.h>
#define RED_ERROR (string("\33[31m[Error]\33[0m In function \"") \
    + __func__ + string("\" (at ") + __FILE__ + string(":") \
    + to_string(__LINE__) + string("): "))

/*! 
 * Implementation of CNN goes here.
 */

CNN::CNN(): _transforms() {

}

CNN::CNN(const string& model_fn) : _transforms() {
  this->read(model_fn);
}

mat concat(const vector<mat>& feature_maps) {
  int nFeatures = feature_maps.size(),
      img_size  = feature_maps[0].getRows(),
      batchSize = feature_maps[0].getCols();

  mat result(img_size * nFeatures, batchSize);

  // TODO

  return result;
}

void CNN::feedForward(mat& fout, const mat& fin) {

  // First 1st layer of CNN MUST have only 1 input feature map
  vector<mat> fins;

  // Transpose the input feature (fin) so that rows = feature dimension, cols =
  // the number of data in a single batch.
  fins.push_back(~fin);

  // FIXME SubSamplingLayer does NOT need temporary buffer.
  // MAYBE just reserve those for ConvolutionalLayer.
  _houts.resize(_transforms.size());

  cout << "======================================" << endl;
  cout << "Layer 0: " << _transforms[0] << endl;
  _transforms[0]->feedForward(_houts[0], fins);

  for (size_t i=1; i<_transforms.size(); ++i) {
    cout << "Layer " << i << ": " << _transforms[i] << endl;
    _transforms[i]->feedForward(_houts[i], _houts[i-1]);
  }

  // Concatenate
  fout = ~concat(_houts.back());
}

void CNN::backPropagate(mat& error, const mat& fin, const mat& fout,
    float learning_rate) {

  vector<mat> fouts, errors, fins;
  fouts.push_back(~fout);
  errors.push_back(~error);
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

      MIMOFeatureTransform* t = new SubSamplingLayer(scale);
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
ConvolutionalLayer::ConvolutionalLayer(size_t n, size_t m, size_t h, size_t w) {
  if (w == -1)
    w = h;

  assert(n > 0 && m > 0 && h > 0 && w > 0);

  _bias.resize(n);

  printf("Initializing %lu x %lu kernels of size %lu x %lu\n", n, m, h, w);
  _kernels.resize(n);
  for (int i=0; i<n; ++i) {
    _kernels[i].assign(m, rand(h, w));
    _bias[i] = 0;
  }
}

/* TODO If every element in fins is a single feature map, then only a data can
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

  int nInputs  = getNumInputMaps(),
      nOutputs = getNumOutputMaps();

  if (fins.size() != nInputs)
    throw std::runtime_error(RED_ERROR + "Number of inputs maps ( = "
	+ to_string(fins.size()) + ") does not match number of kernels ( = "
	+ to_string(nInputs) + ").");

  int batch_size = fins[0].getCols();

  vector<vector<mat> > iImgs(nInputs), oImgs(nOutputs);

  SIZE s = get_output_img_size();

  for (int i=0; i<nInputs; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i], _input_img_size);

  // Allocate memory and initialize with value 0
  for (int j=0; j<nOutputs; ++j) {
    oImgs[j].resize(batch_size);

    for (int k=0; k<batch_size; ++k)
      oImgs[j][k].resize(s.m, s.n, 0);
  }

  for (int k=0; k<batch_size; ++k) {
    for (int j=0; j<nOutputs; ++j) {
      for (int i=0; i<nInputs; ++i)
	oImgs[j][k] += convn(iImgs[i][k], _kernels[i][j], "valid_shm");
      oImgs[j][k] = sigmoid(oImgs[j][k] + _bias[j]);
    }
  }

  if (fouts.size() != nOutputs)
    fouts.resize(nOutputs);

  for (int j=0; j<nOutputs; ++j)
    fouts[j] = reshapeImages2Vectors(oImgs[j]);

  /*for (int i=0; i<nInputs; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i]);

  SIZE s = get_convn_size(iImgs[0][0], _kernels[0][0], "valid");

  for (int j=0; j<nOutputs; ++j) {
    oImgs[j].resize(batch_size);
    for (int k=0; k<batch_size; ++k)
      oImgs[j][k].resize(s.m, s.n, 0);
  }

  for (int b=0; b<batch_size; ++b) {
    for (int j=0; j<nOutputs; j++) {
      for (int i=0; i<nInputs; ++i)
	oImgs[j][b] += convn(iImgs[i][b], _kernels[i][j], "valid");
      oImgs[j][b] = sigmoid(oImgs[j][b] + _bias[j]);
    }
  }

  if (fouts.size() != nOutputs)
    fouts.resize(nOutputs);

  for (int j=0; j<nOutputs; ++j)
    fouts[j] = reshapeImages2Vectors(all_fouts[j]);*/
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

  // FIXME How to backPropagate a BATCH of images AT A TIME.

  // In the following codes, the iteration index i and j stands for
  // i : # of input  features. i = 0 ~ M, where M = fin.size()
  // j : # of output features. j = 0 ~ N, where N = fouts.size() = errors.size()
  int M = fins.size(),
      N = fouts.size();

  // Compute delta from errors over the derivatives of sigmoid function.
  vector<mat> deltas(N);
  for (size_t j=0; j<N; ++j)
    deltas[j] = fouts[j] & ( 1.0f - fouts[j] ) & errors[j];

  // Update kernels with learning rate
  for (size_t j=0; j<N; ++j) {
    for (size_t i=0; i<M; ++i)
      _kernels[i][j] += convn(rot180(fins[i]), deltas[j], "valid") * learning_rate;

    _bias[j] += sum_all(deltas[j]) * learning_rate;
  }

  // Feed Backward
  SIZE s = get_convn_size(errors[0], rot180(_kernels[0][0]), "full");
  vector<mat> err(M);
  for (size_t i=0; i<M; ++i) {
    err[i].resize(s.m, s.n);

    for (size_t j=0; j<N; ++j)
      err[i] += convn(errors[j], rot180(_kernels[j][i]), "full");
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

size_t ConvolutionalLayer::getNumInputMaps() const {
  return _kernels.size();
}

size_t ConvolutionalLayer::getNumOutputMaps() const {
  return _kernels[0].size();
}

SubSamplingLayer::SubSamplingLayer(size_t scale): _scale(scale) {
}

void SubSamplingLayer::status() const {
  printf("+-------------------------------------------------------------+\n");
  printf("|                Sub-Sampling Factor = %-4d                   |\n", _scale);
}
  
size_t SubSamplingLayer::getScale() const {

}

void SubSamplingLayer::feedForward(vector<mat>& fouts, const vector<mat>& fins) {

  fouts.resize(fins.size());

  // Since nInputs == nOutputs for subsampling layer, I just use N.
  int N = fins.size();
  int batch_size = fins[0].getCols();

  vector<vector<mat> > iImgs(N), oImgs(N);
  for (int i=0; i<N; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i], _input_img_size);

  for (size_t i=0; i<N; ++i) {
    oImgs[i].resize(batch_size);
    for (int k=0; k<batch_size; ++k)
      oImgs[i][k] = downsample(iImgs[i][k], _scale);
  }

  if (fouts.size() != N)
    fouts.resize(N);

  for (int j=0; j<N; ++j)
    fouts[j] = reshapeImages2Vectors(oImgs[j]);
}

void SubSamplingLayer::backPropagate(vector<mat>& errors, const vector<mat>& fins,
    const vector<mat>& fouts, float learning_rate) {

  // FIXME A downsampling followed by upsampling may NOT give the same dimension.
  for (size_t i=0; i<errors.size(); ++i)
    errors[i] = upsample(errors[i], _scale);
}
