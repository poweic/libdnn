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

CNN::~CNN() {
  for (size_t i=0; i<_transforms.size(); ++i)
    delete _transforms[i];
}

// Perform the reverse of concat
vector<mat> de_concat(const mat& concated_features, int N) {

  int batch_size = concated_features.getCols();
  vector<mat> smalls(N);

  int MAP_SIZE = concated_features.size() / N;

  SIZE s(MAP_SIZE / batch_size, batch_size);
  
  for (int i=0; i<N; ++i) {
    smalls[i].resize(s.m, s.n);
    CCE(cudaMemcpy(smalls[i].getData(),
		   concated_features.getData() + i * MAP_SIZE,
	  	   sizeof(float) * MAP_SIZE,
	  	   cudaMemcpyDeviceToDevice));
  }
  CCE(cudaDeviceSynchronize());

  return smalls;
}

mat concat(const vector<mat>& smalls) {
  int nFeatures = smalls.size(),
      img_size  = smalls[0].getRows(),
      batchSize = smalls[0].getCols();

  mat big(img_size * nFeatures, batchSize);

  int MAP_SIZE = smalls[0].size();

  for (int i=0; i<nFeatures; ++i) {
    CCE(cudaMemcpy(big.getData() + i * MAP_SIZE,
		   smalls[i].getData(),
		   sizeof(float) * MAP_SIZE,
		   cudaMemcpyDeviceToDevice));
  }

  CCE(cudaDeviceSynchronize());

  return big;
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

  _transforms[0]->feedForward(_houts[0], fins);

  for (size_t i=1; i<_transforms.size(); ++i)
    _transforms[i]->feedForward(_houts[i], _houts[i-1]);

  // Concatenate
  fout = ~concat(_houts.back());
}

void CNN::backPropagate(mat& error, const mat& fin, const mat& fout,
    float learning_rate) {

  int N = _transforms.back()->getNumOutputMaps();
  vector<mat> fouts = de_concat(~fout, N),
	      errors = de_concat(~error, N);

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
ConvolutionalLayer::ConvolutionalLayer(size_t n, size_t m, int h, int w)
  : MIMOFeatureTransform(n, m) {
  if (w == -1)
    w = h;

  assert(n > 0 && m > 0 && h > 0 && w > 0);

  printf("Initializing %lu x %lu kernels of size %d x %d\n", n, m, h, w);
  _kernels.resize(n);
  for (size_t i=0; i<n; ++i)
    _kernels[i].assign(m, rand(h, w));

  _bias.resize(m);
  for (size_t j=0; j<m; ++j)
    _bias[j] = 0;
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

  size_t nInputs  = getNumInputMaps(),
	 nOutputs = getNumOutputMaps();

  if (fins.size() != nInputs)
    throw std::runtime_error(RED_ERROR + "Number of inputs maps ( = "
	+ to_string(fins.size()) + ") does not match number of kernels ( = "
	+ to_string(nInputs) + ").");

  size_t batch_size = fins[0].getCols();

  vector<vector<mat> > iImgs(nInputs), oImgs(nOutputs);

  SIZE s = get_output_img_size();

  for (size_t i=0; i<nInputs; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i], _input_img_size);

  // Allocate memory and initialize with value 0
  for (size_t j=0; j<nOutputs; ++j) {
    oImgs[j].resize(batch_size);

    for (size_t k=0; k<batch_size; ++k)
      oImgs[j][k].resize(s.m, s.n, 0);
  }

  for (size_t k=0; k<batch_size; ++k) {
    for (size_t j=0; j<nOutputs; ++j) {
      for (size_t i=0; i<nInputs; ++i)
	oImgs[j][k] += convn(iImgs[i][k], _kernels[i][j], "valid_shm");
      oImgs[j][k] += _bias[j];
    }
  }

  if (fouts.size() != nOutputs)
    fouts.resize(nOutputs);

  for (size_t j=0; j<nOutputs; ++j)
    fouts[j] = sigmoid(reshapeImages2Vectors(oImgs[j]));

}

void ConvolutionalLayer::feedBackward(
    vector<mat>& errors, const vector<mat>& deltas) {

  // Since nInputs == nOutputs for subsampling layer, I just use N.
  size_t nInputs = getNumInputMaps(),
	 nOutputs = getNumOutputMaps();

  SIZE s = this->get_input_img_size();
  size_t batch_size = deltas[0].getCols();

  /* ... DEBUG CODES BEGIN HERE ... */
  /*errors.resize(nInputs);
  for (int i=0; i<nInputs; ++i)
    errors[i].resize(s.m * s.n, batch_size);
  return;*/
  /* ... DEBUG CODES  END  HERE ... */

  vector<vector<mat> > oImgs(nOutputs), iImgs(nInputs);
  for (size_t j=0; j<nOutputs; ++j)
    oImgs[j] = reshapeVectors2Images(deltas[j], this->get_output_img_size());


  for (size_t i=0; i<nInputs; ++i)
    iImgs[i].resize(batch_size);

  // FIXME beware that upsample may NOT be able to get back to original size
  for (size_t k=0; k<batch_size; ++k) {
    for (size_t i=0; i<nInputs; ++i) {
      iImgs[i][k].resize(s.m, s.n, 0);
      for (size_t j=0; j<nOutputs; ++j) {
	iImgs[i][k] += convn(oImgs[j][k], rot180(_kernels[i][j]), "full");
      }
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
  
  this->feedBackward(errors, deltas);
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
  size_t batch_size = fins[0].getCols();

  vector<vector<mat> > iImgs(N), oImgs(N);
  for (size_t i=0; i<N; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i], _input_img_size);

  for (size_t i=0; i<N; ++i) {
    oImgs[i].resize(batch_size);
    for (size_t k=0; k<batch_size; ++k)
      oImgs[i][k] = downsample(iImgs[i][k], _scale);
  }

  if (fouts.size() != N)
    fouts.resize(N);

  for (size_t j=0; j<N; ++j)
    fouts[j] = reshapeImages2Vectors(oImgs[j]);
}

void SubSamplingLayer::feedBackward(
    vector<mat>& errors, const vector<mat>& deltas) {

  // Since nInputs == nOutputs for subsampling layer, I just use N.
  size_t N = deltas.size();
  size_t batch_size = deltas[0].getCols();

  /* ... DEBUG CODES BEGIN HERE ... */
  /*errors.resize(N);
  for (int i=0; i<N; ++i)
    errors[i].resize(_input_img_size.m * _input_img_size.n, batch_size);
  return;*/
  /* ... DEBUG CODES  END  HERE ... */

  vector<vector<mat> > oImgs(N), iImgs(N);
  for (size_t i=0; i<N; ++i)
    oImgs[i] = reshapeVectors2Images(deltas[i], this->get_output_img_size());

  // FIXME beware that upsample may NOT be able to get back to original size
  for (size_t i=0; i<N; ++i) {
    iImgs[i].resize(batch_size);
    for (size_t k=0; k<batch_size; ++k)
      iImgs[i][k] = upsample(oImgs[i][k], _scale, _input_img_size);
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
