// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <feature-transform.h>
#include <cnn.h>
#define matslog(x) { for (int i=0; i<x.size(); ++i) { printf(#x"[%d] = [\n", i); x[i].print(); printf("]\n"); } }

// CSE stands for Check Stream Error
#define CSE(x) { if (!(x)) \
  throw std::runtime_error("\33[31m[Error]\33[0m Failed when executing \33[33m"#x"\33[0m"); }

/*!
 * Implementation of MIMOFeatureTransform goes here.
 */

MIMOFeatureTransform::MIMOFeatureTransform(size_t n_input_maps, size_t n_output_maps):
  _n_input_maps(n_input_maps), _n_output_maps(n_output_maps) {
  // nothing to do  
}

void MIMOFeatureTransform::read(xml_node<> *node) {
  // # of input feature maps
  auto attr = node->first_attribute("input-maps");
  if (!attr)
    throw std::runtime_error("\33[31m[Error]\33[0m Missing input-maps");
  _n_input_maps = stol(attr->value());

  // # of output feature maps
  attr = node->first_attribute("output-maps");
  if (!attr)
    throw std::runtime_error("\33[31m[Error]\33[0m Missing output-maps");
  _n_output_maps = stol(attr->value());

  // Input dimension of image
  attr = node->first_attribute("input-dim");
  if (!attr)
    throw std::runtime_error("\33[31m[Error]\33[0m Missing input-dim");
  this->set_input_img_size(parseInputDimension(attr->value()));
}

void MIMOFeatureTransform::write(ostream& os) const {
  char buffer[256];
  sprintf(buffer, "input-dim=\"%lu-%lu\" input-maps=\"%lu\" output-maps=\"%lu\"",
      _input_img_size.m, _input_img_size.n, _n_input_maps, _n_output_maps);
  os << buffer;
}

void MIMOFeatureTransform::set_input_img_size(const SIZE& s) {
  _input_img_size = s;
}

SIZE MIMOFeatureTransform::get_input_img_size() const {
  return _input_img_size;
}

size_t MIMOFeatureTransform::getNumInputMaps() const {
  return _n_input_maps;
}

size_t MIMOFeatureTransform::getNumOutputMaps() const {
  return _n_output_maps;
}

ostream& operator << (ostream& os, const MIMOFeatureTransform *ft) {
  // os << ft->get_input_img_size() << " => " << ft->get_output_img_size();
  ft->write(os);
  return os;
}

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
	     kernel_height = str2int(dims[1]),
	     kernel_width = str2int(dims[2]);

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
  ifstream fin(fn.c_str());

  if (!fin.is_open())
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + fn);

  stringstream ss;
  ss << fin.rdbuf() << '\0';
  fin.close();

  _transforms.clear();

  MIMOFeatureTransform* f;

  if (isXmlFormat(ss)) {
    rapidxml::xml_document<> doc;

    vector<char> buffer((istreambuf_iterator<char>(ss)), istreambuf_iterator<char>());
    buffer.push_back('\0');
    doc.parse<0>(&buffer[0]);

    for (auto node = doc.first_node("transform"); node; node = node->next_sibling()) {

      auto x = node->first_attribute("type");

      string token = node->first_attribute("type")->value();
      FeatureTransform::Type type = FeatureTransform::token2type(token);

      switch (type) {
	case FeatureTransform::Affine :
	case FeatureTransform::Sigmoid :
	case FeatureTransform::Softmax :
	  return;
	case FeatureTransform::Convolution : 
	  f = new ConvolutionalLayer;
	  break;
	case FeatureTransform::SubSample :
	  f = new SubSamplingLayer;
	  break;
	default:
	  cerr << "\33[31m[Error]\33[0m Not such type " << token << endl;
	  break;
      }
      

      if (f) {
	f->read(node);
	_transforms.push_back(f);
      }
    }

  }
  else
    clog << "\31[31m[Error]\33[0m Error while reading XML file." << endl;
}

void CNN::save(const string &fn) const {
  ofstream fout(fn.c_str());

  if (!fout.is_open())
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot open file: " + fn);

  fout << *this;

  fout.close();
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

ostream& operator << (ostream& os, const CNN& cnn) {
  for (size_t i=0; i<cnn._transforms.size(); ++i)
    os << cnn._transforms[i];
  return os;
}


/*! 
 * Implementation of ConvolutionalLayer goes here.
 */

ConvolutionalLayer::ConvolutionalLayer(size_t nInputs, size_t nOutputs, int h, int w)
  : MIMOFeatureTransform(nInputs, nOutputs) {
  if (w == -1)
    w = h;

  assert(nInputs > 0 && nOutputs > 0 && h > 0 && w > 0);

  printf("Initializing %lu x %lu kernels of size %d x %d\n", nInputs, nOutputs, h, w);

  size_t fan_in = nInputs * w * h,
	 fan_out = nOutputs * w * h;
  float coeff = 2 * sqrt(6.0f / (fan_in + fan_out));

  _kernels.resize(nInputs);
  for (size_t i=0; i<nInputs; ++i) {
    _kernels[i].resize(nOutputs);
    for (size_t j=0; j<nOutputs; ++j) {
      _kernels[i][j] = (rand(h, w) - 0.5f) * coeff;
#ifdef DEBUG
      _kernels[i][j].print();
#endif
    }
  }

  _bias.resize(nOutputs);
  for (size_t j=0; j<nOutputs; ++j)
    _bias[j] = 0;
}

void ConvolutionalLayer::read(xml_node<> *node) {

  MIMOFeatureTransform::read(node);

  SIZE k = parseInputDimension(node->first_attribute("kernel-dim")->value());

  // Allocate memory for kernels and bias
  _kernels.resize(_n_input_maps);
  for (size_t i=0; i<_kernels.size(); ++i)
    _kernels[i].resize(_n_output_maps);
  _bias.resize(_n_output_maps);

  // Parse kernels and bias
  int j = 0;
  for (auto kernels = node->first_node("kernels"); kernels; kernels = kernels->next_sibling(), ++j) {
    int i = 0;
    for (auto w = kernels->first_node("weight"); w; w = w->next_sibling(), i++) {
      stringstream ss(w->value());

      hmat hw(k.m, k.n);
      for (size_t x=0; x<k.m; ++x)
	for (size_t y=0; y<k.n; ++y)
	  CSE( ss >> hw(x, y) );

      _kernels[i][j] = (mat) hw;
    }
    _bias[j] = stof(kernels->first_attribute("bias")->value());
  }
}

void ConvolutionalLayer::write(ostream& os) const {
  ostringstream oss;
  MIMOFeatureTransform::write(oss);

  char buffer[256];
  sprintf(buffer, "<transform type=\"%s\" learning-rate=\"%f\" kernel-dim=\"%lu-%lu\" %s>",
      "convolution", 0.01, getKernelWidth(), getKernelHeight(), oss.str().c_str());
  os << buffer << endl;

  for (size_t j=0; j<_n_output_maps; ++j) {
    os << "  <kernels bias=\"" << _bias[j] << "\">" << endl;

    for (size_t i=0; i<_n_input_maps; ++i) {
      os << "    <weight>" << endl;
      hmat hw(_kernels[i][j]);
      for (size_t x=0; x<hw.getRows(); ++x) {
	os << "      ";
	for (size_t y=0; y<hw.getCols(); ++y)
	  os << hw(x, y) << " ";
	os << endl;
      }
      os << "    </weight>" << endl;
    }

    os << "  </kernels>" << endl;
  }
  os << "</transform>" << endl;
}

ConvolutionalLayer* ConvolutionalLayer::clone() const {
  return new ConvolutionalLayer(*this);
}

string ConvolutionalLayer::toString() const {
  return "convolution";
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
      fouts[j] += convn(fins[i], _kernels[i][j], _input_img_size, VALID_SHM);
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

  if (errors.size() != nInputs)
    errors.resize(nInputs);

  for (size_t i=0; i<nInputs; ++i)
    errors[i].resize(s.m * s.n, batch_size, 0);

  for (size_t i=0; i<nInputs; ++i)
    for (size_t j=0; j<nOutputs; ++j)
      errors[i] += convn(deltas[j], rot180(_kernels[i][j]), this->get_output_img_size(), FULL_SHM);
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

  assert(learning_rate > 0);
  float lr = learning_rate / batch_size;

  // iImgs represents the input images.
  // oImgs represents the output images. (Before sigmoid or any other activation function)
  vector<vector<mat> > iImgs(nInputs), oImgs(nOutputs);

  for (size_t i=0; i<nInputs; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i], _input_img_size);

  for (size_t j=0; j<nOutputs; ++j)
    oImgs[j] = reshapeVectors2Images(deltas[j], this->get_output_img_size());

  // Update kernels with learning rate
  for (size_t k=0; k<batch_size; ++k) {
    for (size_t j=0; j<nOutputs; ++j) {

      for (size_t i=0; i<nInputs; ++i)
	_kernels[i][j] -= convn(rot180(iImgs[i][k]), oImgs[j][k], VALID_SHM) * lr;

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

SIZE ConvolutionalLayer::get_output_img_size() const {
  SIZE kernel(getKernelHeight(), getKernelWidth());
  return get_convn_size(_input_img_size, kernel, VALID);
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

void SubSamplingLayer::read(xml_node<> *node) {
  MIMOFeatureTransform::read(node);

  auto attr = node->first_attribute("sample-rate");
  if (!attr)
    throw std::runtime_error("\33[31m[Error]\33[0m Missing sample-rate");
  _scale = stol(attr->value());
}

void SubSamplingLayer::write(ostream& os) const {
  ostringstream oss;
  MIMOFeatureTransform::write(oss);

  char buffer[256];
  sprintf(buffer, "<transform type=\"%s\" sample-rate=\"%lu\" %s>",
      "subsample", getScale(), oss.str().c_str());
  os << buffer << endl;

  os << "</transform>" << endl;
}

SubSamplingLayer* SubSamplingLayer::clone() const {
  return new SubSamplingLayer(*this);
}

string SubSamplingLayer::toString() const {
  return "subsample";
}

void SubSamplingLayer::status() const {
  printf("+-------------------------------------------------------------+\n");
  printf("|                Sub-Sampling Factor = %-4lu                   |\n", _scale);
}
  
size_t SubSamplingLayer::getScale() const {
  return _scale;
}

SIZE SubSamplingLayer::get_output_img_size() const {
  return _input_img_size / _scale;
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

  if (errors.size() != N)
    errors.resize(N);

  for (size_t i=0; i<N; ++i)
    errors[i] = upsample(deltas[i], _input_img_size, this->get_output_img_size()) / (_scale * _scale);
}

void SubSamplingLayer::backPropagate(vector<mat>& errors, const vector<mat>& fins,
    const vector<mat>& fouts, float learning_rate) {

  // Copy errors element by element to deltas
  this->feedBackward(errors, errors);
}
