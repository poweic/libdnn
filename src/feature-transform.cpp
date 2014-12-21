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

// CSE stands for Check Stream Error
#define CSE(x) { if (!(x)) \
  throw std::runtime_error(RED_ERROR + "Failed when executing \33[33m"#x"\33[0m"); }

#define VECTOR std::vector
#define WHERE std
#include <operators.inl>
#undef VECTOR
#undef WHERE

std::map<FeatureTransform::Type, string> FeatureTransform::type2token = {
  {FeatureTransform::Affine, "affine"},
  {FeatureTransform::Sigmoid, "sigmoid"},
  {FeatureTransform::Softmax, "softmax"},
  {FeatureTransform::Dropout, "dropout"},
  {FeatureTransform::Convolution, "convolution"},
  {FeatureTransform::SubSample, "subsample"}
};

FeatureTransform::Type FeatureTransform::token2type(string token) {
  std::transform(token.begin(), token.end(), token.begin(), ::tolower);

  for (const auto& itr : type2token) {
    if (itr.second == token)
      return itr.first;
  }

  throw std::runtime_error("Unknown transform type: " + token);
}

string peek_a_token(istream& is) {

  string token;
  
  if (!is) return token;

  char buffer;
  is.read(&buffer, 1);

  while (buffer != ' ') {
    token.push_back(buffer);
    is.read(&buffer, 1);
  }

  is.putback(' ');
  for (int i=token.size() - 1; i>=0; --i)
    is.putback(token[i]);

  return token;
}

bool isXmlFormat(istream& is) {
  string token = peek_a_token(is);
  return token == "<transform" || token == "<?xml";
}

ostream& operator << (ostream& os, FeatureTransform* ft) {
  ft->write(os);
  return os;
}

istream& operator >> (istream& is, FeatureTransform* &ft) {
  string type;
  if (!(is >> type)) {
    ft = NULL;
    return is;
  }

  std::transform(type.begin(), type.end(), type.begin(), ::toupper);

  if (type == "<AFFINETRANSFORM>")
    ft = new AffineTransform(is);
  else if (type == "<SIGMOID>")
    ft = new Sigmoid(is);
  else if (type == "<SOFTMAX>")
    ft = new Softmax(is);
  else {
    ft = NULL;
    while (is >> type);
  }

  return is;
}

/*
 * class FeatureTransform
 *
 * */

FeatureTransform::FeatureTransform(size_t input_dim, size_t output_dim)
  : _input_dim(input_dim), _output_dim(output_dim) {
}

void FeatureTransform::read(xml_node<> *node) {
  auto attr = node->first_attribute("input-dim");
  if (!attr)
    throw std::runtime_error(RED_ERROR + "Missing input-dim");
  _input_dim = stol(attr->value());

  attr = node->first_attribute("output-dim");
  if (!attr)
    throw std::runtime_error(RED_ERROR + "Missing output-dim");
  _output_dim = stol(attr->value());
}

/*
 * class AffineTransform
 *
 * */
AffineTransform::AffineTransform(size_t input_dim, size_t output_dim)
  : FeatureTransform(input_dim, output_dim) {

}

AffineTransform::AffineTransform(const mat& w)
  : FeatureTransform(w.getRows() - 1, w.getCols() - 1), _w(w) {
}

AffineTransform::AffineTransform(istream& is) {
  this->read(is);
}

void AffineTransform::read(xml_node<> * node) {

  FeatureTransform::read(node);

  const size_t& rows = _input_dim;
  const size_t& cols = _output_dim;

  auto attr = node->first_attribute("learning-rate");
  float learning_rate = (attr) ? stof(attr->value()) : 0.1;

  attr = node->first_attribute("momentum");
  float momentum = (attr) ? stof(attr->value()) : 0.1;

  auto weight = node->first_node("weight");
  auto bias = node->first_node("bias");

  if (!weight)
    throw std::runtime_error("Cannot find weight in affine transform");

  if (!bias)
    throw std::runtime_error("Cannot find bias in affine transform");

  hmat hw(rows + 1, cols + 1);
  stringstream ss;
  
  ss << weight->value();
  for (size_t i=0; i<rows; ++i)
    for (size_t j=0; j<cols; ++j)
      CSE( ss >> hw(i, j) );

  ss << bias->value();
  for (size_t j=0; j<cols; ++j)
    CSE( ss >> hw(rows, j) );

  _w = (mat) hw;
}
 
void AffineTransform::read(istream& is) {

  string dummy;

  size_t rows, cols;
  CSE(is >> rows);
  CSE(is >> cols);

  hmat hw(rows + 1, cols + 1);

  // Read matrix
  CSE(is >> dummy);
  for (size_t i=0; i<rows; ++i)
    for (size_t j=0; j<cols; ++j)
      CSE( is >> hw(i, j) );
  CSE(is >> dummy);

  // Read vector (bias)
  CSE(is >> dummy);
  for (size_t j=0; j<cols; ++j)
    CSE( is >> hw(rows, j) );
  CSE(is >> dummy);

  _w = (mat) hw;
  _input_dim = rows;
  _output_dim = cols;
}

void AffineTransform::write(ostream& os) const {

  hmat data(_w);

  size_t rows = data.getRows(),
	 cols = data.getCols();

  char buffer[512];

  sprintf(buffer, "<transform type=\"%s\" input-dim=\"%lu\" output-dim=\"%lu\""
      " momentum=\"%f\" learning-rate=\"%f\" >", this->toString().c_str(),
      rows - 1, cols - 1, 0.1, 0.1);
  os << buffer << endl;


  sprintf(buffer, "  <weight rows=\"%lu\" cols=\"%lu\">", rows - 1, cols - 1);
  os << buffer << endl;

  // Write matrix
  for (size_t j=0; j<rows-1; ++j) {
    os << "    ";
    for (size_t k=0; k<cols-1; ++k)
      os << data[k * rows + j] << " ";
    os << endl;
  }

  os << "  </weight>" << endl;

  sprintf(buffer, "  <bias rows=\"%d\" cols=\"%lu\">", 1, cols - 1);
  os << buffer << endl;

  os << "    ";
  for (size_t j=0; j<cols-1; ++j)
    os << data[j * rows + rows - 1] << " ";

  os << endl
     << "  </bias>" << endl
     << "</transform>" << endl;
  

#if 0
  hmat data(_w);

  size_t rows = data.getRows(),
	 cols = data.getCols();

  os << "<" << this->toString() << "> " << (rows - 1) << " " << (cols - 1) << endl;

  // Write matrix
  os << "[";
  for (size_t j=0; j<rows-1; ++j) {
    os << "\n  ";
    for (size_t k=0; k<cols-1; ++k)
      os << data[k * rows + j] << " ";
  }
  os << "]\n";

  // Write vector (bias)
  os << "[ ";
  for (size_t j=0; j<cols-1; ++j)
    os << data[j * rows + rows - 1] << " ";
  os << "]\n";
#endif
}

void AffineTransform::status() const {
  // TODO
}

size_t AffineTransform::getNumParams() const {
  return getInputDimension() * getOutputDimension() + getOutputDimension();
}

AffineTransform* AffineTransform::clone() const {
  return new AffineTransform(*this);
}

string AffineTransform::toString() const {
  return "Affine";
}

void AffineTransform::feedForward(mat& fout, const mat& fin) {
  fout = add_bias(fin) * _w;
}

void AffineTransform::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {
  mat delta = error;

  error.resize(delta.getRows(), _w.getRows());

  // Perform error = delta(1:end-1) * _w(:, 1:end-1)^T
  // The last row of _w is bias. The last column of _w is reserved for
  // computational efficiency. Therefore, ignore the last column in _w.
  size_t traceLength = delta.getCols() - 1;

  device_matrix<float>::cublas_gemm(
      CUBLAS_OP_N, CUBLAS_OP_T,
      delta.getRows(), _w.getRows(), traceLength, 
      1.0,
      delta.getData(), delta.getRows(),
      _w.getData(), _w.getRows(),
      0.0,
      error.getData(), error.getRows());

  // FIXME later add_bias(fin) is weird !!
  gemm(add_bias(fin), delta, _w, -learning_rate, 1.0f, true, false);
}

mat& AffineTransform::get_w() {
  return _w;
}

mat const& AffineTransform::get_w() const {
  return _w;
}

/*
 * class Activation
 *
 * */

Activation::Activation() {
}

Activation::Activation(size_t input_dim, size_t output_dim)
  : FeatureTransform(input_dim, output_dim) {

}

void Activation::read(xml_node<> * node) {

  FeatureTransform::read(node);
  
  if (_input_dim != _output_dim)
    throw std::runtime_error(RED_ERROR + "Mismatched input/output dimension");
}

void Activation::read(istream& is) {

  string remaining;
  CSE(is >> _input_dim);
  CSE(is >> _output_dim);

  CSE(std::getline(is, remaining));
  
  if (_input_dim != _output_dim)
    throw std::runtime_error(RED_ERROR + "Mismatched input/output dimension");
}

void Activation::write(ostream& os) const {

  os << "<transform type=\"" << this->toString() 
     << "\" input-dim=\"" << _input_dim
     << "\" output-dim=\"" << _output_dim << "\" />" << endl;
#if 0
  os << "<" << this->toString() << "> " << _input_dim << " " << _output_dim << endl;
#endif
}

void Activation::status() const {
  // TODO
}

/*
 * class Sigmoid
 *
 * */

Sigmoid::Sigmoid(size_t input_dim, size_t output_dim)
  : Activation(input_dim, output_dim) {

}

Sigmoid::Sigmoid(istream& is) {
  Activation::read(is);
}

Sigmoid* Sigmoid::clone() const {
  return new Sigmoid(*this);
}

string Sigmoid::toString() const {
  return "Sigmoid";
}

void Sigmoid::feedForward(mat& fout, const mat& fin) {
  fout = sigmoid(fin);
}

void Sigmoid::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {
  error = error & (1.0f - fout) & fout;
}

/*
 * class Softmax
 *
 * */

Softmax::Softmax(size_t input_dim, size_t output_dim)
  : Activation(input_dim, output_dim) {

}

Softmax::Softmax(istream& is) {
  Activation::read(is);
}

Softmax* Softmax::clone() const {
  return new Softmax(*this);
}

string Softmax::toString() const {
  return "Softmax";
}

void Softmax::feedForward(mat& fout, const mat& fin) {
  fout = softmax(fin);
}

void Softmax::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {
  // Do nothing.
}

/*
 * class Dropout
 *
 * */
Dropout::Dropout(): _dropout_ratio(0.0f), _dropout(true) {
}

Dropout::Dropout(size_t input_dim, size_t output_dim)
  : Activation(input_dim, output_dim), _dropout_ratio(0.0f), _dropout(true) {
}

Dropout::Dropout(istream& is) {
  Activation::read(is);
}

void Dropout::read(xml_node<> *node) {

  Activation::read(node);

  auto attr = node->first_attribute("dropout-ratio");
  if (attr)
    _dropout_ratio = stof(attr->value());
}

void Dropout::write(ostream& os) const {
  stringstream ss;
  Activation::write(ss);
  string str = ss.str();
  str.insert(str.find_last_of(' '), " dropout-ratio=\"" + to_string(_dropout_ratio) + "\"");
  os << str;
}

Dropout* Dropout::clone() const {
  return new Dropout(*this);
}

string Dropout::toString() const {
  return "Dropout";
}

void Dropout::feedForward(mat& fout, const mat& fin) {
  if (!_dropout) {
    fout = fin * (1 - _dropout_ratio);
    return;
  }

  if (_dropout_ratio == 0) {
    fout = fin;
    return;
  }

  _dropout_mask = mat(fin.getRows(), fin.getCols(), 1 - _dropout_ratio);
  sample(_dropout_mask, BERNOULLI);
  fout = _dropout_mask & fin;
}

void Dropout::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {
  if (_dropout_ratio != 0)
    error &= _dropout_mask;
}

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
    throw std::runtime_error(RED_ERROR + "Missing input-maps");
  _n_input_maps = stol(attr->value());

  // # of output feature maps
  attr = node->first_attribute("output-maps");
  if (!attr)
    throw std::runtime_error(RED_ERROR + "Missing output-maps");
  _n_output_maps = stol(attr->value());

  // Input dimension of image
  attr = node->first_attribute("input-dim");
  if (!attr)
    throw std::runtime_error(RED_ERROR + "Missing input-dim");
  this->set_input_img_size(parseImageDimension(attr->value()));
}

void MIMOFeatureTransform::write(ostream& os) const {
  char buffer[256];
  sprintf(buffer, "input-dim=\"%lux%lu\" input-maps=\"%lu\" output-maps=\"%lu\"",
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
  ft->write(os);
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

  SIZE k = parseImageDimension(node->first_attribute("kernel-dim")->value());

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
  sprintf(buffer, "<transform type=\"%s\" learning-rate=\"%f\" kernel-dim=\"%lux%lu\" %s>",
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

void ConvolutionalLayer::feedForward(mat& fout, const mat& fin) {

  auto fins = versplit(fin, getNumInputMaps());

  size_t nInputs  = getNumInputMaps(),
	 nOutputs = getNumOutputMaps();

  if (fins.size() != nInputs)
    throw std::runtime_error(RED_ERROR + "Number of inputs maps ( = "
	+ to_string(fins.size()) + ") does not match number of kernels ( = "
	+ to_string(nInputs) + ").");

  size_t batch_size = fins[0].getCols();

  SIZE s = get_output_img_size();

  vector<mat> fouts(nOutputs);

  for (size_t j=0; j<nOutputs; ++j)
    fouts[j].resize(s.m * s.n, batch_size, _bias[j]);

  for (size_t j=0; j<nOutputs; ++j) {
    for (size_t i=0; i<nInputs; ++i)
      fouts[j] += convn(fins[i], _kernels[i][j], _input_img_size, VALID_SHM);
  }

  fout = vercat(fouts);
}

void ConvolutionalLayer::feedBackward(mat& error, const mat& delta) {

  vector<mat> deltas = versplit(delta, getNumOutputMaps());

  // Since nInputs == nOutputs for subsampling layer, I just use N.
  size_t nInputs = getNumInputMaps(),
	 nOutputs = getNumOutputMaps();

  SIZE s = this->get_input_img_size();
  size_t batch_size = deltas[0].getCols();

  vector<mat> errors(nInputs);

  for (size_t i=0; i<nInputs; ++i)
    errors[i].resize(s.m * s.n, batch_size, 0);

  for (size_t i=0; i<nInputs; ++i)
    for (size_t j=0; j<nOutputs; ++j)
      errors[i] += convn(deltas[j], rot180(_kernels[i][j]), this->get_output_img_size(), FULL_SHM);

  error = vercat(errors);
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

void ConvolutionalLayer::backPropagate(mat& error, const mat& fin,
    const mat& fout, float learning_rate) {

  size_t batch_size = fin.getCols();
  size_t nInputs = getNumInputMaps();
  size_t nOutputs = getNumOutputMaps();

  // In the following codes, the iteration index i and j stands for
  // i : # of input  features. i = 0 ~ nInputs - 1 
  // j : # of output features. j = 0 ~ nOutputs - 1

  vector<mat> deltas = versplit(error * learning_rate, nOutputs);

  this->feedBackward(error, mat(error) );

  // iImgs represents the input images.
  vector<vector<mat> > iImgs(nInputs);
  vector<mat> fins = versplit(fin, nInputs);

  for (size_t i=0; i<nInputs; ++i)
    iImgs[i] = reshapeVectors2Images(fins[i], _input_img_size);

  auto Y = reshapeVectors2Images(vercat(deltas), SIZE(deltas[0].getRows(), nOutputs));

  // Update kernels with learning rate
  vector<mat> Z(nInputs, mat(this->get_kernel_size().area(), nOutputs, 0));

  for (size_t i=0; i<nInputs; ++i)
    for (size_t k=0; k<batch_size; ++k)
      Z[i] += convn_2(rot180(iImgs[i][k]), Y[k], this->get_output_img_size());

  for (size_t i=0; i<nInputs; ++i)
    _kernels[i] -= reshapeVectors2Images(Z[i], this->get_kernel_size());

  for (size_t j=0; j<nOutputs; ++j)
    _bias[j] -= sum_all(deltas[j]);
}

size_t ConvolutionalLayer::getInputDimension() const {
  return get_input_img_size().area() * getNumInputMaps();
}

size_t ConvolutionalLayer::getOutputDimension() const {
  return get_output_img_size().area() * getNumOutputMaps();
}

void ConvolutionalLayer::status() const {

  printf("+--------------+---------------+--------------+---------------+\n");
  printf("|      %-5lu   |       %-5lu   |      %-5lu   |       %-5lu   |\n",
      getNumInputMaps(), getNumOutputMaps(), getKernelWidth(), getKernelHeight());
}

size_t ConvolutionalLayer::getNumParams() const {
  return getNumInputMaps() * getNumOutputMaps() 
    * getKernelWidth() * getKernelHeight() + getNumOutputMaps();
}

SIZE ConvolutionalLayer::get_output_img_size() const {
  SIZE kernel(getKernelHeight(), getKernelWidth());
  return get_convn_size(_input_img_size, kernel, VALID);
}

SIZE ConvolutionalLayer::get_kernel_size() const {
  return SIZE(_kernels[0][0].getRows(), _kernels[0][0].getCols());
}

size_t ConvolutionalLayer::getKernelWidth() const {
  return _kernels[0][0].getCols();
}

size_t ConvolutionalLayer::getKernelHeight() const {
  return _kernels[0][0].getRows();
}

SubSamplingLayer::SubSamplingLayer(size_t m, size_t n, size_t scale)
  : MIMOFeatureTransform(m, n), _scale(scale) {
}

void SubSamplingLayer::read(xml_node<> *node) {
  MIMOFeatureTransform::read(node);

  auto attr = node->first_attribute("sample-rate");
  if (!attr)
    throw std::runtime_error(RED_ERROR + "Missing sample-rate");
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

size_t SubSamplingLayer::getInputDimension() const {
  return get_input_img_size().area() * getNumInputMaps();
}

size_t SubSamplingLayer::getOutputDimension() const {
  return get_output_img_size().area() * getNumOutputMaps();
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

void SubSamplingLayer::feedForward(mat& fout, const mat& fin) {

  vector<mat> fins = versplit(fin, getNumInputMaps());

  vector<mat> fouts(fins.size());
  for (size_t i=0; i<fouts.size(); ++i)
    fouts[i] = downsample(fins[i], _scale, _input_img_size);

  fout = vercat(fouts);
}

void SubSamplingLayer::feedBackward(mat& error, const mat& delta) {

  vector<mat> deltas = versplit(delta, getNumOutputMaps());

  vector<mat> errors(deltas.size());
  for (size_t i=0; i<errors.size(); ++i)
    errors[i] = upsample(deltas[i], _input_img_size, get_output_img_size());

  error = vercat(errors);// TODO / (_scale * _scale);
}

void SubSamplingLayer::backPropagate(mat& error, const mat& fin,
    const mat& fout, float learning_rate) {
  // Copy errors element by element to deltas
  this->feedBackward(error, error);
}
