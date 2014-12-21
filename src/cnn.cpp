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

// CSE stands for Check Stream Error
#define CSE(x) { if (!(x)) \
  throw std::runtime_error(RED_ERROR + "Failed when executing \33[33m"#x"\33[0m"); }

#define VECTOR std::vector
#define WHERE std
#include <operators.inl>
#undef VECTOR
#undef WHERE

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

  mat fin_t = fin;
  if (_transforms[0]->toString() == "convolution")
    fin_t = removeBiasAndTranspose(fin_t);

  // FIXME SubSamplingLayer does NOT need temporary buffer.
  // MAYBE just reserve those for ConvolutionalLayer.
  _houts.resize(_transforms.size() - 1);

  if (_houts.size() > 0) {
    _transforms[0]->feedForward(_houts[0], fin_t);

    for (size_t i=1; i<_transforms.size() - 1; ++i) {
      _transforms[i]->feedForward(_houts[i], _houts[i-1]);

      // Handle boundary between CNN and DNN
      if ( is_cnn_dnn_boundary(i) ) {
	// Add one more column, which is bias in DNN.
	_houts[i] = ~_houts[i];
	_houts[i] = add_bias(_houts[i], 1.0f, true);
      }
    }

    _transforms.back()->feedForward(fout, _houts.back());
  }
  else
    _transforms.back()->feedForward(fout, fin_t);

  fout.resize(fout.getRows(), fout.getCols() - 1);
}

void CNN::backPropagate(mat& error, const mat& fin, const mat& fout,
    float learning_rate) {

  // Copy from dnn.cpp -- begin
  mat output = add_bias(fout, 1.0f, true);
  error = add_bias(error, 1.0f, true);
  // Copy from dnn.cpp -- end

  _transforms.back()->backPropagate(error, _houts.back(), output, learning_rate);

  for (int i=_transforms.size() - 2; i >= 1; --i) {
    _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], learning_rate);

    if ( is_cnn_dnn_boundary (i-1) ) {
      // Remove last column, which is bias in DNN.
      _houts[i-1] = removeBiasAndTranspose(_houts[i-1]);
      error = removeBiasAndTranspose(error);
    }
  }

  mat fin_t = fin;
  if (_transforms[0]->toString() == "convolution")
    fin_t = removeBiasAndTranspose(fin_t);

  _transforms[0]->backPropagate(error, fin_t, _houts[0], learning_rate);
  error = ~error;
}

void CNN::feedBackward(mat& error, const mat& delta) {
  // TODO
}

void CNN::init(const string &structure, SIZE img_size) {

  // Parse structure
  vector<string> layers = split(structure, '-');

  size_t nInputMaps = 1;

  for (size_t i=0; i<layers.size(); ++i) {

    if (layers[i].find("s") != string::npos) { // "s" means sub-sampling
      size_t scale = str2int(layers[i].substr(0, layers[i].size() - 1));

      size_t nOutputMaps = nInputMaps;
      MIMOFeatureTransform* t = new SubSamplingLayer( nInputMaps, nOutputMaps, scale);
      t->set_input_img_size(img_size);
      _transforms.push_back(t);

      // Set the input img_size of next layer to be the output img_size of current layer.
      img_size = t->get_output_img_size();
    }
    else if (layers[i].find("x") != string::npos) { // "x" in kernel "m x n"

      vector<string> dims = split(layers[i], 'x');

      size_t nOutputMaps   = str2int(dims[0]),
	     kernel_height = str2int(dims[1]),
	     kernel_width  = str2int(dims[2]);

      MIMOFeatureTransform* t =
	new ConvolutionalLayer( nInputMaps, nOutputMaps, kernel_height, kernel_width);

      t->set_input_img_size(img_size);

      _transforms.push_back(t);

      // Set the input img_size of next layer to be the output img_size of current layer.
      img_size = t->get_output_img_size();
      nInputMaps = nOutputMaps;

      // Add Sigmoid activation
      FeatureTransform* activation =
        new Sigmoid(t->getOutputDimension(), t->getOutputDimension());

      _transforms.push_back(activation);
    }
    else if ( is_number(layers[i]) ) { // pure number means a hidden layer
      size_t fan_in = _transforms.back()->getOutputDimension();
      size_t fan_out = stoi(layers[i]);

      float coeff = 2 * sqrt(6.0f / (fan_in + fan_out + 2) );
      mat weight = coeff * (rand(fan_in + 1, fan_out + 1) - 0.5);
      _transforms.push_back(new AffineTransform(weight));

      if ( i < layers.size() - 1 )
	_transforms.push_back(new Sigmoid(fan_out, fan_out));
      else
	_transforms.push_back(new Softmax(fan_out, fan_out));
    }
    else
      throw std::runtime_error(RED_ERROR + "No such type of layer. \""
	  + layers[i] + "\". Only convolutional/sub-sampling layer are allowed");
  }
}

void CNN::read(const string &fn) {
  ifstream fin(fn.c_str());

  if (!fin.is_open())
    throw std::runtime_error(RED_ERROR + "Cannot load file: " + fn);

  printf("\33[34m[Info]\33[0m Reading model from \33[32m%s\33[0m\n", fn.c_str());

  stringstream ss;
  ss << fin.rdbuf() << '\0';
  fin.close();

  _transforms.clear();

  FeatureTransform* f;

  if (isXmlFormat(ss)) {
    rapidxml::xml_document<> doc;

    vector<char> buffer((istreambuf_iterator<char>(ss)), istreambuf_iterator<char>());
    buffer.push_back('\0');
    doc.parse<0>(&buffer[0]);

    for (auto node = doc.first_node("transform"); node; node = node->next_sibling()) {

      string token = node->first_attribute("type")->value();
      FeatureTransform::Type type = FeatureTransform::token2type(token);

      switch (type) {
	case FeatureTransform::Affine :
	  f = new AffineTransform;
	  break;
	case FeatureTransform::Sigmoid :
	  f = new Sigmoid;
	  break;
	case FeatureTransform::Softmax :
	  f = new Softmax;
	  break;
	case FeatureTransform::Dropout :
	  f = new Dropout;
	  break;
	case FeatureTransform::Convolution : 
	  f = new ConvolutionalLayer;
	  break;
	case FeatureTransform::SubSample :
	  f = new SubSamplingLayer;
	  break;
	default:
	  cerr << RED_ERROR << "Not such type " << token << endl;
	  break;
      }

      if (f) {
	f->read(node);
	_transforms.push_back(f);
      }
    }

  }
  else
    clog << RED_ERROR << "while reading XML file." << endl;
}

void CNN::save(const string &fn) const {
  ofstream fout(fn.c_str());

  if (!fout.is_open())
    throw std::runtime_error(RED_ERROR + "Cannot open file: " + fn);

  fout << *this;

  fout.close();
}

size_t CNN::getInputDimension() const { 
  return _transforms[0]->getInputDimension();
}

size_t CNN::getOutputDimension() const { 
  return _transforms.back()->getOutputDimension();
}

void CNN::status() const {

  const auto& t = _transforms;

  int nHiddens = 0;

  printf("._____._____________.___________.___________._________._________.____________.\n");
  printf("|     |             |           |           |         |         |            |\n");
  printf("|     |  Transform  |   Input   |  Output   | kernel  | kernel  | Number of  |\n");
  printf("| No. |             |           |           |         |         |            |\n");
  printf("|     |    Type     | Dimension | Dimension | number  |  size   | Parameters |\n");
  printf("|_____|_____________|___________|___________|_________|_________|____________|\n");
  printf("|     |             |           |           |         |         |            |\n");

  for (size_t i=0; i<t.size(); ++i) {
    string type = t[i]->toString();
    size_t in  = t[i]->getInputDimension(),
	   out = t[i]->getOutputDimension();

    std::transform(type.begin(), type.end(), type.begin(), ::tolower);

    if (type == "affine")
      ++nHiddens;

    // create string for kernel size
    ConvolutionalLayer* ptr = dynamic_cast<ConvolutionalLayer*>(t[i]);
    string kernel_size, kernel_number;
    if (ptr != nullptr) {
      kernel_size = ptr->get_kernel_size();
      kernel_number = to_string(ptr->getNumInputMaps()) +
	" x " + to_string(ptr->getNumOutputMaps());
    }
    else
      kernel_size = kernel_number = "\33[1;30m  N/A  \33[0m";

    // Compute Number of parameters in this layer.
    char nParamStr[32] = {'\0'};

    float nParams = t[i]->getNumParams();
    
    if (nParams > 1e8)
      sprintf(nParamStr, "~ %6.3f G", nParams / 1e9);
    else if (nParams > 1e5)
      sprintf(nParamStr, "~ %6.3f M", nParams / 1e6);
    else if (nParams > 1e2)
      sprintf(nParamStr, "~ %6.3f K", nParams / 1e3);
    else if (nParams > 0)
      sprintf(nParamStr, "  %5d   ", (int) nParams);
    else
      sprintf(nParamStr, "\33[1;30m       N/A\33[0m");

    printf("|  %-2lu | %-11s |  %6lu   |  %6lu   | %7s | %7s | %10s |\n",
	i, type.c_str(), in, out, kernel_number.c_str(), kernel_size.c_str(), nParamStr);
    printf("|     |             |           |           |         |         |            |\n");
  }

  printf("|_____|_____________|___________|___________|_________|_________|____________|\n");

  nHiddens = std::max(0, nHiddens - 1);
  printf("Number of hidden layers: %2d \n", nHiddens);
}

bool CNN::is_cnn_dnn_boundary(size_t i) const {

  // the boundary between CNN and DNN must be:
  // a instance of MIMOFeatureTransform -> affine
  // and this affine transform must the first one to encounter after CNN.
  
  bool has_mimo = false;
  for (size_t x=0; x<_transforms.size(); ++x) {
    const auto& t = _transforms[x];

    if (dynamic_cast<MIMOFeatureTransform*>(t) != nullptr)
      has_mimo = true;

    if (has_mimo && dynamic_cast<AffineTransform*>(t) != nullptr)
      return (x == i + 1);
  }

  return false;
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
