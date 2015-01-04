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

#include <nnet.h>

/*! 
 * Implementation of NNet goes here.
 */

NNet::NNet(): _transforms() {

}

NNet::NNet(const string& model_fn) : _transforms() {
  this->read(model_fn);
}

NNet::~NNet() {
  for (size_t i=0; i<_transforms.size(); ++i)
    delete _transforms[i];
}

mat NNet::feedForward(const mat& fin) const {
  mat output(fin);

  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i]->feedForward( output, mat(output) );

  output = remove_bias(output);
  return output;
}

void NNet::feedForward(mat& fout, const mat& fin) {

  // FIXME SubSamplingLayer does NOT need temporary buffer.
  // MAYBE just reserve those for ConvolutionalLayer.
  _houts.resize(_transforms.size() - 1);

  if (_houts.size() > 0) {
    _transforms[0]->feedForward(_houts[0], fin);

    for (size_t i=1; i<_transforms.size() - 1; ++i)
      _transforms[i]->feedForward(_houts[i], _houts[i-1]);

    _transforms.back()->feedForward(fout, _houts.back());
  }
  else
    _transforms.back()->feedForward(fout, fin);
}

void NNet::backPropagate(mat& error, const mat& fin, const mat& fout, float lr) {

  _transforms.back()->backPropagate(error, _houts.back(), fout, lr);

  for (int i=_transforms.size() - 2; i >= 1; --i)
    _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], lr);

  _transforms[0]->backPropagate(error, fin, _houts[0], lr);
}

void NNet::feedBackward(mat& error, const mat& delta) {
  // TODO
}

void NNet::init(const string &structure) {

  // Parse structure
  vector<string> layers = split(structure, '-');

  // First token in layers is [# of input images]x[height]x[width], like 3x64x64
  auto input_dims = splitAsInt(layers[0], 'x');
  size_t nInputMaps = input_dims[0];
  SIZE img_size(input_dims[1], input_dims[2]);

  for (size_t i=1; i<layers.size(); ++i) {

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

      _transforms.push_back(new AffineTransform(fan_in, fan_out));

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

void NNet::read(const string &fn) {
  ifstream fin(fn.c_str());

  if (!fin.is_open())
    throw std::runtime_error(RED_ERROR + "Cannot load file: " + fn);

  clog << "\33[34m[Info]\33[0m Reading model from \33[32m" << fn << "\33[0m" << endl;

  stringstream ss;
  ss << fin.rdbuf() << '\0';
  fin.close();

  _transforms.clear();

  FeatureTransform* f;

  if (isXmlFormat(ss)) {
    rapidxml::xml_document<> doc;

    vector<char> buffer((istreambuf_iterator<char>(ss)), istreambuf_iterator<char>());
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
	case FeatureTransform::Tanh :
	  f = new Tanh;
	  break;
	case FeatureTransform::ReLU :
	  f = new ReLU;
	  break;
	case FeatureTransform::Softplus :
	  f = new Softplus;
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

  weight_initialize();
}

void NNet::weight_initialize() {

  const auto& t = _transforms;
  for (size_t i=0; i<t.size(); ++i) {

    // Test if i-th layer is affine transform ? Yes/No
    AffineTransform* p = dynamic_cast<AffineTransform*>(t[i]);

    // No.
    if (p == nullptr) continue;

    // Already initialized.
    if (p->get_w().size() != 0) continue;

    // Get fan-in and fan-out
    size_t in  = p->getInputDimension(),
	   out = p->getOutputDimension();

    // Use uniform random [0.5, 0.5] to initialize
    mat w = rand(out + 1, in + 1) - 0.5;

    // If there's an activation next to it, normalize it by multiplying a coeff
    if ( i + 1 < t.size() and dynamic_cast<Activation*>(t[i+1]) != nullptr ) {
      w *= GetNormalizedInitCoeff(
	  in, out, FeatureTransform::token2type(p->toString()) );
    }

    p->set_w(w);

    assert_nan(p->get_w());
  }
}

void NNet::save(const string &fn) const {
  ofstream fout(fn.c_str());

  if (!fout.is_open())
    throw std::runtime_error(RED_ERROR + "Cannot open file: " + fn);

  fout << *this;

  fout.close();
}

size_t NNet::getInputDimension() const { 
  return _transforms[0]->getInputDimension();
}

size_t NNet::getOutputDimension() const { 
  return _transforms.back()->getOutputDimension();
}

void NNet::status() const {

  const auto& t = _transforms;

  int nHiddens = 0;

  printf(".____._____________.___________.___________.___________._________.____________.\n");
  printf("|    |             |           |           |           |         |            |\n");
  printf("|    |  Transform  |   Input   |  Output   | Number of | Size of | Number of  |\n");
  printf("| No |             |           |           |           |         |            |\n");
  printf("|    |    Type     | Dimension | Dimension | Kernels   | Kernels | Parameters |\n");
  printf("|____|_____________|___________|___________|___________|_________|____________|\n");
  printf("|    |             |           |           |           |         |            |\n");

  for (size_t i=0; i<t.size(); ++i) {
    string type = t[i]->toString();
    size_t in  = t[i]->getInputDimension(),
	   out = t[i]->getOutputDimension();

    if (type == "Affine")
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

    printf("| %2lu | %-11s |  %6lu   |  %6lu   |  %7s  | %7s | %10s |\n",
	i, type.c_str(), in, out, kernel_number.c_str(), kernel_size.c_str(), nParamStr);
    printf("|    |             |           |           |           |         |            |\n");
  }

  printf("|____|_____________|___________|___________|___________|_________|____________|\n");

  nHiddens = std::max(0, nHiddens - 1);
  printf("Number of hidden layers: %2d \n", nHiddens);
}

bool NNet::is_cnn_dnn_boundary(size_t i) const {

  // the boundary between NNet and DNN must be:
  // a instance of MIMOFeatureTransform -> affine
  // and this affine transform must the first one to encounter after NNet.
  
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

void NNet::setDropout(bool flag) {
  auto& t = _transforms;
  for (size_t i=0; i<t.size(); ++i) {
    string type = t[i]->toString();

    auto ptr = dynamic_cast<Dropout*>(t[i]);
    if (ptr != nullptr)
      ptr->setDropout(flag);
  }
}

void NNet::setConfig(const Config& config) {
  _config = config;
}

Config NNet::getConfig() const {
  return _config;
}

ostream& operator << (ostream& os, const NNet& nnet) {
  for (size_t i=0; i<nnet._transforms.size(); ++i)
    os << nnet._transforms[i];
  return os;
}
