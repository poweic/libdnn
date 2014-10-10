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

#include <dnn.h>
#include "tools/rapidxml-1.13/rapidxml_utils.hpp"
#include "tools/rapidxml-1.13/rapidxml_print.hpp"

DNN::DNN(): _transforms(), _config() {}

DNN::DNN(string fn): _transforms(), _config() {
  this->read(fn);
}

DNN::DNN(const Config& config): _transforms(), _config(config) {
}

DNN::DNN(const DNN& source): _transforms(source._transforms.size()), _config() {
  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i] = source._transforms[i]->clone();
}

void DNN::init(const std::vector<mat>& weights) {
  throw std::runtime_error("\33[31m[Error]\33[0m Not implemented yet!!");
  /*_transforms.resize(weights.size());

  for (size_t i=0; i<_transforms.size() - 1; ++i)
      _transforms[i] = new Sigmoid(weights[i]);
  _transforms.back() = new Softmax(weights.back());*/
}

DNN::~DNN() {
  for (size_t i=0; i<_transforms.size(); ++i)
    delete _transforms[i];
}

DNN& DNN::operator = (DNN rhs) {
  swap(*this, rhs);
  return *this;
}
  
void DNN::setConfig(const Config& config) {
  _config = config;
}

size_t DNN::getNLayer() const {
  return _transforms.size() + 1;
}

void DNN::status() const {
  
  const auto& t = _transforms;

  size_t nAffines=0;
  for (size_t i=0; i<t.size(); ++i)
    nAffines += (t[i]->toString() == "AffineTransform");

  printf("\33[33m[INFO]\33[0m # of hidden layers: %2lu \n", nAffines == 0 ? 0 : nAffines - 1);

  for (size_t i=0; i<t.size(); ++i) {
    printf("  %-16s %4lu x %4lu [%-2lu]\n", t[i]->toString().c_str(),
	t[i]->getInputDimension(), t[i]->getOutputDimension(), i);
  }

}

void DNN::read(string fn) {

  ifstream fin(fn.c_str());

  if (!fin.is_open())
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + fn);

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

      auto x = node->first_attribute("type");

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
	case FeatureTransform::Convolution : 
	  break;
	case FeatureTransform::SubSample :
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
  else {
    clog << "\33[33m[Warning]\33[0m The original model format is \33[36mdeprecated\33[0m. "
      << "Please use XML format." << endl;
    while ( ss >> f )
      _transforms.push_back(f);
  }
}

void DNN::save(string fn) const {
  ofstream fout(fn.c_str());

  if (!fout.is_open())
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot open file: " + fn);

  for (size_t i=0; i<_transforms.size(); ++i)
    fout << _transforms[i];
  
  fout.close();
}

std::vector<FeatureTransform*>& DNN::getTransforms() {
  return _transforms;
}

const std::vector<FeatureTransform*>& DNN::getTransforms() const {
  return _transforms;
}

// ========================
// ===== Feed Forward =====
// ========================

void DNN::adjustLearningRate(float trainAcc) {
  static size_t phase = 0;

  if ( (trainAcc > 0.80 && phase == 0) ||
       (trainAcc > 0.85 && phase == 1) ||
       (trainAcc > 0.90 && phase == 2) ||
       (trainAcc > 0.92 && phase == 3) ||
       (trainAcc > 0.95 && phase == 4) ||
       (trainAcc > 0.97 && phase == 5)
     ) {

    float ratio = 0.9;
    printf("\33[33m[Info]\33[0m Adjust learning rate from \33[32m%.7f\33[0m to \33[32m%.7f\33[0m\n", _config.learningRate, _config.learningRate * ratio);
    _config.learningRate *= ratio;
    ++phase;
  }
}

mat DNN::feedForward(const mat& fin) const {

  mat y;

  _transforms[0]->feedForward(y, fin);

  for (size_t i=1; i<_transforms.size(); ++i)
    _transforms[i]->feedForward(y, y);

  y.resize(y.getRows(), y.getCols() - 1);

  return y;
}

void DNN::feedForward(mat& output, const mat& fin) {

  // FIXME This should be an ASSERTION, not resizing.
  if (_houts.size() != this->getNLayer() - 2)
    _houts.resize(this->getNLayer() - 2);

  if (_houts.size() > 0) {
    _transforms[0]->feedForward(_houts[0], fin);

    for (size_t i=1; i<_transforms.size()-1; ++i)
      _transforms[i]->feedForward(_houts[i], _houts[i-1]);

    _transforms.back()->feedForward(output, _houts.back());
  }
  else {
    _transforms.back()->feedForward(output, fin);
  }

  output.resize(output.getRows(), output.getCols() - 1);
}

// ============================
// ===== Back Propagation =====
// ============================

void DNN::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {

  mat output(fout);
  output.reserve(output.size() + output.getRows());
  output.resize(output.getRows(), output.getCols() + 1);

  error.reserve(error.size() + error.getRows());
  error.resize(error.getRows(), error.getCols() + 1);

  assert(error.getRows() == output.getRows() && error.getCols() == output.getCols());

  if (_houts.size() > 0) {
    _transforms.back()->backPropagate(error, _houts.back(), output, learning_rate);

    for (int i=_transforms.size() - 2; i >= 1; --i)
      _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], learning_rate);

    _transforms[0]->backPropagate(error, fin, _houts[0], learning_rate);
  }
  else
    _transforms.back()->backPropagate(error, fin, output, learning_rate);
}

Config DNN::getConfig() const {
  return _config;
}

void swap(DNN& lhs, DNN& rhs) {
  using std::swap;
  swap(lhs._transforms, rhs._transforms);
  swap(lhs._config, rhs._config);
}

// =============================
// ===== Utility Functions =====
// =============================

/*mat l2error(mat& targets, mat& predicts) {
  mat err(targets - predicts);

  thrust::device_ptr<float> ptr(err.getData());
  thrust::transform(ptr, ptr + err.size(), ptr, func::square<float>());

  mat sum_matrix(err.getCols(), 1);
  err *= sum_matrix;
  
  return err;
}
*/
