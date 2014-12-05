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

  FeatureTransform::Type type;

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

AffineTransform* AffineTransform::clone() const {
  return new AffineTransform(*this);
}

string AffineTransform::toString() const {
  return "Affine";
}

void AffineTransform::feedForward(mat& fout, const mat& fin) {
  fout = fin * _w;
  fillLastColumnWith(fout, (float) 1.0);
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

  gemm(fin, delta, _w, -learning_rate, 1.0f, true, false);
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
  fillLastColumnWith(fout, (float) 1.0);
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
