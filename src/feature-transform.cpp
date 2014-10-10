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

#define PARSE_ASSERT(x) { if (!(x)) \
  throw std::runtime_error("\33[31m[Error]\33[0m failed when reading"); }

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
  else 
    ft = NULL;

  return is;
}

/*
 * class FeatureTransform
 *
 * */

FeatureTransform::FeatureTransform(size_t input_dim, size_t output_dim)
  : _input_dim(input_dim), _output_dim(output_dim) {
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
 
void AffineTransform::read(istream& is) {

  string dummy;

  size_t rows, cols;
  PARSE_ASSERT(is >> rows);
  PARSE_ASSERT(is >> cols);

  hmat hw(rows + 1, cols + 1);

  // Read matrix
  PARSE_ASSERT(is >> dummy);
  for (size_t i=0; i<rows; ++i)
    for (size_t j=0; j<cols; ++j)
      PARSE_ASSERT( is >> hw(i, j) );
  PARSE_ASSERT(is >> dummy);

  // Read vector (bias)
  PARSE_ASSERT(is >> dummy);
  for (size_t j=0; j<cols; ++j)
    PARSE_ASSERT( is >> hw(rows, j) );
  PARSE_ASSERT(is >> dummy);

  _w = (mat) hw;
  _input_dim = _w.getRows();
  _output_dim = _w.getCols();
}

void AffineTransform::write(ostream& os) const {

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
}

AffineTransform* AffineTransform::clone() const {
  return new AffineTransform(*this);
}

string AffineTransform::toString() const {
  return "AffineTransform";
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

void Activation::read(istream& is) {

  bool success;
  success = is >> _input_dim;
  if (!success) throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");
  success = is >> _output_dim;
  if (!success) throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");

  string remaining;
  success = std::getline(is, remaining);
  if (!success) throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");
  
  if (_input_dim != _output_dim)
    throw std::runtime_error("\33[31m[Error]\33[0m Mismatched input/output dimension");
}

void Activation::write(ostream& os) const {
  os << "<" << this->toString() << "> " << _input_dim << " " << _output_dim << endl;
}

/*
 * class Sigmoid
 *
 * */

Sigmoid::Sigmoid(size_t input_dim, size_t output_dim)
  : Activation(input_dim, output_dim) {

}

Sigmoid::Sigmoid(istream& is) {
  this->read(is);
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
  this->read(is);
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
