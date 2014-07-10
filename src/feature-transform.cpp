#include <feature-transform.h>

FeatureTransform* FeatureTransform::create(FILE* fid) {
  char c_type[128];
  if ( fscanf(fid, "%s", c_type) == EOF )
    return NULL;

  string type(c_type);

  if (type == "<AffineTransform>")
    return new AffineTransform(fid);
  else if (type == "<Sigmoid>")
    return new Sigmoid(fid);
  else if (type == "<Softmax>")
    return new Softmax(fid);

  return NULL;
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

AffineTransform::AffineTransform(FILE* fid) {
  this->read(fid);
}

#pragma GCC diagnostic ignored "-Wunused-result"
void AffineTransform::read(FILE* fid) {

  size_t rows, cols;
  if ( fscanf(fid, "%lu %lu\n", &rows, &cols) == EOF)
    throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");

  hmat hw(rows + 1, cols + 1);

  // Read matrix
  fscanf(fid, "[\n");
  for (size_t i=0; i<rows; ++i)
    for (size_t j=0; j<cols; ++j)
      fscanf(fid, "%f ", &hw(i, j) );
  fscanf(fid, "]\n");

  // Read vector (bias)
  fscanf(fid, "[");
  for (size_t j=0; j<cols; ++j)
    fscanf(fid, "%f ", &hw(rows, j) );
  fscanf(fid, "]\n");

  _w = (mat) hw;
  _input_dim = _w.getRows();
  _output_dim = _w.getCols();
}

void AffineTransform::write(FILE* fid) const {

  hmat data(_w);

  size_t rows = data.getRows(),
	 cols = data.getCols();

  fprintf(fid, "<%s> %lu %lu\n", this->toString().c_str() , rows - 1, cols - 1);

  // Write matrix
  fprintf(fid, "[");
  for (size_t j=0; j<rows-1; ++j) {
    fprintf(fid, "\n  ");
    for (size_t k=0; k<cols-1; ++k)
      fprintf(fid, "%g ", data[k * rows + j]);
  }
  fprintf(fid, "]\n");

  // Write vector (bias)
  fprintf(fid, "[ ");
  for (size_t j=0; j<cols-1; ++j)
    fprintf(fid, "%g ", data[j * rows + rows - 1]);
  fprintf(fid, "]\n");
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

  // The last row of _w is bias, and the last column of _w is saved only for computational efficiency.
  // Therefore, ignore last column, which is the bias.
  size_t traceLength = delta.getCols() - 1;

  error.resize(delta.getRows(), _w.getRows());

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

/*
 * class Activation
 *
 * */

Activation::Activation() {
}

Activation::Activation(size_t input_dim, size_t output_dim)
  : FeatureTransform(input_dim, output_dim) {

}

void Activation::read(FILE* fid) {
  if ( fscanf(fid, "%lu %lu\n [\n", &_input_dim, &_output_dim) == EOF)
    throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");

  if (_input_dim != _output_dim)
    throw std::runtime_error("\33[31m[Error]\33[0m Mismatched input/output dimension");
}

void Activation::write(FILE* fid) const {
  fprintf(fid, "<%s> %lu %lu\n", this->toString().c_str(), _input_dim, _output_dim);
}

/*
 * class Sigmoid
 *
 * */

Sigmoid::Sigmoid(size_t input_dim, size_t output_dim)
  : Activation(input_dim, output_dim) {

}

Sigmoid::Sigmoid(FILE* fid) {
  this->read(fid);
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

Softmax::Softmax(FILE* fid) {
  this->read(fid);
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
