#include <feature-transform.h>
#define PAUSE { printf("Press Enter key to continue..."); fgetc(stdin); }

AffineTransform::AffineTransform(): _isOutputLayer(false) {}

AffineTransform::AffineTransform(const AffineTransform& source):
  _isOutputLayer(source._isOutputLayer),
  _w(source._w),
  _dw(source._dw) {

  }

AffineTransform::AffineTransform(const mat& w): _w(w), _dw(w.getRows(), w.getCols()), _isOutputLayer(false) {

}

AffineTransform::AffineTransform(size_t rows, size_t cols): _w(rows, cols), _dw(rows, cols), _isOutputLayer(false) {
  ext::randn(_w);
}

AffineTransform& AffineTransform::operator = (AffineTransform rhs) {
  swap(*this, rhs);
  return *this;
}

void AffineTransform::setOutputLayer(bool flag) {
  _isOutputLayer = flag;
}

mat& AffineTransform::getW() {
  return _w;
}

const mat& AffineTransform::getW() const {
  return _w;
}

mat& AffineTransform::getDw() {
  return _dw;
}

const mat& AffineTransform::getDw() const {
  return _dw;
}

void AffineTransform::update(float learning_rate) {
  _dw *= learning_rate;
  _w -= _dw;
}

void AffineTransform::resize(size_t rows, size_t cols) {
  _w.resize(rows, cols);
  _dw.resize(rows, cols);
}

string AffineTransform::toString() const {
  return "affinetransform";
}

void AffineTransform::feedForward(mat& fout, const mat& fin, size_t offset, size_t nData) {
  fout = ext::sigmoid(const_cast<mat&>(fin) * _w);
  fillLastColumnWith(fout, (float) 1.0);
}

void AffineTransform::backPropagate(const mat& fin, const mat& fout, mat& error) {

  mat delta = error & (1 - fout) & fout;

  _dw = ~const_cast<mat&>(fin) * delta;

  // Ignore last column, which is the bias
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
}

void swap(AffineTransform& lhs, AffineTransform& rhs) {
  std::swap(lhs._isOutputLayer, rhs._isOutputLayer);
  std::swap(lhs._w, rhs._w);
  std::swap(lhs._dw, rhs._dw);
}

// ===================
// ===== Softmax =====
// ===================

Softmax::Softmax(const mat& w): AffineTransform(w) {
}

Softmax::Softmax(size_t rows, size_t cols): AffineTransform(rows, cols) {
}

Softmax& Softmax::operator = (Softmax rhs) {
  AffineTransform::operator=(rhs);
  swap(*this, rhs);
  return *this;
}

string Softmax::toString() const {
  return "softmax";
}

vector<float> copyToHost(const mat& m);

void Softmax::feedForward(mat& fout, const mat& fin, size_t offset, size_t nData) {

  mat x = const_cast<mat&>(fin) * _w;
  x.resize(x.getRows(), x.getCols() - 1);

  // matlog(x);

  std::vector<float> hx = copyToHost(x);

  float* h_max = new float[x.getRows()];

  for (size_t i=0; i<x.getRows(); ++i) {
    float m = hx[i];
    for (size_t j=0; j<x.getCols(); ++j) {
      float v = hx[j * x.getRows() + i];
      if (v > m)
	m = v;
    }
    h_max[i] = m;
  }

  mat d_max = mat(h_max, x.getRows(), 1) * (mat(1, x.getCols()) += 1);
  delete [] h_max;
  x -= d_max;

  // matlog(d_max);
  // matlog(x);

  mat p(x.getRows(), x.getCols());

  thrust::device_ptr<float> xPtr(x.getData());
  thrust::device_ptr<float> pPtr(p.getData());
  thrust::transform(xPtr, xPtr + x.size(), pPtr, func::exp<float>());

  // matlog(p);

  mat sumOfProb = p * (mat(p.getCols(), p.getCols()) += 1);

  // matlog(sumOfProb);

  fout.resize(p.getRows(), p.getCols() + 1);
  thrust::device_ptr<float> foutPtr(fout.getData());
  thrust::device_ptr<float> sPtr(sumOfProb.getData());
  thrust::transform(pPtr, pPtr + p.size(), sPtr, foutPtr, thrust::divides<float>());

  // matlog(fout);

}

void Softmax::backPropagate(const mat& fin, const mat& fout, mat& error) {

  thrust::device_ptr<float> finPtr(fin.getData());
  thrust::device_ptr<float> foutPtr(fout.getData());
  thrust::device_ptr<float> ePtr(error.getData());

  mat T2(error.getRows(), error.getCols());
  thrust::device_ptr<float> T2Ptr(T2.getData());

  thrust::transform(ePtr, ePtr + error.size(), foutPtr, T2Ptr, thrust::multiplies<float>());

  mat sum = T2 * (mat(T2.getCols(), T2.getCols()) += 1);
  thrust::device_ptr<float> sPtr(sum.getData());

  mat delta(error.getRows(), error.getCols());
  thrust::device_ptr<float> dPtr(delta.getData());
  thrust::transform(ePtr, ePtr + error.size(), sPtr, dPtr, thrust::minus<float>());

  _dw = ~const_cast<mat&>(fin) * delta;

  // Ignore last column, which is the bias
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

}

void swap(Softmax& lhs, Softmax& rhs) {
}

