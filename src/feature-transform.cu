#include <feature-transform.h>

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

  /*
  size_t nData = delta.getRows();
  size_t D1 = _w.getRows() - 1;
  size_t D2 = delta.getCols() - 1;

  _dw = ~fin * delta;

  //   delta = delta(:, 1:end-1) * ~_w[i]
  //
  //                  (temp)
  //     delta'    =  delta    x     (weigth)^T
  // -------------------------------------------
  //       7                             7
  // |<--------->|   ----->|       |<--------->|
  // o o o o o o o = o o o o o x | o o o o o o o 
  // o o o o o o o   o o o o o   | o o o o o o o 
  // o o o o o o o   o o o o o   | o o o o o o o 
  //                             v o o o o o o o 
  //                               o o o o o o o  (<== bias, don't use them when back-propagate)

  mat tmp(delta);
  delta.resize(nData, D1 + 1);

  device_matrix<float>::cublas_gemm(
      CUBLAS_OP_N, CUBLAS_OP_T,
      nData, D1 + 1, D2, // Ignore last column, which is the bias
      1.0,
      tmp.getData(), nData,
      _w.getData(), D1 + 1,
      0.0,
      delta.getData(), nData);

  thrust::device_vector<float> temp(fin.size());

  thrust::device_ptr<float> output(fin.getData());
  thrust::transform(output, output + fin.size(), temp.begin(), func::dsigma<float>());

  thrust::device_ptr<float> dv1(delta.getData());
  thrust::transform(dv1, dv1 + delta.size(), temp.begin(), dv1, thrust::multiplies<float>());
  */

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

void Softmax::feedForward(mat& fout, const mat& fin, size_t offset, size_t nData) {

  mat x = const_cast<mat&>(fin) * _w;
  thrust::device_ptr<float> xPtr(x.getData());
  // mat p(x.getRows(), x.getCols());

  /*thrust::device_ptr<float> pPtr(p.getData());
  thrust::transform(xPtr, xPtr + x.size(), pPtr, func::exp<float>()); */

  // matlog(x);
  // matlog(p);




  mat sum = x * (mat(x.getCols(), x.getCols()) += 1);

  // matlog(sum);

  fout.resize(x.getRows(), x.getCols());
  thrust::device_ptr<float> foutPtr(fout.getData());
  thrust::device_ptr<float> sPtr(sum.getData());
  thrust::transform(xPtr, xPtr + x.size(), sPtr, foutPtr, thrust::divides<float>());

  // matlog(fout);

  /*vector<float> hm(fout.size());
  thrust::device_ptr<float> dPtr(fout.getData());
  thrust::copy(dPtr, dPtr + fout.size(), hm.begin());
  for (size_t i=0; i<hm.size(); ++i)
    assert(hm[i] >= 0);*/

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

