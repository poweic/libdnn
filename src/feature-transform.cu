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

__global__ void substract_max_per_row(float* const A, unsigned int rows, unsigned int cols) {
  extern __shared__ float sdata[];

  // Matrix index
  int ty = threadIdx.y;
  int x = threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  sdata[x * blockDim.y + ty] = A[x * rows + y];

  for (unsigned int s = blockDim.x/2 ; s > 0; s >>= 1) {
    if (x < s && x + s < cols) {
      if (sdata[(x + s) * blockDim.y + ty] > sdata[x * blockDim.y + ty])
	sdata[x * blockDim.y + ty] = sdata[(x + s) * blockDim.y + ty];
    }
    __syncthreads();
  }

  A[x * rows + y] -= sdata[ty];
}

void substractMaxPerRow(mat& x) {
  size_t rows = x.getRows(),
	 cols = x.getCols();

  const size_t N = 32;
  assert(cols <= N);

  dim3 grid;
  grid.x = 1;
  grid.y = (unsigned int) ceil((float) rows / N);
  dim3 threads(N, N);

  size_t smSize = N * N * sizeof(float);

  substract_max_per_row<<< grid, threads, smSize >>>(x.getData(), rows, cols);
  CCE(cudaDeviceSynchronize());
}


void Softmax::feedForward(mat& fout, const mat& fin, size_t offset, size_t nData) {

  mat x = const_cast<mat&>(fin) * const_cast<mat&>(_w);
  x.resize(x.getRows(), x.getCols() - 1);
  substractMaxPerRow(x);

  mat p(x.getRows(), x.getCols());

  thrust::device_ptr<float> xPtr(x.getData());
  thrust::device_ptr<float> pPtr(p.getData());
  thrust::transform(xPtr, xPtr + x.size(), pPtr, func::exp<float>());

  mat sumOfProb = p * (mat(p.getCols(), p.getCols()) += 1);

  fout.resize(p.getRows(), p.getCols() + 1);
  thrust::device_ptr<float> foutPtr(fout.getData());
  thrust::device_ptr<float> sPtr(sumOfProb.getData());
  thrust::transform(pPtr, pPtr + p.size(), sPtr, foutPtr, thrust::divides<float>());

  /*
  mat x = const_cast<mat&>(fin) * _w;
  x.resize(x.getRows(), x.getCols() - 1);

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

  mat p(x.getRows(), x.getCols());

  thrust::device_ptr<float> xPtr(x.getData());
  thrust::device_ptr<float> pPtr(p.getData());
  thrust::transform(xPtr, xPtr + x.size(), pPtr, func::exp<float>());

  mat sumOfProb = p * (mat(p.getCols(), p.getCols()) += 1);

  fout.resize(p.getRows(), p.getCols() + 1);
  thrust::device_ptr<float> foutPtr(fout.getData());
  thrust::device_ptr<float> sPtr(sumOfProb.getData());
  thrust::transform(pPtr, pPtr + p.size(), sPtr, foutPtr, thrust::divides<float>());
  */
}

mat rowSum(mat& m) {
  return m * (mat(m.getCols(), m.getCols()) += 1);
}

void Softmax::backPropagate(const mat& fin, const mat& fout, mat& error) {

  mat error_times_fout = error & fout;
  mat sum = rowSum(error_times_fout);

  mat sum_times_fout = sum & fout;
  mat delta = error_times_fout - sum_times_fout;

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

