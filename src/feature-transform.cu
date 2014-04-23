#include <feature-transform.h>

void FeatureTransform::print(FILE* fid, const host_matrix<float>& data, string type) {
  fprintf(fid, "<%s> %lu %lu\n", type.c_str(), data.getRows() - 1, data.getCols() - 1);

  size_t rows = data.getRows(),
	 cols = data.getCols();

  fprintf(fid, " [");

  for (size_t j=0; j<rows-1; ++j) {
    fprintf(fid, "\n  ");
    for (size_t k=0; k<cols-1; ++k)
      fprintf(fid, "%g ", data[k * rows + j]);
  }
  fprintf(fid, "]\n");

  fprintf(fid, "<bias> \n [");
  for (size_t j=0; j<cols-1; ++j)
    fprintf(fid, "%g ", data[j * rows + rows - 1]);
  fprintf(fid, " ]\n");
}

mat rowSum(mat& m) {
  return m * mat(m.getCols(), m.getCols(), 1);
}

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T>
{
  T C; // number of columns

  __host__ __device__
    linear_index_to_row_index(T C) : C(C) {}

  __host__ __device__
    T operator()(T i)
    {
      return i / C;
    }
};

void substractMaxPerRow(mat& x);
mat getRowMax(mat& A);
__global__ void substract_max_per_row(float* const A, float* const rmax, unsigned int rows, unsigned int cols);

void substractMaxPerRow(mat& x) {
  mat rmax = getRowMax(x);

  ALLOCATE_GRIDS_AND_THREADS(x.getRows(), x.getCols());
  substract_max_per_row<<< grids, threads >>>(x.getData(), rmax.getData(), x.getRows(), x.getCols());
  CCE(cudaDeviceSynchronize());
}

__global__ void substract_max_per_row(float* const A, float* const rmax, unsigned int rows, unsigned int cols) {
  // Matrix index
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  A[x * rows + y] -= rmax[y];
}

mat getRowMax(mat& A) {
  mat rmax(A.getRows(), 1);
  mat At = ~A;

  // allocate storage for per-row results and indices
  thrust::device_vector< float > row_indices(A.getRows());
  thrust::device_vector< float > row_results(A.getRows());

  // compute row sums by summing values with equal row indices
  thrust::reduce_by_key
    (thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(A.getCols())),
     thrust::make_transform_iterator(thrust::counting_iterator<int>(0), linear_index_to_row_index<int>(A.getCols())) + A.size(),
     thrust::device_ptr<float>(At.getData()),
     row_indices.begin(),
     thrust::device_ptr<float>(rmax.getData()),
     thrust::equal_to<float>(),
     thrust::maximum<float>());

  return rmax;
}

// ============================
// ===== FeatureTransform =====
// ============================

FeatureTransform::FeatureTransform(const FeatureTransform& source): _w(source._w) {
}

FeatureTransform::FeatureTransform(const mat& w): _w(w){
}

size_t FeatureTransform::getInputDimension() const {
  return _w.getRows();
}

size_t FeatureTransform::getOutputDimension() const {
  return _w.getCols();
}

void FeatureTransform::print(FILE* fid) const {
  FeatureTransform::print(fid, this->_w, this->toString());
}

void FeatureTransform::feedBackward(mat& error, const mat& delta) {
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
}

// ===================
// ===== Sigmoid =====
// ===================

Sigmoid::Sigmoid(const mat& w): FeatureTransform(w) {
}

Sigmoid::Sigmoid(const Sigmoid& src): FeatureTransform(src) {
}

Sigmoid* Sigmoid::clone() const {
  return new Sigmoid(*this);
}

string Sigmoid::toString() const {
  return "sigmoid";
}

void Sigmoid::feedForward(mat& fout, const mat& fin) {
  // fout = sigmoid(fin * _w);
  fout = transform(fin * _w, func::sigmoid<float>());
  fillLastColumnWith(fout, (float) 1.0);
}

void Sigmoid::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {
  mat delta = error & (1.0f - fout) & fout;
  this->feedBackward(error, delta);
  gemm(fin, delta, _w, -learning_rate, 1.0f, true, false);
}

// ===================
// ===== Softmax =====
// ===================

Softmax::Softmax(const mat& w): FeatureTransform(w) {
}

Softmax::Softmax(const Softmax& src): FeatureTransform(src) {
}

Softmax* Softmax::clone() const {
  return new Softmax(*this);
}

string Softmax::toString() const {
  return "softmax";
}

void Softmax::feedForward(mat& fout, const mat& fin) {

  mat x = fin * _w;
  x.resize(x.getRows(), x.getCols() - 1);
  substractMaxPerRow(x);

  mat p(x.getRows(), x.getCols());

  thrust::device_ptr<float> xPtr(x.getData());
  thrust::device_ptr<float> pPtr(p.getData());
  thrust::transform(xPtr, xPtr + x.size(), pPtr, func::exp<float>());

  mat sumOfProb = p * mat(p.getCols(), p.getCols(), 1);

  fout.resize(p.getRows(), p.getCols() + 1);
  thrust::device_ptr<float> foutPtr(fout.getData());
  thrust::device_ptr<float> sPtr(sumOfProb.getData());
  thrust::transform(pPtr, pPtr + p.size(), sPtr, foutPtr, thrust::divides<float>());
}

void Softmax::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {

  // This is much faster and easier
  mat delta = error;
  this->feedBackward(error, delta);
  gemm(fin, delta, _w, -learning_rate, 1.0f, true, false);

  // cf. /usr/local/lib/python2.7/dist-packages/theano/tensor/nnet/nnet.py:251
  /*mat error_times_fout = error & fout;
  mat delta = error_times_fout - (rowSum(error_times_fout) & fout);

  this->feedBackward(error, delta);

  gemm(fin, delta, _w, -learning_rate, 1.0f, true, false);*/
}
