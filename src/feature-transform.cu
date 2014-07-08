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

// convert a linear index to a row index
template <typename T>
struct linear_index_to_row_index : public thrust::unary_function<T,T> {
  T cols; // number of columns

  __host__ __device__ linear_index_to_row_index(T cols) : cols(cols) {}

  __host__ __device__ T operator()(T i) { return i / cols; }
};


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

__global__ void substract_max_per_row(float* const A, float* const rmax, unsigned int rows, unsigned int cols) {
  // Matrix index
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  A[x * rows + y] -= rmax[y];
}

void substractMaxPerRow(mat& x) {
  mat rmax = getRowMax(x);

  ALLOCATE_GRIDS_AND_THREADS(x.getRows(), x.getCols());
  substract_max_per_row<<< grids, threads >>>(x.getData(), rmax.getData(), x.getRows(), x.getCols());
  CCE(cudaDeviceSynchronize());
}


/*
 * class FeatureTransform
 *
 * */

FeatureTransform::FeatureTransform(size_t input_dim, size_t output_dim)
  : _input_dim(input_dim), _output_dim(output_dim) {
}

FeatureTransform::FeatureTransform(const FeatureTransform& source) {
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

AffineTransform::AffineTransform(const AffineTransform& src): FeatureTransform(src) {
}

AffineTransform::AffineTransform(FILE* fid) {
  this->read(fid);
}

void AffineTransform::read(FILE* fid) {
  size_t rows, cols;
  if ( fscanf(fid, "%lu %lu", &rows, &cols) == EOF)
    throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");

  hmat hw(rows + 1, cols + 1);

  // Read matrix
  fscanf(fid, " [\n");
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
  fprintf(fid, "<%s> %lu %lu\n", this->toString().c_str()
      , data.getRows() - 1, data.getCols() - 1);

  size_t rows = data.getRows(),
	 cols = data.getCols();

  fprintf(fid, "[");

  for (size_t j=0; j<rows-1; ++j) {
    fprintf(fid, "\n  ");
    for (size_t k=0; k<cols-1; ++k)
      fprintf(fid, "%g ", data[k * rows + j]);
  }
  fprintf(fid, "]\n");

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
 * class Sigmoid
 *
 * */

Sigmoid::Sigmoid(size_t input_dim, size_t output_dim)
  : FeatureTransform(input_dim, output_dim) {

}

Sigmoid::Sigmoid(const Sigmoid& src): FeatureTransform(src) {
}

Sigmoid::Sigmoid(FILE* fid) {
  this->read(fid);
}

void Sigmoid::read(FILE* fid) {
  if ( fscanf(fid, "%lu %lu\n [\n", &_input_dim, &_output_dim) == EOF)
    throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");

  if (_input_dim != _output_dim)
    throw std::runtime_error("\33[31m[Error]\33[0m Mismatched input/output dimension");
}

void Sigmoid::write(FILE* fid) const {
  fprintf(fid, "<%s> %lu %lu\n", this->toString().c_str(), _input_dim, _output_dim);
}

Sigmoid* Sigmoid::clone() const {
  return new Sigmoid(*this);
}

string Sigmoid::toString() const {
  return "Sigmoid";
}

void Sigmoid::feedForward(mat& fout, const mat& fin) {
  fout = transform(fin, func::sigmoid<float>());
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
  : FeatureTransform(input_dim, output_dim) {

}

Softmax::Softmax(const Softmax& src): FeatureTransform(src) {
}

Softmax::Softmax(FILE* fid) {
  this->read(fid);
}

void Softmax::read(FILE* fid) {
  if ( fscanf(fid, "%lu %lu\n [\n", &_input_dim, &_output_dim) == EOF)
    throw std::runtime_error("\33[31m[Error]\33[0m failed when reading");

  if (_input_dim != _output_dim)
    throw std::runtime_error("\33[31m[Error]\33[0m Mismatched input/output dimension");
}

void Softmax::write(FILE* fid) const {
  fprintf(fid, "<%s> %lu %lu\n", this->toString().c_str(), _input_dim, _output_dim);
}

Softmax* Softmax::clone() const {
  return new Softmax(*this);
}

string Softmax::toString() const {
  return "Softmax";
}

void Softmax::feedForward(mat& fout, const mat& fin) {

  mat x = fin;
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
  // Do nothing.
}
