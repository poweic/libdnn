#include <feature-transform.h>

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

  const int N = 32;
  dim3 grid;
  grid.x = (unsigned int) ceil((float) x.getCols() / N);
  grid.y = (unsigned int) ceil((float) x.getRows() / N);
  dim3 threads(N, N);

  substract_max_per_row<<<grid, threads>>>(x.getData(), rmax.getData(), x.getRows(), x.getCols());
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

string toString(std::vector<float> data, size_t rows, size_t cols) {
  stringstream ss;
  ss << " [";

  for (size_t j=0; j<rows-1; ++j) {
    ss << "\n  ";
    for (size_t k=0; k<cols; ++k)
      ss << data[k * rows + j] << " ";
  }
  ss << "]\n";

  ss << "<bias> \n [";
  for (size_t j=0; j<cols; ++j)
    ss << data[j * rows + rows - 1] << " ";
  ss << " ]\n";

  return ss.str();
}

// ============================
// ===== FeatureTransform =====
// ============================

FeatureTransform::FeatureTransform(const FeatureTransform& source): _w(source._w), _dw(source._dw) {
}

FeatureTransform::FeatureTransform(const mat& w): _w(w), _dw(w.getRows(), w.getCols()) {
}

FeatureTransform::FeatureTransform(size_t rows, size_t cols, float variance): _w(rows, cols), _dw(rows, cols) {
  ext::randn(_w, 0.0f, variance);
}

mat& FeatureTransform::getW() {
  return _w;
}

const mat& FeatureTransform::getW() const {
  return _w;
}

mat& FeatureTransform::getDw() {
  return _dw;
}

const mat& FeatureTransform::getDw() const {
  return _dw;
}

void FeatureTransform::update(float learning_rate) {
  _dw *= learning_rate;
  _w -= _dw;
}

// ===================
// ===== Sigmoid =====
// ===================

Sigmoid::Sigmoid(const mat& w): FeatureTransform(w) {
}

Sigmoid::Sigmoid(size_t rows, size_t cols, float variance): FeatureTransform(rows, cols, variance) {
}

Sigmoid::Sigmoid(const Sigmoid& src): FeatureTransform(src) {
}

Sigmoid* Sigmoid::clone() const {
  return new Sigmoid(*this);
}

string Sigmoid::toString() const {
  size_t rows = _w.getRows(),
	 cols = _w.getCols() - 1;

  stringstream ss;
  ss << "<sigmoid> " << rows - 1 << " " << cols << endl;
  ss << ::toString(copyToHost(_w), rows, cols);
  return ss.str();
}

void Sigmoid::feedForward(mat& fout, const mat& fin) {
  fout = ext::sigmoid(const_cast<mat&>(fin) * _w);
  fillLastColumnWith(fout, (float) 1.0);
}

void Sigmoid::backPropagate(mat& error, const mat& fin, const mat& fout) {
  mat delta = error & (1.0f - fout) & fout;

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

// ===================
// ===== Softmax =====
// ===================

Softmax::Softmax(const mat& w): FeatureTransform(w) {
}

Softmax::Softmax(size_t rows, size_t cols, float variance): FeatureTransform(rows, cols, variance) {
}

Softmax::Softmax(const Softmax& src): FeatureTransform(src) {
}

Softmax* Softmax::clone() const {
  return new Softmax(*this);
}

string Softmax::toString() const {
  size_t rows = _w.getRows(),
	 cols = _w.getCols() - 1;

  stringstream ss;
  ss << "<softmax> " << rows - 1 << " " << cols << endl;
  ss << ::toString(copyToHost(_w), rows, cols);
  return ss.str();
}

/*__global__ void substract_max_per_row(float* const A, unsigned int rows, unsigned int cols) {
  extern __shared__ float sdata[];

  // Matrix index
  int ty = threadIdx.y;
  int tx = threadIdx.x;
  int x = threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x >= cols || y >= rows)
    return;

  unsigned int idx = x * blockDim.y + ty;
  sdata[idx] = A[x * rows + y];

  for (unsigned int s = blockDim.x/2 ; s > 0; s >>= 1) {
    if (x >= s || x + s >= cols)
      continue;

    if (sdata[(x + s) * blockDim.y + ty] > sdata[idx])
      sdata[idx] = sdata[(x + s) * blockDim.y + ty];

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
}*/

void Softmax::feedForward(mat& fout, const mat& fin) {

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
}

mat rowSum(mat& m) {
  return m * (mat(m.getCols(), m.getCols()) += 1);
}

void Softmax::backPropagate(mat& error, const mat& fin, const mat& fout) {

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
