#include <dnn-utility.h>

map<int, int> getLabelMapping(const hmat& labels) {
  map<int, int> classes;
  for (size_t i=0; i<labels.size(); ++i)
    classes[(int) labels[i]] = 1;

  int counter = 0;
  map<int, int>::iterator itr = classes.begin();
  for (; itr != classes.end(); ++itr)
    itr->second = ++counter;

  return classes;
}

namespace ext {

  void rescale(mat& data, float lower, float upper) {
    float min = ext::min(data);
    float max = ext::max(data);

    float ratio = (upper - lower) / (max - min);
    data = (data - min) * ratio + lower;
  }

  float max(const mat& v) {
    thrust::device_ptr<float> vPtr(v.getData());
    thrust::device_ptr<float> maxPtr = thrust::max_element(vPtr, vPtr + v.size());
    thrust::host_vector<float> hMaxPtr(maxPtr, maxPtr + 1);
    return hMaxPtr[0];
  }

  float min(const mat& v) {
    thrust::device_ptr<float> vPtr(v.getData());
    thrust::device_ptr<float> minPtr = thrust::min_element(vPtr, vPtr + v.size());
    thrust::host_vector<float> hMaxPtr(minPtr, minPtr + 1);
    return hMaxPtr[0];
  }

  float max(const hmat& v) {
    float* m = thrust::max_element(v.getData(), v.getData() + v.size());
    return *m;
  }

  float min(const hmat& v) {
    float* m = thrust::min_element(v.getData(), v.getData() + v.size());
    return *m;
  }
};

__global__ void dcrossentropy_kernel(float* error, float* const target, float* const output, unsigned int rows, unsigned int cols) {
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // Matrix index
  int x = blockIdx.x*blockDim.x + tx;
  int y = blockIdx.y*blockDim.y + ty;

  if (x >= cols || y >= rows)
    return;

  int i = x * rows + y;

  float o = output[i];
  if (o < 1e-10)
    o = 1e-10;

  error[i] = (target[y] - 1 == x) ? (-1.0 / o) : 0; 
  __syncthreads();
}

void dCrossEntropy(mat& error, const mat &target, const mat& output) {

  const size_t N = 32;
  dim3 threads(N, N);
  dim3 grid;
  grid.x = (unsigned int) ceil((float) output.getCols() / N);
  grid.y = (unsigned int) ceil((float) output.getRows() / N);

  dcrossentropy_kernel<<< grid, threads >>>(error.getData(), target.getData(), output.getData(), output.getRows(), output.getCols());

  CCE(cudaDeviceSynchronize());
}

mat getError(const mat& target, const mat& output, ERROR_MEASURE errorMeasure) {

  mat error;

  switch (errorMeasure) {
    case L2ERROR: 
      // FIXME
      /*error = output - target;
      error.reserve(error.getRows() * (error.getCols() + 1));
      error.resize(error.getRows(), error.getCols() + 1);*/

      break;
    case CROSS_ENTROPY: {

	size_t output_dim = output.getCols();

	error.resize(output.getRows(), output.getCols() + 1);

	dCrossEntropy(error, target, output);

	/* Legacy Code
	thrust::device_ptr<float> oPtr(output.getData());
	thrust::device_vector<float> TMP(output.size());
	thrust::transform(oPtr, oPtr + output.size(), TMP.begin(), func::min_threshold<float>(1e-10));

	thrust::device_ptr<float> pPtr(target.getData());
	thrust::device_ptr<float> ePtr(error.getData());
	thrust::transform(pPtr, pPtr + target.size(), TMP.begin(), ePtr, func::dcrossentropy<float>());
	*/

	break;
      }

    default:
      break;
  }

  return error;
}

mat posteriorProb2Label(const mat& prob) {

  assert(prob.getCols() > 1);

  size_t rows = prob.getRows(),
	 cols = prob.getCols();

  float* h_prob = new float[prob.size()];
  float* h_labels  = new float[rows];
  CCE(cudaMemcpy(h_prob, prob.getData(), sizeof(float) * prob.size(), cudaMemcpyDeviceToHost));

  for (size_t i=0; i<rows; ++i) {

    float max = -1e10;
    size_t maxIdx = 0;

    for (size_t j=0; j<cols; ++j) {
      if (h_prob[j * rows + i] > max) {
	max = h_prob[j * rows + i];
	maxIdx = j;
      }
    }

    h_labels[i] = maxIdx + 1;
  }

  mat labels(h_labels, rows, 1);

  delete [] h_prob;
  delete [] h_labels;

  return labels;
}

vector<float> copyToHost(const mat& m) {
  vector<float> hm(m.size());
  thrust::device_ptr<float> dPtr(m.getData());
  thrust::copy(dPtr, dPtr + m.size(), hm.begin());
  return hm;
}

size_t countDifference(const mat& m1, const mat& m2) {
  assert(m1.size() == m2.size());

  size_t L = m1.size();
  thrust::device_ptr<float> ptr1(m1.getData());
  thrust::device_ptr<float> ptr2(m2.getData());

  size_t nDiff = thrust::inner_product(ptr1, ptr1 + L, ptr2, 0.0, thrust::plus<float>(), thrust::not_equal_to<float>());
  return nDiff;
}


size_t zeroOneError(const mat& prob, const mat& label, ERROR_MEASURE errorMeasure) {
  assert(prob.getRows() == label.getRows());
  assert(label.getCols() == 1);

  size_t nError = 0;

  if (errorMeasure == L2ERROR) {
    // nError = countDifference(label, prob);
  }
  else {
    mat L = posteriorProb2Label(prob);
    nError = countDifference(L, label);
  }

  return nError;
}
