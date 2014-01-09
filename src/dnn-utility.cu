#include <dnn-utility.h>

map<int, int> getLabelMapping(const mat& labels) {
  thrust::device_ptr<float> dptr(labels.getData());
  thrust::host_vector<float> y(dptr, dptr + labels.size());

  map<int, int> classes;
  for (size_t i=0; i<y.size(); ++i)
    classes[(int) y[i]] = 1;

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

};

mat label2PosteriorProb(const mat& labels) {

  map<int, int> classes = getLabelMapping(labels);
  size_t nClasses = classes.size();
  size_t nData = labels.getRows();

  // Convert labels to posterior probabilities
  float* h_prob = new float[nData * nClasses];
  memset(h_prob, 0, sizeof(float) * nData * nClasses);

  vector<float> hy = copyToHost(labels);
  for (size_t i=0; i<nData; ++i)
    h_prob[(size_t) (hy[i] - 1) * nData + i] = 1;

  mat probs(h_prob, nData, nClasses);

  delete [] h_prob;

  return probs;
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

    // matlog(prob); matlog(L); PAUSE;
  }

  return nError;
}

mat& calcError(const mat& output, const mat& trainY, size_t offset, size_t nData) {

  mat error(nData, trainY.getCols());

  device_matrix<float>::cublas_geam(
      CUBLAS_OP_N, CUBLAS_OP_N,
      nData, trainY.getCols(),
      1.0, output.getData(), nData,
      -1.0, trainY.getData() + offset, trainY.getRows(),
      error.getData(), nData);

  return error;
}

void showAccuracy(size_t nError, size_t nTotal) {
  size_t nCorr = nTotal - nError;
  printf("Accuracy = %.2f%% ( %lu / %lu ) \n", (float) nCorr / nTotal * 100, nCorr, nTotal);
}

float str2float(const string &s) {
  return atof(s.c_str());
}

vector<string>& split(const string &s, char delim, vector<string>& elems) {
  stringstream ss(s);
  string item;
  while(getline(ss, item, delim))
    elems.push_back(item);
  return elems;
}

vector<string> split(const string &s, char delim) {
  vector<string> elems;
  return split(s, delim, elems);
}

vector<size_t> splitAsInt(const string &s, char delim) {
  vector<string> tokens = split(s, delim);
  vector<size_t> ints(tokens.size());

  for (size_t i=0; i<ints.size(); ++i)
    ints[i] = ::atoi(tokens[i].c_str());

  return ints;
}

std::vector<size_t> randperm(size_t N) {
  std::vector<size_t> perm(N);

  for (size_t i=0; i<N; ++i)
    perm[i] = i;
  
  std::random_shuffle ( perm.begin(), perm.end() );

  return perm;
}

bool isLabeled(const mat& labels) {
  return getLabelMapping(labels).size() > 1;
}
