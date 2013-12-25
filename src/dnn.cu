#include <dnn.h>

DNN::DNN() {}

DNN::DNN(string fn): _dims(0) {
  this->read(fn);
}

DNN::DNN(const std::vector<size_t>& dims): _dims(dims) {
  _weights.resize(_dims.size() - 1);

  for (size_t i=0; i<_weights.size(); ++i) {
    size_t M = _dims[i] + 1;
    size_t N = _dims[i + 1];
    _weights[i].resize(M, N);
  }

  randInit();
}

DNN::DNN(const DNN& source): _dims(source._dims), _weights(source._weights) {
}

DNN& DNN::operator = (DNN rhs) {
  swap(*this, rhs);
  return *this;
}

size_t DNN::getNLayer() const {
  return _dims.size(); 
}

size_t DNN::getDepth() const {
  return _dims.size() - 2;
}

#pragma GCC diagnostic ignored "-Wunused-result"
void readweight(FILE* fid, float* w, size_t rows, size_t cols) {

  for (size_t i=0; i<rows - 1; ++i)
    for (size_t j=0; j<cols; ++j)
      fscanf(fid, "%f ", &(w[j * rows + i]));

  fscanf(fid, "]\n<sigmoid>\n [");

  for (size_t j=0; j<cols; ++j)
    fscanf(fid, "%f ", &(w[j * rows + rows - 1]));
  fscanf(fid, "]\n");

}

#pragma GCC diagnostic ignored "-Wunused-result"
void DNN::read(string fn) {
  FILE* fid = fopen(fn.c_str(), "r");

  _dims.clear();
  _weights.clear();

  size_t rows, cols;

  while (fscanf(fid, "<affinetransform> %lu %lu\n [\n", &rows, &cols) != EOF) {

    printf("rows = %lu, cols = %lu \n", rows, cols);

    float* w = new float[(rows + 1) * cols];
    readweight(fid, w, rows + 1, cols);
    _weights.push_back(mat(w, rows + 1, cols));
    delete [] w;

    _dims.push_back(rows);
  }
  _dims.push_back(cols);

  fclose(fid);
}

void DNN::save(string fn) const {
  FILE* fid = fopen(fn.c_str(), "w");

  for (size_t i=0; i<_weights.size(); ++i) {
    const mat& w = _weights[i];

    size_t rows = w.getRows();
    size_t cols = w.getCols();

    fprintf(fid, "<affinetransform> %lu %lu \n", rows - 1, cols);
    fprintf(fid, " [");

    // ==============================
    float* data = new float[w.size()];
    CCE(cudaMemcpy(data, w.getData(), sizeof(float) * w.size(), cudaMemcpyDeviceToHost));

    for (size_t j=0; j<rows-1; ++j) {
      fprintf(fid, "\n  ");
      for (size_t k=0; k<cols; ++k)
	fprintf(fid, "%.7f ", data[k * rows + j]);
    }
    fprintf(fid, "]\n");

    fprintf(fid, "<sigmoid> \n [");
    for (size_t j=0; j<cols; ++j)
      fprintf(fid, "%.7f ", data[j * rows + rows - 1]);
    fprintf(fid, " ]\n");

    delete [] data;
  }
  
  fclose(fid);
}

void DNN::print() const {
  for (size_t i=0; i<_weights.size(); ++i)
    _weights[i].print(stdout);
}

void DNN::getEmptyGradient(std::vector<mat>& g) const {
  g.resize(_weights.size());
  for (size_t i=0; i<_weights.size(); ++i) {
    int m = _weights[i].getRows();
    int n = _weights[i].getCols();
    g[i].resize(m, n);
  }
}

std::vector<mat>& DNN::getWeights() { return _weights; }
const std::vector<mat>& DNN::getWeights() const { return _weights; }
std::vector<size_t>& DNN::getDims() { return _dims; }
const std::vector<size_t>& DNN::getDims() const { return _dims; }


void DNN::randInit() {
  for (size_t i=0; i<_weights.size(); ++i)
    ext::randn(_weights[i]);
}

// ========================
// ===== Feed Forward =====
// ========================

void print(const thrust::host_vector<float>& hv) {
  cout << "\33[33m[";
  for (size_t i=0; i<hv.size(); ++i)
    cout << hv[i] << " ";
  cout << " ] \33[0m" << endl << endl;
}

void print(const mat& m) {
  thrust::device_ptr<float> dm(m.getData());
  thrust::host_vector<float> hm(dm, dm + m.size());

  ::print(hm);
}

void print(const thrust::device_vector<float>& dv) {
  thrust::host_vector<float> hv(dv.begin(), dv.end());
  ::print(hv);
}

void DNN::feedForward(const mat& x, std::vector<mat>* hidden_output) {
  assert(hidden_output != NULL);

  std::vector<mat>& O = *hidden_output;
  assert(O.size() == _dims.size());

  O[0] = add_bias(x);

  /*for (size_t i=0; i<_weights.size(); ++i) {
    cout << "_weights[" << i << "] = " << endl;
    ::print(_weights[i]);
  }*/

  for (size_t i=1; i<O.size() - 1; ++i)
    O[i] = ext::b_sigmoid(O[i-1] * _weights[i-1]);

  size_t end = O.size() - 1;
  O.back() = ext::sigmoid(O[end - 1] * _weights[end - 1]);
}

// ============================
// ===== Back Propagation =====
// ============================

void DNN::backPropagate(mat& delta, std::vector<mat>& O, std::vector<mat>& gradient, const vec& coeff) {
  assert(gradient.size() == _weights.size());

  for (int i=_weights.size() - 1; i >= 0; --i) {

    gradient[i] = ~O[i] * delta;
    delta *= ~_weights[i];
    
    /*printf("after *= \n");
    ::print(delta);*/

    /*cout << "gradient[" << i << "] = " << endl;
    ::print(gradient[i]);*/

    thrust::device_vector<float> temp(O[i].size());

    thrust::device_ptr<float> output(O[i].getData());
    thrust::transform(output, output + O[i].size(), temp.begin(), func::dsigma<float>());

    thrust::device_ptr<float> dv1(delta.getData());
    thrust::transform(dv1, dv1 + delta.size(), temp.begin(), dv1, thrust::multiplies<float>());

    // Remove bias (last column)
    //cout << "before resize" << endl;
    //mylog(delta.getRows());
    //mylog(delta.getCols());
    //::print(delta);

    delta.resize(delta.getRows(), delta.getCols() - 1);

    //cout << "after resize" << endl;
    //mylog(delta.getRows());
    //mylog(delta.getCols());
    //::print(delta);

  }
}

void DNN::updateParameters(std::vector<mat>& gradient, float learning_rate) { 
  for (size_t i=0; i<_weights.size(); ++i) {
    gradient[i] *= learning_rate;

    /*float ng = nrm2(gradient[i]);
    float nw = nrm2(_weights[i]);
    float ratio = ng/nw;

    printf("w[%lu]: ng / nw = %.6e / %.6e = %.6e\n", i, ng, nw, ratio);
    if (ratio > 0.05)
      learning_rate *= 0.05 / ratio;*/

    _weights[i] -= /*learning_rate * */gradient[i];

    // learning_rate *= 4;
  }
}

void swap(DNN& lhs, DNN& rhs) {
  using WHERE::swap;
  swap(lhs._dims   , rhs._dims   );
  swap(lhs._weights, rhs._weights);
}

// =============================
// ===== Utility Functions =====
// =============================

mat l2error(mat& targets, mat& predicts) {
  mat err(targets - predicts);

  thrust::device_ptr<float> ptr(err.getData());
  thrust::transform(ptr, ptr + err.size(), ptr, func::square<float>());

  mat sum_matrix(err.getCols(), 1);
  err *= sum_matrix;
  
  return err;
}

