#include <dnn.h>
#include <thrust/extrema.h>

DNN::DNN(): _transforms(), _config() {}

DNN::DNN(string fn): _transforms(), _config() {
  this->read(fn);
}

DNN::DNN(const Config& config): _transforms(), _config(config) {
}

DNN::DNN(const DNN& source): _transforms(source._transforms.size()), _config() {

  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i] = source._transforms[i]->clone();
}

void DNN::init(const std::vector<mat>& weights) {
  _transforms.resize(weights.size());

  for (size_t i=0; i<_transforms.size(); ++i)
      _transforms[i] = new Sigmoid(weights[i]);
  _transforms.back() = new Softmax(weights.back());
}

void DNN::init(const std::vector<size_t>& dims) {
  assert(dims.size() > 0);
  size_t L = dims.size() - 1;

  _transforms.resize(L);

  for (size_t i=0; i<L; ++i) {
    size_t M = dims[i] + 1;
    size_t N = dims[i+1] + 1;

    if (i == L-1)
      _transforms[i] = new Softmax(M, N, _config.variance);
    else
      _transforms[i] = new Sigmoid(M, N, _config.variance);
  }
}

DNN::~DNN() {
  for (size_t i=0; i<_transforms.size(); ++i)
    delete _transforms[i];
}

DNN& DNN::operator = (DNN rhs) {
  swap(*this, rhs);
  return *this;
}
  
void DNN::setConfig(const Config& config) {
  _config = config;
}

size_t DNN::getNLayer() const {
  return _transforms.size() + 1;
}

#pragma GCC diagnostic ignored "-Wunused-result"
void readweight(FILE* fid, float* w, size_t rows, size_t cols) {

  for (size_t i=0; i<rows - 1; ++i)
    for (size_t j=0; j<cols; ++j)
      fscanf(fid, "%f ", &(w[j * rows + i]));

  fscanf(fid, "]\n<bias>\n [");

  for (size_t j=0; j<cols; ++j)
    fscanf(fid, "%f ", &(w[j * rows + rows - 1]));
  fscanf(fid, "]\n");

}

void DNN::read(string fn) {
  FILE* fid = fopen(fn.c_str(), "r");

  _transforms.clear();

  size_t rows, cols;
  char type[80];

  while (fscanf(fid, "%s", type) != EOF) {
    fscanf(fid, "%lu %lu\n [\n", &rows, &cols);
    printf("\33[34m%-17s\33[0m %-6lu x %-6lu \n", type, rows, cols);

    float* hw = new float[(rows + 1) * (cols + 1)];
    readweight(fid, hw, rows + 1, cols);

    // Reserve one more column for bias)
    mat w(hw, rows + 1, cols + 1);

    string transformType = string(type);
    if (transformType == "<sigmoid>")
      _transforms.push_back(new Sigmoid(w));
    else if (transformType == "<softmax>")
      _transforms.push_back(new Softmax(w));

    delete [] hw;
  }

  fclose(fid);
}

void DNN::save(string fn) const {
  FILE* fid = fopen(fn.c_str(), "w");

  for (size_t i=0; i<_transforms.size(); ++i) {
    const mat& w = _transforms[i]->getW();

    size_t rows = w.getRows();
    size_t cols = w.getCols() - 1;

    fprintf(fid, "<%s> %lu %lu \n", _transforms[i]->toString().c_str(), rows - 1, cols);
    fprintf(fid, " [");

    // ==============================
    std::vector<float> data = copyToHost(w);
    // float* data = new float[w.size()];
    // CCE(cudaMemcpy(data, w.getData(), sizeof(float) * w.size(), cudaMemcpyDeviceToHost));

    for (size_t j=0; j<rows-1; ++j) {
      fprintf(fid, "\n  ");
      for (size_t k=0; k<cols; ++k)
	fprintf(fid, "%g ", data[k * rows + j]);
    }
    fprintf(fid, "]\n");

    fprintf(fid, "<bias> \n [");
    for (size_t j=0; j<cols; ++j)
      fprintf(fid, "%g ", data[j * rows + rows - 1]);
    fprintf(fid, " ]\n");

    // delete [] data;
  }

  printf("nn_structure ");
  for (size_t i=0; i<_transforms.size(); ++i)
    printf("%lu ", _transforms[i]->getW().getRows());
  printf("%lu\n", _transforms.back()->getW().getCols());
  
  fclose(fid);
}

void DNN::print() const {
  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i]->getW().print();
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

void print(const thrust::device_vector<float>& dv) {
  thrust::host_vector<float> hv(dv.begin(), dv.end());
  ::print(hv);
}

void DNN::adjustLearningRate(float trainAcc) {
  static size_t phase = 0;

  if ( (trainAcc > 0.80 && phase == 0) ||
       (trainAcc > 0.85 && phase == 1) ||
       (trainAcc > 0.90 && phase == 2) ||
       (trainAcc > 0.92 && phase == 3) ||
       (trainAcc > 0.95 && phase == 4) ||
       (trainAcc > 0.97 && phase == 5)
     ) {

    float ratio = 0.9;
    printf("\33[33m[Info]\33[0m Adjust learning rate from \33[32m%.7f\33[0m to \33[32m%.7f\33[0m\n", _config.learningRate, _config.learningRate * ratio);
    _config.learningRate *= ratio;
    ++phase;
  }
}

void DNN::feedForward(mat& output, const mat& fin) {

  // FIXME This should be an ASSERTION, not resizing.
  if (_houts.size() != this->getNLayer() - 2)
    _houts.resize(this->getNLayer() - 2);

  _transforms[0]->feedForward(_houts[0], fin);

  for (size_t i=1; i<_transforms.size()-1; ++i)
    _transforms[i]->feedForward(_houts[i], _houts[i-1]);

  _transforms.back()->feedForward(output, _houts.back());

  output.resize(output.getRows(), output.getCols() - 1);
}

// ============================
// ===== Back Propagation =====
// ============================

void DNN::backPropagate(mat& error, const mat& fin, const mat& fout) {
  _transforms.back()->backPropagate(error, _houts.back(), fout);

  for (int i=_transforms.size() - 2; i >= 1; --i)
    _transforms[i]->backPropagate(error, _houts[i-1], _houts[i]);

  _transforms[0]->backPropagate(error, fin, _houts[0]);
}

void DNN::update(float learning_rate) { 
  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i]->update(learning_rate);
}

Config DNN::getConfig() const {
  return _config;
}

void swap(DNN& lhs, DNN& rhs) {
  using WHERE::swap;
  swap(lhs._transforms, rhs._transforms);
  swap(lhs._config, rhs._config);
}

// =============================
// ===== Utility Functions =====
// =============================

/*mat l2error(mat& targets, mat& predicts) {
  mat err(targets - predicts);

  thrust::device_ptr<float> ptr(err.getData());
  thrust::transform(ptr, ptr + err.size(), ptr, func::square<float>());

  mat sum_matrix(err.getCols(), 1);
  err *= sum_matrix;
  
  return err;
}
*/
