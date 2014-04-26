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

  for (size_t i=0; i<_transforms.size() - 1; ++i)
      _transforms[i] = new Sigmoid(weights[i]);
  // _transforms.back() = new Softmax(weights.back());
  _transforms.back() = new Sigmoid(weights.back());
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

std::vector<size_t> DNN::getDimensions() const {
  std::vector<size_t> dims;
  for (int i=0; i<_transforms.size(); ++i)
    dims.push_back(_transforms[i]->getInputDimension());
  dims.push_back(_transforms.back()->getOutputDimension());
  return dims;
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

  if (!fid)
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + fn);

  _transforms.clear();

  //printf("+-------------------------------------+\n");
  //printf("| File: %-30s"                       "|\n", fn.c_str());
  //printf("+-------------------+--------+--------+\n");
  //printf("| Feature Transform |  rows  |  cols  |\n");
  //printf("+-------------------+--------+--------+\n");

  char type[80];

  while (fscanf(fid, "%s", type) != EOF) {
    size_t rows, cols;
    fscanf(fid, "%lu %lu\n [\n", &rows, &cols);

    // printf("|     %-13s |  %-5lu |  %-5lu |\n", type, rows, cols);

    float* hw = new float[(rows + 1) * (cols + 1)];
    readweight(fid, hw, rows + 1, cols);

    // Reserve one more column/row for bias
    mat w(hw, rows + 1, cols + 1);

    string transformType = string(type);
    if (transformType == "<sigmoid>")
      _transforms.push_back(new Sigmoid(w));
    else if (transformType == "<softmax>")
      _transforms.push_back(new Softmax(w));

    delete [] hw;
  }
  //printf("+-------------------+--------+--------+\n\n");

  fclose(fid);
}

void DNN::save(string fn) const {
  FILE* fid = fopen(fn.c_str(), "w");

  for (size_t i=0; i<_transforms.size(); ++i)
    _transforms[i]->print(fid);
  
  fclose(fid);
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

mat DNN::feedForward(const mat& fin) const {

  mat y;

  _transforms[0]->feedForward(y, fin);

  for (size_t i=1; i<_transforms.size(); ++i)
    _transforms[i]->feedForward(y, y);

  y.resize(y.getRows(), y.getCols() - 1);

  return y;
}

void DNN::feedForward(mat& output, const mat& fin) {

  // FIXME This should be an ASSERTION, not resizing.
  if (_houts.size() != this->getNLayer() - 2)
    _houts.resize(this->getNLayer() - 2);

  if (_houts.size() > 0) {
    _transforms[0]->feedForward(_houts[0], fin);

    for (size_t i=1; i<_transforms.size()-1; ++i)
      _transforms[i]->feedForward(_houts[i], _houts[i-1]);

    _transforms.back()->feedForward(output, _houts.back());
  }
  else {
    _transforms.back()->feedForward(output, fin);
  }

  output.resize(output.getRows(), output.getCols() - 1);
}

// ============================
// ===== Back Propagation =====
// ============================

void DNN::backPropagate(mat& error, const mat& fin, const mat& fout, float learning_rate) {

  mat output(fout);
  output.reserve(output.size() + output.getRows());
  output.resize(output.getRows(), output.getCols() + 1);

  error.reserve(error.size() + error.getRows());
  error.resize(error.getRows(), error.getCols() + 1);

  assert(error.getRows() == output.getRows() && error.getCols() == output.getCols());

  if (_houts.size() > 0) {
    _transforms.back()->backPropagate(error, _houts.back(), output, learning_rate);

    for (int i=_transforms.size() - 2; i >= 1; --i)
      _transforms[i]->backPropagate(error, _houts[i-1], _houts[i], learning_rate);

    _transforms[0]->backPropagate(error, fin, _houts[0], learning_rate);
  }
  else
    _transforms.back()->backPropagate(error, fin, output, learning_rate);
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
