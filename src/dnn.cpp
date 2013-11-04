#include <dnn.h>
#include <utility.h>

DNN::DNN() {}

DNN::DNN(string fn): _dims(0) {
  this->read(fn);
}

DNN::DNN(const std::vector<size_t>& dims): _dims(dims) {
  _weights.resize(_dims.size() - 1);

  foreach (i, _weights) {
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

void DNN::load(string prefix) {
  std::vector<string> n_ppWeights = bash::ls(prefix + "*");
  _weights.resize(n_ppWeights.size());

  foreach (i, _weights)
    _weights[i] = mat(prefix + int2str(i));

  _dims.resize(_weights.size() + 1);
  
  _dims[0] = _weights[0].getRows() - 1;
  range (i, _weights.size())
    _dims[i + 1] = _weights[i].getCols();
}

size_t DNN::getNLayer() const {
  return _dims.size(); 
}

size_t DNN::getDepth() const {
  return _dims.size() - 2;
}

#pragma GCC diagnostic ignored "-Wunused-result"
void DNN::read(string fn) {
  FILE* fid = fopen(fn.c_str(), "r");

  _dims.clear();
  _weights.clear();

  size_t rows, cols;

  while (fscanf(fid, "<affinetransform> %lu %lu\n [\n", &rows, &cols) != EOF) {

    printf("rows = %lu, cols = %lu \n", rows, cols);

    mat w(rows + 1, cols);
    for (size_t i=0; i<rows; ++i)
      for (size_t j=0; j<cols; ++j)
	fscanf(fid, "%f ", &(w[i][j]));

    fscanf(fid, "]\n<sigmoid>\n [");

    for (size_t j=0; j<cols; ++j)
      fscanf(fid, "%f ", &(w[rows][j]));
    fscanf(fid, "]\n");

    _dims.push_back(rows);
    _weights.push_back(w);
  }
  _dims.push_back(cols);

  fclose(fid);
}

void DNN::save(string fn) const {
  FILE* fid = fopen(fn.c_str(), "w");

  foreach (i, _weights) {
    const mat& w = _weights[i];
    fprintf(fid, "<affinetransform> %lu %lu \n", w.getRows() - 1, w.getCols());
    fprintf(fid, " [\n");

    fprintf(fid, "  ");
    for (size_t j=0; j<w.getRows() - 1; ++j) {
      for (size_t k=0; k<w.getCols(); ++k)
	fprintf(fid, "%.7f ", w[j][k]);
      
      fprintf(fid, (j == w.getRows() - 2) ? "]\n" : "\n");
    }

    // FIXME
    // Format in Kaldi-project is something like "<sigmoid> 2048 2048",
    // but I haven't have a clue about why.
    fprintf(fid, "<sigmoid> \n");
    fprintf(fid, " [");
    for (size_t j=0; j<w.getCols(); ++j)
      fprintf(fid, "%.7f ", w[w.getRows() - 1][j]);
    fprintf(fid, " ]\n");
  }
  
  fclose(fid);
}

void DNN::print() const {
  foreach (i, _weights)
    _weights[i].print(5);
}

void DNN::getEmptyGradient(std::vector<mat>& g) const {
  g.resize(_weights.size());
  foreach (i, _weights) {
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
  foreach (i, _weights)
    ext::randn(_weights[i]);
}

// ========================
// ===== Feed Forward =====
// ========================

void DNN::feedForward(const mat& x, std::vector<mat>* hidden_output) {
  assert(hidden_output != NULL);

  std::vector<mat>& O = *hidden_output;
  assert(O.size() == _dims.size());

  O[0] = add_bias(x);

  for (size_t i=1; i<O.size() - 1; ++i)
    O[i] = ext::b_sigmoid(O[i-1] * _weights[i-1]);

  size_t end = O.size() - 1;
  O.back() = ext::sigmoid(O[end - 1] * _weights[end - 1]);
}

/*void DNN::feedForward(const mat& x, std::vector<mat>* hidden_output) {
  assert(hidden_output != NULL);

  std::vector<mat>& O = *hidden_output;
  assert(O.size() == _dims.size());

  O[0] = add_bias(x);

  for (size_t i=1; i<O.size() - 1; ++i)
    O[i] = ext::b_sigmoid(O[i-1] * _weights[i-1]);

  size_t end = O.size() - 1;
  O.back() = ext::sigmoid(O[end - 1] * _weights[end - 1]);
}*/

// ============================
// ===== Back Propagation =====
// ============================
void DNN::backPropagate(vec& p, std::vector<vec>& O, std::vector<mat>& gradient) {

  assert(gradient.size() == _weights.size());

  reverse_foreach (i, _weights) {
    gradient[i] = O[i] * p;
    p = dsigma(O[i]) & (p * ~_weights[i]); // & stands for .* in MATLAB

    // Remove bias
    remove_bias(p);
  }
}

void DNN::backPropagate(mat& p, std::vector<mat>& O, std::vector<mat>& gradient, const vec& coeff) {
  assert(gradient.size() == _weights.size());

  reverse_foreach (i, _weights) {
    gradient[i] = ~O[i] * (p & coeff);
    p = dsigma(O[i]) & (p * ~_weights[i]);

    // Remove bias
    remove_bias(p);
  }
}

void DNN::updateParameters(std::vector<mat>& gradient, float learning_rate) {
  foreach (i, _weights)
    _weights[i] -= learning_rate * gradient[i];
}

void swap(DNN& lhs, DNN& rhs) {
  using WHERE::swap;
  swap(lhs._dims   , rhs._dims   );
  swap(lhs._weights, rhs._weights);
}

void swap(HIDDEN_OUTPUT& lhs, HIDDEN_OUTPUT& rhs) {
  using WHERE::swap;
  swap(lhs.hox, rhs.hox);
  swap(lhs.hoy, rhs.hoy);
  swap(lhs.hoz, rhs.hoz);
  swap(lhs.hod, rhs.hod);
}

void swap(GRADIENT& lhs, GRADIENT& rhs) {
  using WHERE::swap;
  swap(lhs.grad1, rhs.grad1);
  swap(lhs.grad2, rhs.grad2);
  swap(lhs.grad3, rhs.grad3);
  swap(lhs.grad4, rhs.grad4);
}
