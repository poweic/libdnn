#include <dataset.h>
#include <dnn-utility.h>

DataSet::DataSet(): _dim(0) {
}

DataSet::DataSet(const string &fn, bool rescale): _dim(0) {

  read(fn, rescale);

  this->convertToStandardLabels();
  this->label2PosteriorProb();
  this->shuffleFeature();

  _hx = ~_hx;
  _hy = ~_hy;
  _hp = ~_hp;
}

void DataSet::convertToStandardLabels() {
  assert(_hy.getCols() == 1);

  // Replace labels to 1, 2, 3, N, using mapping
  map<int, int> classes = getLabelMapping(_hy);
  for (size_t i=0; i<_hy.getRows(); ++i)
    _hy[i] = classes[_hy[i]];
}

size_t DataSet::getInputDimension() const {
  return _dim;
}

size_t DataSet::getOutputDimension() const {
  return _hp.getCols();
}

void DataSet::rescaleFeature(float lower, float upper) {

  size_t rows = _hx.getRows(),
	 cols = _hx.getCols();

  for (size_t i=0; i<rows; ++i) {
    float min = _hx(i, 0),
	  max = _hx(i, 0);

    for (size_t j=0; j<cols; ++j) {
      float x = _hx(i, j);
      if (x > max) max = x;
      if (x < min) min = x;
    }

    if (max == min) {
      for (size_t j=0; j<cols; ++j)
	_hx(i, j) = upper;
      continue;
    }

    float ratio = (upper - lower) / (max - min);
    for (size_t j=0; j<cols; ++j)
      _hx(i, j) = (_hx(i, j) - min) * ratio + lower;
  }
}

void DataSet::read(const string &fn, bool rescale) {
  ifstream fin(fn.c_str());

  bool isSparse = isFileSparse(fn);

  _dim = isSparse ? findMaxDimension(fin) : findDimension(fin);
  size_t N = getLineNumber(fin);

  _hx.resize(N, _dim);
  _hy.resize(N, 1);

  if (isSparse)
    readSparseFeature(fin);
  else
    readDenseFeature(fin);

  fin.close();

  // --------------------------------------
  if (rescale) {
    printf("\33[33m[Info]\33[0m rescale each feature to [0, 1]\n");
    rescaleFeature();
  }

  _hx.reserve(N * (_dim + 1));
  _hx.resize(N, _dim + 1);

  float* lastColumn = _hx.getData() + N * _dim;
  std::fill(lastColumn, lastColumn + N, 1.0f);

  // --------------------------------------
}

void DataSet::readSparseFeature(ifstream& fin) {

  string line, token;
  size_t i = 0;
  while (std::getline(fin, line)) {
    stringstream ss(line);
  
    ss >> token;
    _hy[i] = str2float(token);

    while (ss >> token) {
      size_t pos = token.find(':');
      if (pos == string::npos)
	continue;

      size_t j = str2float(token.substr(0, pos)) - 1;
      float value = str2float(token.substr(pos + 1));
      
      _hx(i, j) = value;
    }
    ++i;
  }
}

void DataSet::readDenseFeature(ifstream& fin) {
  
  size_t rows = _hx.getRows(),
	 cols = _hx.getCols();

  string line, token;
  size_t i = 0;
  while (std::getline(fin, line)) {
    stringstream ss(line);
  
    ss >> token;
    _hy[i] = str2float(token);

    size_t j = 0;
    while (ss >> token)
      _hx(i, j++) = str2float(token);
    ++i;
  }
}

void DataSet::showSummary() const {

  printf("+--------------------------------+-----------+\n");
  printf("| Number of classes              | %9lu |\n", this->getClassNumber());
  printf("| Number of input feature (data) | %9lu |\n", this->size());
  printf("| Dimension of  input feature    | %9lu |\n", this->getInputDimension());
  printf("+--------------------------------+-----------+\n");

}

size_t DataSet::getClassNumber() const {
  return getLabelMapping(_hy).size();
}

void DataSet::label2PosteriorProb() {
  
  map<int, int> classes = getLabelMapping(_hy);
  size_t nClasses = classes.size();

  // Convert labels to posterior probabilities
  _hp.resize(_hy.getRows(), nClasses);
  _hp.fillwith(0);

  for (size_t i=0; i<_hp.getRows(); ++i)
    _hp(i, (_hy[i] - 1)) = 1;
}

bool isFileSparse(string train_fn) {
  ifstream fin(train_fn.c_str());
  string line;
  std::getline(fin, line);
  return line.find(':') != string::npos;
}

size_t getLineNumber(ifstream& fin) {
  int previous_pos = fin.tellg();
  string a;
  size_t n = 0;
  while(std::getline(fin, a) && ++n);
  fin.clear();
  fin.seekg(previous_pos);
  return n;
}

size_t findMaxDimension(ifstream& fin) {
  int previous_pos = fin.tellg();

  string token;
  size_t maxDimension = 0;
  while (fin >> token) {
    size_t pos = token.find(':');
    if (pos == string::npos)
      continue;

    size_t dim = atoi(token.substr(0, pos).c_str());
    if (dim > maxDimension)
      maxDimension = dim;
  }

  fin.clear();
  fin.seekg(previous_pos);

  return maxDimension;
}

size_t findDimension(ifstream& fin) {

  size_t dim = 0;

  int previous_pos = fin.tellg();

  string line;
  std::getline(fin, line);
  stringstream ss(line);

  // First token is class label
  string token;
  ss >> token;

  while (ss >> token)
    ++dim;
  
  fin.clear();
  fin.seekg(previous_pos);

  return dim;
}

void DataSet::shuffleFeature() {

  size_t rows = _hx.getRows(),
	 cols = _hx.getCols();

  std::vector<size_t> perm = randperm(rows);

  hmat x(_hx), y(_hy);

  for (size_t i=0; i<rows; ++i) {
    for (size_t j=0; j<cols; ++j)
      _hx(perm[i], j) = x (i, j);
    _hy[perm[i]] = y[i];
  }

  label2PosteriorProb();
}

bool DataSet::isLabeled() const {
  return getLabelMapping(_hy).size() > 1;
}

size_t DataSet::size() const {
  return _hy.size();
}

mat DataSet::getX(size_t offset, size_t nData) const {
  size_t dim = _hx.getRows();
  mat x_transposed(_hx.getData() + offset * dim, dim, nData);
  return ~x_transposed;
}

mat DataSet::getX() const {
  return ~mat(_hx.getData(), _hx.getRows(), _hx.getCols());
}

mat DataSet::getY() const {
  return ~mat(_hy.getData(), _hy.getRows(), _hy.getCols());
}

mat DataSet::getY(size_t offset, size_t nData) const {
  size_t dim = _hy.getRows();
  mat y_transposed(_hy.getData() + offset * dim, dim, nData);
  return ~y_transposed;
}

mat DataSet::getProb() const {
  return ~mat(_hp.getData(), _hp.getRows(), _hp.getCols());
}

mat DataSet::getProb(size_t offset, size_t nData) const {

  size_t dim = _hp.getRows();
  mat p_transposed(_hp.getData() + offset * dim, dim, nData);
  return ~p_transposed;
}

void DataSet::splitIntoTrainAndValidSet(DataSet& train, DataSet& valid, int ratio) {

  size_t inputDim = _hx.getRows(),
	 outputDim = _hp.getRows();
  
  size_t nValid = size() / ratio,
	 nTrain = size() - nValid;

  printf("| nTrain                         | %9lu |\n", nTrain);
  printf("| nValid                         | %9lu |\n", nValid);

  // Copy data to training set
  train._hx.resize(inputDim , nTrain);
  train._hy.resize(1	    , nTrain);
  train._hp.resize(outputDim, nTrain);

  memcpy(train._hx.getData(), _hx.getData(), sizeof(float) * train._hx.size());
  memcpy(train._hy.getData(), _hy.getData(), sizeof(float) * train._hy.size());
  memcpy(train._hp.getData(), _hp.getData(), sizeof(float) * train._hp.size());

  /*for (size_t i=0; i<nTrain; ++i) {
    for (size_t j=0; j<inputDim; ++j)
      train._hx(i, j) = _hx(i, j);

    for (size_t j=0; j<outputDim; ++j)
      train._hp(i, j) = _hp(i, j);

    train._hy[i] = _hy[i];
  }*/

  // Copy data to validation set
  valid._hx.resize(inputDim , nValid);
  valid._hy.resize(1	    , nValid);
  valid._hp.resize(outputDim, nValid);

  memcpy(valid._hx.getData(), _hx.getData() + train._hx.size(), sizeof(float) * valid._hx.size());
  memcpy(valid._hy.getData(), _hy.getData() + train._hy.size(), sizeof(float) * valid._hy.size());
  memcpy(valid._hp.getData(), _hp.getData() + train._hp.size(), sizeof(float) * valid._hp.size());

  /*for (size_t i=0; i<nValid; ++i) {
    for (size_t j=0; j<inputDim; ++j)
      valid._hx(i, j) = _hx[j * rows + i + nTrain];

    for (size_t j=0; j<outputDim; ++j)
      valid._hp(i, j) = _hp[j * rows + i + nTrain];

    valid._hy[i] = _hy[i + nTrain];
  }*/
}
