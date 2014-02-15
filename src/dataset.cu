#include <dataset.h>
#include <dnn-utility.h>

DataSet::DataSet(): _dim(0) {
}

DataSet::DataSet(const string &fn, bool rescale): _dim(0) {

  read(fn, rescale);

  this->convertToStandardLabels();
  this->label2PosteriorProb();
  this->shuffleFeature();
}

size_t DataSet::getInputDimension() const {
  return _dim;
}

size_t DataSet::getOutputDimension() const {
  return _hp.getCols();
}

size_t DataSet::size() const {
  return _hy.size();
}

size_t DataSet::getClassNumber() const {
  return getLabelMapping(_hy).size();
}

bool DataSet::isLabeled() const {
  return getLabelMapping(_hy).size() > 1;
}

void DataSet::showSummary() const {

  printf("+--------------------------------+-----------+\n");
  printf("| Number of classes              | %9lu |\n", this->getClassNumber());
  printf("| Number of input feature (data) | %9lu |\n", this->size());
  printf("| Dimension of  input feature    | %9lu |\n", this->getInputDimension());
  printf("+--------------------------------+-----------+\n");

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

  // Copy data to validation set
  valid._hx.resize(inputDim , nValid);
  valid._hy.resize(1	    , nValid);
  valid._hp.resize(outputDim, nValid);

  memcpy(valid._hx.getData(), _hx.getData() + train._hx.size(), sizeof(float) * valid._hx.size());
  memcpy(valid._hy.getData(), _hy.getData() + train._hy.size(), sizeof(float) * valid._hy.size());
  memcpy(valid._hp.getData(), _hp.getData() + train._hp.size(), sizeof(float) * valid._hp.size());
}


void DataSet::read(const string &fn, bool rescale) {
  ifstream fin(fn.c_str());

  bool isSparse = isFileSparse(fn);

  _dim = isSparse ? findMaxDimension(fin) : findDimension(fin);
  size_t N = getLineNumber(fin);

  _hx.resize(_dim + 1, N);
  _hy.resize(1, N);

  if (isSparse)
    readSparseFeature(fin);
  else
    readDenseFeature(fin);

  fin.close();

  if (rescale) {
    printf("\33[33m[Info]\33[0m rescale each feature to [0, 1]\n");
    rescaleFeature();
  }

  for (size_t i=0; i<N; ++i)
    _hx(_dim, i) = 1;
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
      
      _hx(j, i) = value;
    }
    ++i;
  }
}

void DataSet::readDenseFeature(ifstream& fin) {
  
  string line, token;
  size_t i = 0;
  while (std::getline(fin, line)) {
    stringstream ss(line);
  
    ss >> token;
    _hy[i] = str2float(token);

    size_t j = 0;
    while (ss >> token)
      _hx(j++, i) = str2float(token);
    ++i;
  }
}

void DataSet::rescaleFeature(float lower, float upper) {

  for (size_t i=0; i<this->size(); ++i) {
    float min = _hx(0, i),
	  max = _hx(0, i);

    for (size_t j=0; j<_dim; ++j) {
      float x = _hx(j, i);
      if (x > max) max = x;
      if (x < min) min = x;
    }

    if (max == min) {
      for (size_t j=0; j<_dim; ++j)
	_hx(j, i) = upper;
      continue;
    }

    float ratio = (upper - lower) / (max - min);
    for (size_t j=0; j<_dim; ++j)
      _hx(j, i) = (_hx(j, i) - min) * ratio + lower;
  }
}

void DataSet::convertToStandardLabels() {
  assert(_hy.getRows() == 1);

  // Replace labels to 1, 2, 3, N, using mapping
  map<int, int> classes = getLabelMapping(_hy);
  for (size_t i=0; i<_hy.size(); ++i)
    _hy[i] = classes[_hy[i]];
}

void DataSet::label2PosteriorProb() {
  
  map<int, int> classes = getLabelMapping(_hy);
  size_t nClasses = classes.size();

  // Convert labels to posterior probabilities
  _hp.resize(nClasses, _hy.getCols());
  _hp.fillwith(0);

  for (size_t i=0; i<_hp.getCols(); ++i)
    _hp((_hy[i] - 1), i) = 1;
}

void DataSet::shuffleFeature() {

  std::vector<size_t> perm = randperm(size());

  hmat x(_hx), y(_hy);

  for (size_t i=0; i<size(); ++i) {
    for (size_t j=0; j<_dim + 1; ++j)
      _hx(j, perm[i]) = x(j, i);
    _hy[perm[i]] = y[i];
  }

  label2PosteriorProb();
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
