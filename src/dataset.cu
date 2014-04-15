#include <dataset.h>
#include <dnn-utility.h>

mat getBatchData(const hmat& data, const Batches::Batch& b) {
  return ~mat(data.getData() + b.offset * data.getRows(), data.getRows(), b.nData);
}

DataSet::DataSet(): _dim(0) {
}

DataSet::DataSet(const string &fn, bool rescale): _dim(0) {

  read(fn);

  if (rescale)
    rescaleFeature();

  this->cvtLabelsToZeroBased();
}

size_t DataSet::getInputDimension() const {
  return _dim;
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

const hmat& DataSet::getX() const {
  return _hx;
}

const hmat& DataSet::getY() const {
  return _hy;
}

mat DataSet::getX(const Batches::Batch& b) const {
  return getBatchData(_hx, b);
}

mat DataSet::getY(const Batches::Batch& b) const {
  return getBatchData(_hy, b);
}

void DataSet::splitIntoTrainAndValidSet(DataSet& train, DataSet& valid, int ratio) {

  size_t inputDim = _hx.getRows();
  
  size_t nValid = size() / ratio,
	 nTrain = size() - nValid;

  printf("| nTrain                         | %9lu |\n", nTrain);
  printf("| nValid                         | %9lu |\n", nValid);

  // Copy data to training set
  train._hx.resize(inputDim , nTrain);
  train._hy.resize(1	    , nTrain);

  memcpy(train._hx.getData(), _hx.getData(), sizeof(float) * train._hx.size());
  memcpy(train._hy.getData(), _hy.getData(), sizeof(float) * train._hy.size());

  // Copy data to validation set
  valid._hx.resize(inputDim , nValid);
  valid._hy.resize(1	    , nValid);

  memcpy(valid._hx.getData(), _hx.getData() + train._hx.size(), sizeof(float) * valid._hx.size());
  memcpy(valid._hy.getData(), _hy.getData() + train._hy.size(), sizeof(float) * valid._hy.size());
}


void DataSet::read(const string &fn) {
  ifstream fin(fn.c_str());

  if (!fin.is_open())
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + fn);

  bool isSparse = isFileSparse(fn);

  perf::Timer timer;

  printf("Finding feature dimension...\n");
  timer.start();
  _dim = isSparse ? findMaxDimension(fin) : findDimension(fin);
  timer.elapsed();

  printf("Getting # of feature vector...\n");
  timer.start();
  size_t N = getLineNumber(fin);
  timer.elapsed();

  _hx.resize(_dim + 1, N);
  _hy.resize(1, N);

  printf("Parsing features...\n");
  timer.start();
  if (isSparse)
    readSparseFeature(fin);
  else
    readDenseFeature(fin);
  timer.elapsed();

  fin.close();

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

  printf("\33[33m[Info]\33[0m rescale each feature to [%f, %f]\n", lower, upper);

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

void DataSet::cvtLabelsToZeroBased() {
  assert(_hy.getRows() == 1);

  // Replace labels to 1, 2, 3, N, using mapping
  map<int, int> classes = getLabelMapping(_hy);
  for (size_t i=0; i<_hy.size(); ++i)
    _hy[i] = classes[_hy[i]];
}

void DataSet::shuffleFeature() {

  std::vector<size_t> perm = randperm(size());

  hmat x(_hx), y(_hy);

  for (size_t i=0; i<size(); ++i) {
    for (size_t j=0; j<_dim + 1; ++j)
      _hx(j, perm[i]) = x(j, i);
    _hy[perm[i]] = y[i];
  }
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
