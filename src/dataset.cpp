#include <dataset.h>
#include <dnn-utility.h>
#include <thread>
#include <future>

/*! /brief Get a batch of data. Because of the original ill-design,
 *         the data fed into DNN need tranpose.
 *
 *	     |  Before Transposed  |  After Transposed    |
 *           |                     | (the thing returned) |
 * ----------+---------------------+----------------------+
 * # of rows |  feature dimension  |  # of data in batch  |
 * # of cols |  # of data in batch |  feature dimension   |
 *
 */

std::ifstream& goToLine(std::ifstream& file, unsigned long num){
  file.seekg(std::ios::beg);
  
  if (num == 0)
    return file;

  for(size_t i=0; i < num; ++i)
    file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

  return file;
}

DataStream::DataStream(): _line_number(0), _start(0), _end(-1) {
}

DataStream::DataStream(const string& filename, size_t start, size_t end) {
  this->init(filename, start, end);
}

DataStream::~DataStream() {
  _fs.close();
}

void DataStream::init(const string& filename, size_t start, size_t end) {
  _filename = filename;
  _start = start;
  _end = end;
  _line_number = _start;

  if (_fs.is_open())
    _fs.close();

  _fs.open(_filename.c_str());

  if (!_fs.is_open())
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + filename);

  std::ifstream fin(_filename.c_str()); 
  _nLines = std::count(std::istreambuf_iterator<char>(fin), 
      std::istreambuf_iterator<char>(), '\n');
  fin.close();

  goToLine(_fs, _start);

  _end = min(_nLines, _end);
  _nLines = min(_nLines, _end - _start);
}

string DataStream::getline() {
  string line;

  // printf("_line_number = %lu, _end = %lu\n", _line_number, _end);

  if ( _line_number >= _end )
    this->rewind();

  if ( !std::getline(_fs, line) ) {
    this->rewind();
    std::getline(_fs, line);
  }

  ++_line_number;

  return line;
}

void DataStream::rewind() {
  _fs.clear();
  goToLine(_fs, _start);
  _line_number = _start;
}

size_t DataStream::count_lines() const {
  return _nLines;
}

mat getBatchData(const hmat& data, const Batches::Batch& b) {
  return ~mat(data.getData() + b.offset * data.getRows(), data.getRows(), b.nData);
}

DataSet::DataSet(): _dim(0) {
}

DataSet::DataSet(const string &fn, size_t dim, size_t start, size_t end)
  : _dim(dim), _stream(fn, start, end), _sparse(isFileSparse(fn)) {
}

void DataSet::normalizeToStandardScore(const hmat& mean, const hmat& deviation) {
  hmat& data = _hx;

  size_t nData = data.getCols();

  for (size_t i=0; i<_dim; ++i) {
    for (size_t j=0; j<nData; ++j)
      data(i, j) -= mean[i];
    
    if (deviation[i] == 0)
      continue;

    for (size_t j=0; j<nData; ++j)
      data(i, j) /= deviation[i];
  }
}

void DataSet::normalizeToStandardScore() {
  hmat& data = _hx;
  size_t nData = data.getCols();

  for (size_t i=0; i<_dim; ++i) {
    float mean = 0;
    for (size_t j=0; j<nData; ++j)
      mean += data(i, j);
    mean /= nData;

    for (size_t j=0; j<nData; ++j)
      data(i, j) -= mean;

    if (nData <= 1)
      continue;

    float deviation = 0;
    for (size_t j=0; j<nData; ++j)
      deviation += pow(data(i, j), 2.0f);
    deviation = sqrt(deviation / (nData - 1));

    // printf("mean = %.7e, deviation = %.7e\n", mean, deviation);

    if (deviation == 0)
      continue;

    for (size_t j=0; j<nData; ++j)
      data(i, j) /= deviation;
  }
}

void DataSet::normalize(const string &type) {

  return;

  if (type == "0") {
    return;
  }
  else if (type == "1") {
    clog << "\33[34m[Info]\33[0m Rescale to [0, 1] linearly" << endl;
    // Rescale each dimension to [0, 1] (for Bernoulli-Bernoulli RBM)
    //printf("\33[33m[Info]\33[0m Rescale each dimension to [0, 1]\n");
    linearScaling(0, 1);
  }
  else if (type == "2")  {
    clog << "\33[34m[Info]\33[0m Normalize to standard score" << endl;
    // Normalize to standard score z = (x-u)/sigma (i.e. CMVN in speech)
    //printf("\33[33m[Info]\33[0m Normalize each dimension to standard score\n");
    normalizeToStandardScore();
  }
  else {
    string fn = type;

    clog << "\33[34m[Info]\33[0m Normalize using \"" << fn << "\"" << endl;

    mat ss(fn);
    hmat statistics(ss);
    hmat mean(_dim, 1);
    hmat deviation(_dim, 1);

    if (_dim == statistics.getRows()) {
      for (size_t i=0; i<_dim; ++i) {
	mean[i] = statistics(i, 0);
	deviation[i] = statistics(i, 1);
      }
    }
    else if (_dim == statistics.getCols()) {
      for (size_t i=0; i<_dim; ++i) {
	mean[i] = statistics(0, i);
	deviation[i] = statistics(1, i);
      }
    }
    else
      throw runtime_error("ERROR: dimension mismatch");

    normalizeToStandardScore(mean, deviation);
  }
}

size_t DataSet::getFeatureDimension() const {
  return _dim;
}

size_t DataSet::size() const {
  return _stream.count_lines();
}

size_t DataSet::getClassNumber() const {
  return getLabelMapping(_hy).size();
}

bool DataSet::isLabeled() const {
  return getLabelMapping(_hy).size() > 1;
}

void DataSet::showSummary() const {
  return;

  printf("+--------------------------------+-----------+\n");
  printf("| Number of classes              | %9lu |\n", this->getClassNumber());
  printf("| Number of input feature (data) | %9lu |\n", this->size());
  printf("| Dimension of  input feature    | %9lu |\n", this->getFeatureDimension());
  printf("+--------------------------------+-----------+\n");

}

const hmat& DataSet::getX() const {
  return _hx;
}

const hmat& DataSet::getY() const {
  return _hy;
}

mat DataSet::getX(const Batches::Batch& b) {
  this->readMoreFeature(b.nData);
  return ((mat) _hx) / 255;
  // return _hx;
}

mat DataSet::getY(const Batches::Batch& b) {
  return ((mat) _hy) - 1;
}

void DataSet::set_sparse(bool sparse) {
  _sparse = sparse;
}

void DataSet::splitIntoTrainAndValidSet(DataSet& train, DataSet& valid, int ratio) {

  size_t nLines = this->_stream.count_lines();
  
  size_t nValid = nLines / ratio,
	 nTrain = nLines - nValid;

  printf("nLines = %lu, nTrain = %lu, nValid = %lu\n", nLines, nTrain, nValid);

  train.set_dimension(this->_dim);
  valid.set_dimension(this->_dim);

  train._sparse = this->_sparse;
  valid._sparse = this->_sparse;

  train._stream.init(this->_stream._filename, 0, nTrain);
  valid._stream.init(this->_stream._filename, nTrain, -1);
}

void DataSet::readMoreFeature(int N) {

  if (_sparse)
    this->readSparseFeature(N);
  else
    this->readDenseFeature(N);

  /*static std::future<hmat> hx, hy;

  _hx = hx.get();
  _hy = hy.get();

  hx = std::async(readSparseFeature, N);
  static std::thread* t = nullptr;  
  if (t) {
    t->join();
    delete t;
  }*/

  /*if (_sparse)
    t = new std::thread(readSparseFeature, N);*/
}

void DataSet::readSparseFeature(int N) {

  _hx.resize(N, _dim + 1);
  _hy.resize(N, 1);

  _hx.fillwith(0);
  _hx.fillwith(0);

  string line, token;

  for (int i=0; i<N; ++i) {
    string line = _stream.getline();
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
  
    // FIXME I'll remove it and move this into DNN. Since bias is only need by DNN,
    // not by CNN or other classifier.
    _hx(i, _dim) = 1.0f;
  }
}

void DataSet::readDenseFeature(int N) {
  
/*  string line, token;
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
*/
}

void DataSet::linearScaling(float lower, float upper) {

  // FIXME
  // This function rescale every pixel int 0~255 to float 0~1
  // , rather than rescale each dimension to 0~1
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

void DataSet::checkLabelBase(int base) {
  return;

  assert(_hy.getRows() == 1);

  int min_idx = _hy[0];
  for (size_t i=0; i<_hy.size(); ++i)
    min_idx = min(min_idx, (int) _hy[i]);

  if (min_idx != base)
    clog << "\33[33m[Warning]\33[0m The array is told to be " << base << "-based."
      "However, the minimum class id is " << min_idx << "." << endl;

  // This is IMPORTANT
  // Change to 0-based if it's 1-based originally.
  for (size_t i=0; i<_hy.size(); ++i)
    _hy[i] -= base;

}

void DataSet::shuffle() {

  return;

  std::vector<size_t> perm = randperm(size());

  hmat x(_hx), y(_hy);

  for (size_t i=0; i<size(); ++i) {
    for (size_t j=0; j<_dim + 1; ++j)
      _hx(j, perm[i]) = x(j, i);
    _hy[perm[i]] = y[i];
  }
}

bool isFileSparse(string fn) {
  ifstream fin(fn.c_str());
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
