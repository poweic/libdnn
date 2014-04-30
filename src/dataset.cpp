#include <dataset.h>
#include <dnn-utility.h>
#include <thread>
#include <future>

size_t countLines(const string& fn) {
  printf("Loading file: \33[32m%s\33[0m (try to find out how many data) ...", fn.c_str());
  fflush(stdout);
  
  std::ifstream fin(fn.c_str()); 
  size_t N = std::count(std::istreambuf_iterator<char>(fin), 
      std::istreambuf_iterator<char>(), '\n');
  fin.close();

  printf("\t\33[32m[Done]\33[0m\n");
  return N;
}

std::ifstream& goToLine(std::ifstream& file, unsigned long num){
  file.seekg(std::ios::beg);
  
  if (num == 0)
    return file;

  for(size_t i=0; i < num; ++i)
    file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

  return file;
}

DataStream::DataStream(): _nLines(0), _line_number(0), _start(0), _end(-1) {
}

DataStream::DataStream(const string& filename, size_t start, size_t end) : _nLines(0) {
  this->init(filename, start, end);
}

DataStream::DataStream(const DataStream& src) : _nLines(src._nLines),
    _line_number(src._line_number), _filename(src._filename),
    _start(src._start), _end(src._end) {
  this->init(_filename, _start, _end);
}

DataStream::~DataStream() {
  _fs.close();
}

DataStream& DataStream::operator = (DataStream that) {
  swap(*this, that);
  return *this;
}

void DataStream::init(const string& filename, size_t start, size_t end) {
  if (_fs.is_open())
    _fs.close();

  _filename = filename;
  _start = start;
  _end = end;
  _line_number = _start;

  _fs.open(_filename.c_str());

  if (!_fs.is_open())
    throw std::runtime_error("\33[31m[Error]\33[0m Cannot load file: " + filename);

  if (_nLines == 0)
    _nLines = countLines(_filename);

  goToLine(_fs, _start);

  _end = min(_nLines, _end);
  _nLines = min(_nLines, _end - _start);
}

string DataStream::getline() {
  string line;

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

void swap(DataStream& a, DataStream& b) { 
  std::swap(a._nLines, b._nLines);
  std::swap(a._line_number, b._line_number);
  std::swap(a._filename, b._filename);
  std::swap(a._start, b._start);
  std::swap(a._end, b._end);
}

/* \brief Constructors for DataSet
 */
DataSet::DataSet() {
}

DataSet::DataSet(const string &fn, size_t dim, int base, size_t start, size_t end)
  : _dim(dim), _stream(fn, start, end), _sparse(isFileSparse(fn)), _base(base) {
}

DataSet::DataSet(const DataSet& src)
  : _dim(src._dim), _stream(src._stream), _sparse(src._sparse), _type(src._type),
  _base(src._base), _mean(src._mean), _dev(src._dev), _min(src._min), _max(src._max) {
}

DataSet& DataSet::operator = (DataSet that) {
  swap(*this, that);
  return *this;
}

void DataSet::normalize(NormType type) {
  switch (type) {
    case NO_NORMALIZATION: break;
    case LINEAR_SCALING: normalizeByLinearScaling(); break;
    case STANDARD_SCORE: normalizeToStandardScore(); break;
  }
}

void DataSet::normalizeByLinearScaling() {
  size_t nData = _hx.getRows();

  for (size_t i=0; i<_dim; ++i) {
    float r = _max[i] - _min[i];
    if (r == 0)
      continue;

    for (size_t j=0; j<nData; ++j)
      _hx(j, i) = (_hx(j, i) - _min[i]) / r;
  }
}

void DataSet::normalizeToStandardScore() {
  size_t nData = _hx.getRows();

  for (size_t i=0; i<_dim; ++i) {
    for (size_t j=0; j<nData; ++j)
      _hx(j, i) -= _mean[i];
    
    if (_dev[i] == 0)
      continue;

    for (size_t j=0; j<nData; ++j)
      _hx(j, i) /= _dev[i];
  }
}

void DataSet::findMaxAndMinOfEachDimension() {
  size_t N = _stream.count_lines();

  assert(_dim > 0);
  assert(N > 0);

  _min.resize(_dim);
  _max.resize(_dim);

  for (size_t j=0; j<_dim; ++j) {
    _min[j] = std::numeric_limits<double>::max();
    _max[j] = -std::numeric_limits<double>::max();
  }

  Batches batches(1024, N);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    this->readMoreFeature(itr->nData);

    for (size_t i=0; i<_hx.getRows(); ++i) {
      for (size_t j=0; j<_dim; ++j) {
	_min[j] = min( (float) _min[j], _hx(i, j));
	_max[j] = max( (float) _max[j], _hx(i, j));
      }
    }
  }

  this->_stream.rewind();
}

void DataSet::computeMeanAndDeviation() {

  size_t N = _stream.count_lines();

  assert(_dim > 0);
  assert(N > 0);

  _mean.resize(_dim);
  _dev.resize(_dim);

  for (size_t j=0; j<_dim; ++j)
    _mean[j] = _dev[j] = 0;

  Batches batches(1024, N);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    this->readMoreFeature(itr->nData);

    for (size_t i=0; i<_hx.getRows(); ++i) {
      for (size_t j=0; j<_dim; ++j) {
	_mean[j] += _hx(i, j);
	_dev[j] += pow( (double) _hx(i, j), 2);
      }
    }
  }

  for (size_t j=0; j<_dim; ++j) {
    _mean[j] /= N;
    _dev[j] = sqrt((_dev[j] / N) - pow(_mean[j], 2));
  }

  this->_stream.rewind();
}

void DataSet::loadPrecomputedStatistics(string fn) {
  if (fn.empty())
    return;

  clog << "\33[34m[Info]\33[0m Normalize using \"" << fn << "\"" << endl;

  mat ss(fn);
  hmat statistics(ss);

  vector<double> a(_dim), b(_dim);

  if (_dim == statistics.getRows()) {
    for (size_t i=0; i<_dim; ++i) {
      a[i] = statistics(i, 0);
      b[i] = statistics(i, 1);
    }
  }
  else if (_dim == statistics.getCols()) {
    for (size_t i=0; i<_dim; ++i) {
      a[i] = statistics(0, i);
      b[i] = statistics(1, i);
    }
  }
  else
    throw runtime_error("ERROR: dimension mismatch");

  switch (_type) {
    case NO_NORMALIZATION: break;
    case LINEAR_SCALING: _min = a; _max = b; break;
    case STANDARD_SCORE: _mean = a; _dev = b; break;
  }
}

void DataSet::setNormType(NormType type) {

  _type = type;
  switch (_type) {
    case NO_NORMALIZATION: break;
    case LINEAR_SCALING:
      clog << "\33[34m[Info]\33[0m Rescale to [0, 1] linearly" << endl;
      findMaxAndMinOfEachDimension();
      break;
    case STANDARD_SCORE:
      clog << "\33[34m[Info]\33[0m Normalize to standard score" << endl;
      computeMeanAndDeviation();
      break;
  }
}

size_t DataSet::size() const {
  return _stream.count_lines();
}

void DataSet::showSummary() const {

  printf("+--------------------------------+-----------+\n");
  printf("| Number of input feature (data) | %9lu |\n", this->size());
  printf("| Dimension of  input feature    | %9lu |\n", _dim);
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
  this->normalize(_type);
  return _hx;
}

mat DataSet::getY(const Batches::Batch& b) {
  return _hy;
}

void DataSet::set_sparse(bool sparse) {
  _sparse = sparse;
}

void DataSet::split( const DataSet& data, DataSet& train, DataSet& valid, int ratio) {

  size_t nLines = data._stream.count_lines();
  
  size_t nValid = nLines / ratio,
	 nTrain = nLines - nValid;

  train = data;
  valid = data;

  train._stream.init(data._stream._filename, 0, nTrain);
  valid._stream.init(data._stream._filename, nTrain, -1);
}

void DataSet::readMoreFeature(int N) {

  if (_sparse)
    this->readSparseFeature(N);
  else
    this->readDenseFeature(N);
}

void DataSet::readSparseFeature(int N) {

  _hx.resize(N, _dim + 1);
  _hy.resize(N, 1);

  _hx.fillwith(0);
  _hx.fillwith(0);

  string token;

  for (int i=0; i<N; ++i) {
    stringstream ss(_stream.getline());
  
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
    _hx(i, _dim) = 1;
  }

  for (int i=0; i<N; ++i)
    _hy[i] -= _base;
}

void DataSet::readDenseFeature(int N) {

  _hx.resize(N, _dim + 1);
  _hy.resize(N, 1);

  _hx.fillwith(0);
  _hx.fillwith(0);
  
  string token;

  for (int i=0; i<N; ++i) {
    stringstream ss(_stream.getline());
  
    ss >> token;
    _hy[i] = str2float(token);

    size_t j = 0;
    while (ss >> token)
      _hx(i, j++) = str2float(token);

    // FIXME I'll remove it and move this into DNN. Since bias is only need by DNN,
    // not by CNN or other classifier.
    _hx(i, _dim) = 1;
  }

  for (int i=0; i<N; ++i)
    _hy[i] -= _base;
}

void DataSet::setDimension(size_t dim) {
  _dim = dim;
}

void DataSet::setLabelBase(int base) {
  _base = base;
}

DataStream& DataSet::getDataStream() {
  return _stream;
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

void swap(DataSet& a, DataSet& b) {
  std::swap(a._dim, b._dim);
  swap(a._stream, b._stream);
  std::swap(a._sparse, b._sparse);
  std::swap(a._type, b._type);
  std::swap(a._base, b._base);
  std::swap(a._mean, b._mean);
  std::swap(a._dev, b._dev);
  std::swap(a._max, b._max);
  std::swap(a._min, b._min);
  std::swap(a._hx, b._hx);
  std::swap(a._hy, b._hy);
}
