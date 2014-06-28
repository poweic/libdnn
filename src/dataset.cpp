#include <dataset.h>
#include <dnn-utility.h>
#include <thread>
#include <future>

BatchData readMoreFeature(DataStream& stream, int N, size_t dim, size_t base, bool sparse) {

  BatchData data;
  if (sparse)
    readSparseFeature(stream, N, dim, base, data);
  else
    readDenseFeature(stream, N, dim, base, data);

  return data;
}

void readSparseFeature(DataStream& stream, int N, size_t dim, size_t base, BatchData& data) {

  data.x.resize(N, dim + 1);
  data.y.resize(N, 1);

  data.x.fillwith(0);
  data.y.fillwith(0);

  string token;

  for (int i=0; i<N; ++i) {
    stringstream ss(stream.getline());
  
    ss >> token;
    data.y[i] = str2float(token);

    while (ss >> token) {
      size_t pos = token.find(':');
      if (pos == string::npos)
	continue;

      size_t j = str2float(token.substr(0, pos)) - 1;
      float value = str2float(token.substr(pos + 1));

      data.x(i, j) = value;
    }
  
    // FIXME I'll remove it and move this into DNN. Since bias is only need by DNN,
    // not by CNN or other classifier.
    data.x(i, dim) = 1;
  }

  for (int i=0; i<N; ++i)
    data.y[i] -= base;
}

void readDenseFeature(DataStream& stream, int N, size_t dim, size_t base, BatchData& data) {

  data.x.resize(N, dim + 1);
  data.y.resize(N, 1);

  data.x.fillwith(0);
  data.y.fillwith(0);
  
  string token;

  for (int i=0; i<N; ++i) {
    stringstream ss(stream.getline());
  
    ss >> token;
    data.y[i] = str2float(token);

    size_t j = 0;
    while (ss >> token)
      data.x(i, j++) = str2float(token);

    // FIXME I'll remove it and move this into DNN. Since bias is only need by DNN,
    // not by CNN or other classifier.
    data.x(i, dim) = 1;
  }

  for (int i=0; i<N; ++i)
    data.y[i] -= base;
}

/* \brief constructor of class DataStream
 *
 * */
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
 *
 *
 */
DataSet::DataSet(): _normalizer(NULL) {
}

DataSet::DataSet(const string &fn, size_t dim, int base, size_t start, size_t end)
  : _dim(dim), _stream(fn, start, end), _sparse(isFileSparse(fn)),
  _base(base), _normalizer(NULL) {
}

DataSet::DataSet(const DataSet& src)
  : _dim(src._dim), _stream(src._stream), _sparse(src._sparse), _type(src._type),
  _base(src._base), _normalizer(NULL) {
    if (src._normalizer)
      _normalizer = src._normalizer->clone();
}

DataSet::~DataSet() {
  if (_normalizer)
    delete _normalizer;
}

DataSet& DataSet::operator = (DataSet that) {
  swap(*this, that);
  return *this;
}

void DataSet::loadPrecomputedStatistics(string fn) {
  if (fn.empty())
    return;

  _normalizer->load(fn);
}

void DataSet::setNormType(NormType type) {

  _type = type;
  switch (_type) {
    case NO_NORMALIZATION: break;
    case LINEAR_SCALING:
      clog << "\33[34m[Info]\33[0m Rescale to [0, 1] linearly" << endl;
      _normalizer = new ZeroOne();
      _normalizer->stat(*this);
      break;
    case STANDARD_SCORE:
      clog << "\33[34m[Info]\33[0m Normalize to standard score" << endl;
      _normalizer = new StandardScore();
      _normalizer->stat(*this);
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

BatchData DataSet::operator [] (const Batches::Batch& b) {
  // auto data = readMoreFeature(_stream, b.nData, _dim, _base, _sparse);
  /*std::future<BatchData> f_data = std::async(std::launch::async, readMoreFeature,
      std::ref(_stream), b.nData, _dim, _base, _sparse);
  auto data = f_data.get();*/

  auto data = readMoreFeature(_stream, b.nData, _dim, _base, _sparse);

  if (_normalizer)
    _normalizer->normalize(data);

  return data;
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

void DataSet::setDimension(size_t dim) {
  _dim = dim;
}

void DataSet::setLabelBase(int base) {
  _base = base;
}

DataStream& DataSet::getDataStream() {
  return _stream;
}

/* Other Utility Functions 
 *
 * */
bool isFileSparse(string fn) {
  ifstream fin(fn.c_str());
  string line;
  std::getline(fin, line);
  return line.find(':') != string::npos;
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

std::ifstream& goToLine(std::ifstream& file, unsigned long num){
  file.seekg(std::ios::beg);
  
  if (num == 0)
    return file;

  for(size_t i=0; i < num; ++i)
    file.ignore(std::numeric_limits<std::streamsize>::max(),'\n');

  return file;
}

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


/* 
 * Class StandardScore (inherited from Normalization)
 */

StandardScore::StandardScore() {}

StandardScore::StandardScore(const StandardScore& src):
  _mean(src._mean), _dev(src._dev) {
}

void StandardScore::load(const string& fn) {
  mat dx(fn);
  hmat x(dx);

  size_t dim = x.getCols();

  _mean.resize(dim);
  _dev.resize(dim);

  for (size_t i=0; i<dim; ++i) {
    _mean[i] = x(0, i);
    _dev[i] = x(1, i);
  }
}

void StandardScore::normalize(BatchData& data) const {
  size_t nData = data.x.getRows(),
	 dim = _mean.size();

  for (size_t i=0; i<dim; ++i) {
    for (size_t j=0; j<nData; ++j)
      data.x(j, i) -= _mean[i];

    if (_dev[i] == 0)
      continue;

    for (size_t j=0; j<nData; ++j)
      data.x(j, i) /= _dev[i];
  }
}

void StandardScore::stat(DataSet& data) {

  DataStream& stream = data._stream;
  size_t N = stream.count_lines(),
	 dim = data._dim,
	 base = data._base,
	 sparse = data._sparse;

  assert(dim > 0);
  assert(N > 0);

  _mean.resize(dim);
  _dev.resize(dim);

  for (size_t j=0; j<dim; ++j)
    _mean[j] = _dev[j] = 0;

  Batches batches(1024, N);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    BatchData data = readMoreFeature(stream, itr->nData, dim, base, sparse);

    for (size_t i=0; i<data.x.getRows(); ++i) {
      for (size_t j=0; j<dim; ++j) {
	_mean[j] += data.x(i, j);
	_dev[j] += pow( (double) data.x(i, j), 2);
      }
    }
  }

  for (size_t j=0; j<dim; ++j) {
    _mean[j] /= N;
    _dev[j] = sqrt((_dev[j] / N) - pow(_mean[j], 2));
  }

  stream.rewind();
}

Normalization* StandardScore::clone() const {
  return new StandardScore(*this);
}


/*
 * Class ZeroOne (inherited from Normalization)
 */

ZeroOne::ZeroOne() {}

ZeroOne::ZeroOne(const ZeroOne& src): _min(src._min), _max(src._max) {}

void ZeroOne::load(const string& fn) {
  mat dx(fn);
  hmat x(dx);

  size_t dim = x.getCols();

  _min.resize(dim);
  _max.resize(dim);

  for (size_t i=0; i<dim; ++i) {
    _min[i] = x(0, i);
    _max[i] = x(1, i);
  }
}

void ZeroOne::normalize(BatchData& data) const {
  size_t nData = data.x.getRows(), 
	 dim = _min.size();

  for (size_t i=0; i<dim; ++i) {
    float r = _max[i] - _min[i];
    if (r == 0)
      continue;

    for (size_t j=0; j<nData; ++j)
      data.x(j, i) = (data.x(j, i) - _min[i]) / r;
  }
}

void ZeroOne::stat(DataSet& data) {

  DataStream& stream = data._stream;
  size_t N = stream.count_lines(),
	 dim = data._dim,
	 base = data._base,
	 sparse = data._sparse;

  assert(dim > 0);
  assert(N > 0);

  _min.resize(dim);
  _max.resize(dim);

  for (size_t j=0; j<dim; ++j) {
    _min[j] = std::numeric_limits<double>::max();
    _max[j] = -std::numeric_limits<double>::max();
  }

  Batches batches(1024, N);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    BatchData data = readMoreFeature(stream, itr->nData, dim, base, sparse);

    for (size_t i=0; i<data.x.getRows(); ++i) {
      for (size_t j=0; j<dim; ++j) {
	_min[j] = min( (float) _min[j], data.x(i, j));
	_max[j] = max( (float) _max[j], data.x(i, j));
      }
    }
  }

  stream.rewind();
}

Normalization* ZeroOne::clone() const {
  return new ZeroOne(*this);
}
