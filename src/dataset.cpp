// Copyright 2013-2014 [Author: Po-Wei Chou]
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <dataset.h>
#include <dnn-utility.h>

string read_docid(FILE* fid) {
  char buffer[512];
  int result = fscanf(fid, "%s ", buffer);

  return (result != 1) ? "" : string(buffer);
}

void readKaldiFeature(DataStream& stream, int N, size_t dim, size_t base, BatchData& data);

BatchData readMoreFeature(DataStream& stream, int N, size_t dim, size_t base, bool sparse) {

  BatchData data;
  if (stream.is_pipe()) {
    readKaldiFeature(stream, N, dim, base, data);
  }
  else {
    if (sparse)
      readSparseFeature(stream, N, dim, base, data);
    else
      readDenseFeature(stream, N, dim, base, data);
  }

  return data;
}

void readKaldiFeature(DataStream& stream, int N, size_t dim, size_t base, BatchData& data) {

  data.x.resize(N, dim + 1, 0);
  data.y.resize(N, 1, 0);

  // Read kaldi feature
  FILE* &fis = stream._feat_ps;
  FILE* &lis = stream._label_ps;

  int counter = 0;
  int& r = stream._remained;

  while (true) {

    if (r == 0) {
      string docid1, docid2;
      docid1 = read_docid(fis);
      docid2 = read_docid(lis);

      if (docid1.empty() or docid2.empty()) {
	stream.rewind();
	docid1 = read_docid(fis);
	docid2 = read_docid(lis);
      }

      if (docid1 != docid2)
	throw std::runtime_error(RED_ERROR + "Cannot find " + docid2 + " in label");

      char s[6]; 
      int frame;

      fread((void*) s, 1, 6, fis); // fis.read(s, 6);
      fread((void*) &frame, 4, 1, fis); // fis.read((char*) &frame, 4);
      fread((void*) s, 1, 1, fis); // fis.read(s, 1);
      fread((void*) s, 4, 1, fis); // fis.read((char*) &dim, 4);

      r = frame;
    }

    for(int i = 0; i < r; i++) {
      for(int j = 0; j < dim; j++) {
	fread((void*) &data.x(counter, j), sizeof(float), 1, fis);
	// fis.read((char*) &data.x(counter, j), sizeof(float));
      }
      data.x(counter, dim) = 1;

      size_t y;
      fscanf(lis, "%lu", &y);
      data.y[counter] = y;

      if (++counter == N) {
	r -= i + 1;
	return;
      }
    }

    r = 0;
  }

  // Read label
  // TODO Use "comm" to remove un-alignmented feature
}

void readSparseFeature(DataStream& stream, int N, size_t dim, size_t base, BatchData& data) {

  data.x.resize(N, dim + 1, 0);
  data.y.resize(N, 1, 0);

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

  data.x.resize(N, dim + 1, 0);
  data.y.resize(N, 1, 0);
  
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

void DataSet::init(const string &fn, size_t dim, int base, size_t start, size_t end) {
  // TODO;
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

Normalization* DataSet::getNormalizer() const {
  return _normalizer;
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
  return _stream.get_line_number();
}

void DataSet::showSummary() const {

  printf("+--------------------------------+-----------+\n");
  printf("| Number of input feature (data) | %9lu |\n", this->size());
  printf("| Dimension of  input feature    | %9lu |\n", _dim);
  printf("+--------------------------------+-----------+\n");

}

BatchData DataSet::operator [] (const Batches::iterator& b) {  

  if (!f_data.valid())
    f_data = std::async(std::launch::async, readMoreFeature,
	std::ref(_stream), b->nData, _dim, _base, _sparse);

  f_data.wait();

  auto data = f_data.get();

  auto b_next = b+1;
  if ( !b_next.isEnd() )
    f_data = std::async(std::launch::async, readMoreFeature,
	std::ref(_stream), (b+1)->nData, _dim, _base, _sparse);

  if (_normalizer)
    _normalizer->normalize(data);

  return data;
}                                                              

void DataSet::set_sparse(bool sparse) {
  _sparse = sparse;
}

void DataSet::split( const DataSet& data, DataSet& train, DataSet& valid, int ratio) {

  size_t nLines = data.size();
  
  size_t nValid = nLines / ratio,
	 nTrain = nLines - nValid;

  train = data;
  valid = data;

  train._stream.init(data._stream.get_filename(), 0, nTrain);
  valid._stream.init(data._stream.get_filename(), nTrain, -1);
}

void DataSet::setDimension(size_t dim) {
  _dim = dim;
}

void DataSet::setLabelBase(int base) {
  _base = base;
}

void DataSet::rewind() {
  /*if (f_data.valid())
    f_data.wait();*/
  this->_stream.rewind();
}

/* Other Utility Functions 
 *
 * */
bool isFileSparse(string fn) {
  ifstream fin(fn.c_str());
  string line;
  std::getline(fin, line);
  fin.close();
  return line.find(':') != string::npos;
}

/*size_t findMaxDimension(ifstream& fin) {
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
}*/

/*size_t findDimension(ifstream& fin) {

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
}*/

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
  size_t N = data.size(),
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

  data.rewind();
}

Normalization* StandardScore::clone() const {
  return new StandardScore(*this);
}

void StandardScore::print(FILE* fid) const {
  // fprintf(fid, "_mean = [ ");
  for (size_t i=0; i<_mean.size(); ++i)
    fprintf(fid, "%.14e ", _mean[i]); 
  fprintf(fid, "\n");
  //fprintf(fid, "];\n");

  // fprintf(fid, "_dev = [ ");
  for (size_t i=0; i<_dev.size(); ++i)
    fprintf(fid, "%.14e ", _dev[i]); 
  // fprintf(fid, "];");
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
  size_t N = data.size(),
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

void ZeroOne::print(FILE* fid) const {
  fprintf(fid, "_min = [ ");
  for (size_t i=0; i<_min.size(); ++i)
    fprintf(fid, "%.14e ", _min[i]); 
  fprintf(fid, "];\n");

  fprintf(fid, "_max = [ ");
  for (size_t i=0; i<_max.size(); ++i)
    fprintf(fid, "%.14e ", _max[i]); 
  fprintf(fid, "];");
}

