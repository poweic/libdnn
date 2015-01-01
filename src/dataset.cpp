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

#include <functional>
#include <dataset.h>
#include <dnn-utility.h>
using namespace std::placeholders;

/* \brief Constructors for DataSet
 *
 */
DataSet::DataSet(): _normalizer(nullptr) {
}

DataSet::DataSet(const string &fn, size_t dim, int base, NormType n_type, string norm_file)
  : _dim(dim), _base(base), _normalizer(nullptr) {
    if (fn.empty())
      throw std::runtime_error(RED_ERROR + "No filename provided.");
    this->_stream = DataStream::create(fn, 0, -1);
    this->setNormType(n_type, norm_file);
}

void DataSet::init(const string &fn, size_t dim, int base, size_t start, size_t end) {
  // TODO
}

DataSet::DataSet(const DataSet& src)
  : _dim(src._dim), _stream(src._stream->clone()), _type(src._type),
  _base(src._base), _normalizer(nullptr) {
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

void DataSet::setNormType(NormType type, string norm_file) {

  _type = type;

  switch (_type) {
    case NO_NORMALIZATION: break;
    case LINEAR_SCALING:
      _normalizer = new ZeroOne();
      break;
    case STANDARD_SCORE:
      _normalizer = new StandardScore();
      break;
  }

  if (!_normalizer) 
    return;

  if (norm_file.empty())
    _normalizer->stat(*this);
  else
    _normalizer->load(norm_file);
}

size_t DataSet::size() const {
  return _stream->size();
}

void DataSet::showSummary() const {

  fprintf(stderr, ".______________________________._________.\n");
  fprintf(stderr, "|                              |         |\n");
  fprintf(stderr, "| # of input features (data)   | %7lu |\n", this->size());
  fprintf(stderr, "| Dimension of input features  | %7lu |\n", _dim);
  fprintf(stderr, "|______________________________|_________|\n");

}

BatchData DataSet::operator [] (const Batches::iterator& b) {

  auto f = std::bind(&DataStream::read, _stream, _1, _2, _3);

  if (!f_data.valid())
    f_data = std::async(std::launch::async, f, b->nData, _dim, _base);

  f_data.wait();

  auto data = f_data.get();

  auto b_next = b+1;
  if ( !b_next.isEnd() )
    f_data = std::async(std::launch::async, f, b_next->nData, _dim, _base);

  if (_normalizer)
    _normalizer->normalize(data);

  return data;
}

void DataSet::split( const DataSet& data, DataSet& train, DataSet& valid, int ratio) {

  size_t nValid = data.size() / ratio;
  size_t nTrain = data.size() - nValid;

  train = data;
  valid = data;

  train._stream->init(0, nTrain);
  valid._stream->init(nTrain, -1);
}

void DataSet::setLabelBase(int base) {
  _base = base;
}

void DataSet::rewind() {
  if (f_data.valid()) {
    f_data.wait();
    auto throwaway = f_data.get();
  }
  
  _stream->rewind();
}

/* 
 * Class StandardScore (inherited from Normalization)
 */

StandardScore::StandardScore() {
  clog << "\33[34m[Info]\33[0m Normalize to standard score" << endl;
}

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

void StandardScore::stat(DataSet& dataset) {

  size_t N = dataset.size(),
	 dim = dataset._dim,
	 base = dataset._base;

  assert(dim > 0);
  assert(N > 0);

  _mean.resize(dim);
  _dev.resize(dim);

  for (size_t j=0; j<dim; ++j)
    _mean[j] = _dev[j] = 0;

  Batches batches(1024, N);
  for (Batches::iterator itr = batches.begin(); itr != batches.end(); ++itr) {
    auto data = dataset._stream->read(itr->nData, dim, base);

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

  dataset.rewind();
}

Normalization* StandardScore::clone() const {
  return new StandardScore(*this);
}

void StandardScore::print(FILE* fid) const {
  fprintf(fid, "%.7e", _mean[0]);
  for (size_t i=1; i<_mean.size(); ++i)
    fprintf(fid, " %.7e", _mean[i]); 
  fprintf(fid, "\n");

  fprintf(fid, "%.7e", _dev[0]);
  for (size_t i=1; i<_dev.size(); ++i)
    fprintf(fid, " %.7e", _dev[i]); 
  fprintf(fid, "\n");
}

/*
 * Class ZeroOne (inherited from Normalization)
 */

ZeroOne::ZeroOne() {
  clog << "\33[34m[Info]\33[0m Rescale to [0, 1] linearly" << endl;
}

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

void ZeroOne::stat(DataSet& dataset) {

  size_t N = dataset.size(),
	 dim = dataset._dim,
	 base = dataset._base;

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
    auto data = dataset._stream->read(itr->nData, dim, base);

    for (size_t i=0; i<data.x.getRows(); ++i) {
      for (size_t j=0; j<dim; ++j) {
	_min[j] = min( (float) _min[j], data.x(i, j));
	_max[j] = max( (float) _max[j], data.x(i, j));
      }
    }
  }

  dataset.rewind();
}

Normalization* ZeroOne::clone() const {
  return new ZeroOne(*this);
}

void ZeroOne::print(FILE* fid) const {
  fprintf(fid, "%.7e", _min[0]);
  for (size_t i=1; i<_min.size(); ++i)
    fprintf(fid, " %.7e", _min[i]); 
  fprintf(fid, "\n");

  fprintf(fid, "%.7e", _max[0]);
  for (size_t i=1; i<_max.size(); ++i)
    fprintf(fid, " %.7e", _max[i]); 
  fprintf(fid, "\n");
}
