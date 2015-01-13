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

DataSet::DataSet(const string &fn, size_t dim, size_t output_dim, int base) :
  _dim(dim), _base(base), _output_dim(output_dim), _size(0), _feat(nullptr), _label(nullptr),
  _normalizer(nullptr) {

  if (fn.empty())
    throw std::runtime_error(RED_ERROR + "No filename provided.");

  auto tokens = ::split(fn, ',');
  assert(tokens.size() <= 2);

  string data_fn = tokens[0];
  string label_fn = tokens.size() > 1 ? tokens[1] : "";

  auto data_format = IFileParser::GetFormat(data_fn);
  auto label_format = IFileParser::GetFormat(label_fn);

  if (data_format == IFileParser::Dense && label_format != IFileParser::Unknown)
    throw runtime_error(RED_ERROR + "You provide data in the dense format and "
	"a seperate file for label. Don't know whether the first column in " +
	data_fn + " is label or not.");

  this->SetSize(data_fn, data_format, label_fn, label_format);

  _feat = IFileParser::create(data_fn, data_format, _size);
  _label = IFileParser::create(label_fn, label_format, _size);

  _output_dim = this->isMultiLabel() ? _output_dim : 1;
}

DataSet::DataSet(const DataSet& src) : _dim(src._dim), _base(src._base),
  _output_dim(src._output_dim), _size(src._size), _feat(nullptr), _label(nullptr),
  _normalizer(nullptr) {

    if (src._feat)
      _feat = src._feat->clone();

    if (src._label)
      _label = src._label->clone();

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

void DataSet::SetSize(const string& data_fn, IFileParser::Format data_format, 
    const string& label_fn, IFileParser::Format label_format) {

  if (data_format == IFileParser::KaldiArchive) {
    if (label_fn.empty())
      _size = KaldiArchiveParser::CountLines(data_fn.substr(4));
    else
      _size = KaldiLabelParser::CountLines(label_fn.substr(4));
  }
  else {
    size_t s1 = FileStream::CountLines(data_fn);

    if (!label_fn.empty()) {
      size_t s2 = FileStream::CountLines(label_fn);

      if (s1 != s2)
	throw runtime_error(RED_ERROR + "# of data (" + to_string(s1) +
	    ") != # of label (" + to_string(s2) + " ).");
    }

    _size = s1;
  }
}

Normalization* DataSet::getNormalizer() const {
  return _normalizer;
}

void DataSet::normalize(NormType type, string norm_file) {

  Normalization* normalizer = nullptr;

  switch (type) {
    case NO_NORMALIZATION:
      return;
    case LINEAR_SCALING:
      normalizer = new ZeroOne();
      break;
    case STANDARD_SCORE:
      normalizer = new StandardScore();
      break;
  }

  if (norm_file.empty())
    normalizer->stat(*this);
  else
    normalizer->load(norm_file);

  _normalizer = normalizer;
}

size_t DataSet::size() const {
  return _size;
}

void DataSet::showSummary() const {

  fprintf(stderr, ".______________________________._________.\n");
  fprintf(stderr, "|                              |         |\n");
  fprintf(stderr, "| # of input features (data)   | %7lu |\n", this->size());
  fprintf(stderr, "| Dimension of input features  | %7lu |\n", _dim);
  fprintf(stderr, "|______________________________|_________|\n");

}

bool DataSet::isMultiLabel() const {
  if (!_label)
    return false;

  return dynamic_cast<KaldiLabelParser*>(_label) == nullptr;
}

BatchData DataSet::ReadDataAndLabels(size_t N) {
  BatchData data;

  // Allocate memory
  data.x.resize(N, _dim + 1);
  data.y.resize(N, _output_dim);

  // Parse data and labels
  if (_label) {
    _feat->read(&data.x, N, _dim);
    _label->read(&data.y, N, _output_dim);
  }
  else {
    _feat->read(&data.x, N, _dim, &data.y, _base);
  }

  return data;
}

BatchData DataSet::operator [] (const Batches::iterator& b) {

  auto data = ReadDataAndLabels(b->nData);
  if ( (b+1).isEnd() )
    this->rewind();

  if (_normalizer)
    _normalizer->normalize(data);
  return data;

  /*auto f = std::bind(&DataSet::ReadDataAndLabels, this, _1);

  if (!f_data.valid())
    f_data = std::async(std::launch::async, f, b->nData);

  f_data.wait();

  auto data = f_data.get();

  auto b_next = b+1;
  if ( !b_next.isEnd() )
    f_data = std::async(std::launch::async, f, b_next->nData);
  else
    this->rewind();

  if (_normalizer)
    _normalizer->normalize(data);

  return data;*/
}

void DataSet::split( const DataSet& data, DataSet& train, DataSet& valid, int ratio) {

  size_t nValid = data.size() / ratio;
  size_t nTrain = data.size() - nValid;

  train = data;
  valid = data;

  train._feat->setRange(0, nTrain);
  train._size = nTrain;

  valid._feat->setRange(nTrain, -1);
  valid._size = data._size - nTrain;
}

void DataSet::rewind() {

  // In src/rbm.cpp, getFreeEnergyGap() need only the first two batches.
  // So it must rewind the data pointer to the head after using it. But
  // before doing that, call f_data.wait() to prevent "read after file 
  // close" hazard.
  if (f_data.valid()) {
    f_data.wait();
    auto throwaway = f_data.get();
  }
  
  _feat->rewind();

  if (_label)
    _label->rewind();
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
  for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
    auto data = dataset[itr];

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
  for (auto itr = batches.begin(); itr != batches.end(); ++itr) {
    auto data = dataset[itr];

    for (size_t i=0; i<data.x.getRows(); ++i) {
      for (size_t j=0; j<dim; ++j) {
	_min[j] = min( (float) _min[j], data.x(i, j));
	_max[j] = max( (float) _max[j], data.x(i, j));
      }
    }
  }
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
