#ifndef __DATASET_H_
#define __DATASET_H_

#include <device_matrix.h>
#include <host_matrix.h>
#include <batch.h>
typedef device_matrix<float> mat;
typedef host_matrix<float> hmat;

enum NormType {
  NO_NORMALIZATION,
  LINEAR_SCALING,
  STANDARD_SCORE
};

class DataStream {
public:
  DataStream();
  DataStream(const string& filename, size_t start = 0, size_t end = -1);
  DataStream(const DataStream& src);
  ~DataStream();

  DataStream& operator = (DataStream that);

  friend void swap(DataStream& a, DataStream& b);

  size_t count_lines() const;

  void init(const string& filename, size_t start, size_t end);

  string getline();
  void rewind();

  size_t _nLines;
  size_t _line_number;
  string _filename;
  ifstream _fs;
  size_t _start, _end;
};

struct BatchData {
  hmat x, y;
};

BatchData readMoreFeature(DataStream& stream, int N, size_t dim, size_t base, bool sparse);
void readSparseFeature(DataStream& stream, int N, size_t dim, size_t base, BatchData& data);
void readDenseFeature(DataStream& stream, int N, size_t dim, size_t base, BatchData& data);

class Normalization;

class DataSet {
public:
  DataSet();
  DataSet(const string &fn, size_t dim = 0, int base = 0,
      size_t start = 0, size_t end = -1);

  DataSet(const DataSet& data);
  ~DataSet();

  DataSet& operator = (DataSet that);

  void loadPrecomputedStatistics(string fn);
  void setNormType(NormType type);

  void setLabelBase(int base);
  DataStream& getDataStream();

  size_t getFeatureDimension() const;

  size_t size() const;

  bool isLabeled() const;
  void showSummary() const;

  BatchData operator [] (const Batches::Batch& b);

  /*mat getX(const Batches::Batch& b);
  mat getY(const Batches::Batch& b);*/

  static void 
    split(const DataSet& data, DataSet& train, DataSet& valid, int ratio);

  friend class ZeroOne;
  friend class StandardScore;

  friend void swap(DataSet& a, DataSet& b) {
    swap(a._dim, b._dim);
    swap(a._stream, b._stream);
    swap(a._sparse, b._sparse);
    swap(a._type, b._type);
    swap(a._base, b._base);
    swap(a._normalizer, b._normalizer);
  }

private:

  void setDimension(size_t dim);

  void set_sparse(bool sparse);

  size_t _dim;
  DataStream _stream;
  bool _sparse;
  NormType _type;
  int _base;

  Normalization* _normalizer;
};

class Normalization {
public:
  virtual void load(const string& fn) = 0;
  virtual void normalize(BatchData& data) const = 0;
  virtual void stat(DataSet& data) = 0;
  virtual Normalization* clone() const = 0;
};

class StandardScore : public Normalization {
public:
  StandardScore() {}

  StandardScore(const StandardScore& src): _mean(src._mean), _dev(src._dev) {}

  virtual void load(const string& fn) {
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

  virtual void normalize(BatchData& data) const {
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

  virtual void stat(DataSet& data) {

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

  virtual Normalization* clone() const {
    return new StandardScore(*this);
  }

private:
  vector<double> _mean;
  vector<double> _dev;
};

class ZeroOne : public Normalization {
public:
  ZeroOne() {}

  ZeroOne(const ZeroOne& src): _min(src._min), _max(src._max) {}

  virtual void load(const string& fn) {
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

  virtual void normalize(BatchData& data) const {
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

  virtual void stat(DataSet& data) {

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

  virtual Normalization* clone() const {
    return new ZeroOne(*this);
  }

private:
  vector<double> _min;
  vector<double> _max;
};


bool isFileSparse(string train_fn);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);

std::ifstream& goToLine(std::ifstream& file, unsigned long num);
size_t countLines(const string& fn);

#endif
