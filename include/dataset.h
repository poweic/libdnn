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


bool isFileSparse(string train_fn);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);

std::ifstream& goToLine(std::ifstream& file, unsigned long num);
size_t countLines(const string& fn);


class Normalization {
public:
  virtual void load(const string& fn) = 0;
  virtual void normalize(BatchData& data) const = 0;
  virtual void stat(DataSet& data) = 0;
  virtual Normalization* clone() const = 0;
};

class StandardScore : public Normalization {
public:
  StandardScore();
  StandardScore(const StandardScore& src);

  virtual void load(const string& fn);
  virtual void normalize(BatchData& data) const;
  virtual void stat(DataSet& data);
  virtual Normalization* clone() const;

  virtual void print(FILE* fid = stdout) const;

private:
  vector<double> _mean;
  vector<double> _dev;
};

class ZeroOne : public Normalization {
public:
  ZeroOne();
  ZeroOne(const ZeroOne& src);

  virtual void load(const string& fn);
  virtual void normalize(BatchData& data) const;
  virtual void stat(DataSet& data);
  virtual Normalization* clone() const;

  virtual void print(FILE* fid = stdout) const;

private:
  vector<double> _min;
  vector<double> _max;
};

#endif
