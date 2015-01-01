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

#ifndef __DATASET_H_
#define __DATASET_H_

#include <data-io.h>

#include <batch.h>
#include <thread>
#include <future>

enum NormType {
  NO_NORMALIZATION,
  LINEAR_SCALING,
  STANDARD_SCORE
};

class Normalization;

class DataSet {
public:
  DataSet();
  DataSet(const string &fn, size_t dim, int base, NormType n_type, string norm_file = "");

  void init(const string &fn, size_t dim, int base, size_t start, size_t end);

  DataSet(const DataSet& data);
  ~DataSet();

  DataSet& operator = (DataSet that);

  void loadPrecomputedStatistics(string fn);
  void setNormType(NormType type, string norm_file);

  void setLabelBase(int base);
  void rewind();

  size_t getFeatureDimension() const;

  Normalization* getNormalizer() const;

  size_t size() const;

  bool isLabeled() const;
  void showSummary() const;

  BatchData operator [] (const Batches::iterator& b);

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
  std::future<BatchData> f_data;

  size_t _dim;
  DataStream* _stream;
  bool _sparse;
  NormType _type;
  int _base;

  Normalization* _normalizer;
};

bool isFileSparse(string train_fn);

std::ifstream& goToLine(std::ifstream& file, unsigned long num);
size_t countLines(const string& fn);

class Normalization {
public:
  virtual void load(const string& fn) = 0;
  virtual void normalize(BatchData& data) const = 0;
  virtual void stat(DataSet& data) = 0;
  virtual Normalization* clone() const = 0;

  virtual void print(FILE* fid = stdout) const = 0;
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
