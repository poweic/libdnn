#ifndef __DATASET_H_
#define __DATASET_H_

#include <device_matrix.h>
#include <host_matrix.h>
#include <batch.h>
typedef device_matrix<float> mat;
typedef host_matrix<float> hmat;

mat getBatchData(const hmat& data, const Batches::Batch& b);

class DataSet {
public:
  DataSet();
  DataSet(const string &fn, bool rescale = false);

  size_t getInputDimension() const;

  size_t size() const;
  size_t getClassNumber() const;

  bool isLabeled() const;
  void showSummary() const;

  const hmat& getX() const;
  const hmat& getY() const;

  mat getX(const Batches::Batch& b) const;
  mat getY(const Batches::Batch& b) const;

  void shuffleFeature();
  void splitIntoTrainAndValidSet(DataSet& train, DataSet& valid, int ratio);

private:

  void read(const string &fn);
  void readSparseFeature(ifstream& fin);
  void readDenseFeature(ifstream& fin);
  void rescaleFeature(float lower = 0, float upper = 1);

  void cvtLabelsToZeroBased();

  size_t _dim;
  hmat _hx, _hy;
};

bool isFileSparse(string train_fn);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);

#endif
