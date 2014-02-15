#ifndef __DATASET_H_
#define __DATASET_H_

#include <device_matrix.h>
#include <host_matrix.h>
typedef device_matrix<float> mat;
typedef host_matrix<float> hmat;

class DataSet {
public:
  DataSet();
  DataSet(const string &fn, bool rescale = false);

  size_t getInputDimension() const;
  size_t getOutputDimension() const;

  size_t size() const;
  size_t getClassNumber() const;

  bool isLabeled() const;
  void showSummary() const;

  mat getX(size_t offset, size_t nData) const;
  mat getX() const;

  mat getY(size_t offset, size_t nData) const;
  mat getY() const;

  mat getProb(size_t offset, size_t nData) const;
  mat getProb() const;

  void shuffleFeature();
  void splitIntoTrainAndValidSet(DataSet& train, DataSet& valid, int ratio);

private:

  void read(const string &fn, bool rescale);
  void readSparseFeature(ifstream& fin);
  void readDenseFeature(ifstream& fin);
  void rescaleFeature(float lower = 0, float upper = 1);

  void convertToStandardLabels();
  void label2PosteriorProb();

  size_t _dim;
  hmat _hx, _hy, _hp;
};

bool isFileSparse(string train_fn);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);

#endif
