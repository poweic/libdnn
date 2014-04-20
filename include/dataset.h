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
  DataSet(const string &fn, size_t dim = 0);

  void normalize(int type);

  size_t getFeatureDimension() const;
  size_t getClassNumber() const;

  size_t size() const;

  bool isLabeled() const;
  void showSummary() const;

  const hmat& getX() const;
  const hmat& getY() const;

  mat getX(const Batches::Batch& b) const;
  mat getY(const Batches::Batch& b) const;

  void shuffle();
  void splitIntoTrainAndValidSet(DataSet& train, DataSet& valid, int ratio);

private:

  void read(const string &fn);
  void readSparseFeature(ifstream& fin);
  void readDenseFeature(ifstream& fin);

  void linearScaling(float lower = 0, float upper = 1);
  void normalizeToStandardScore();

  void cvtLabelsToZeroBased();

  size_t _dim;

  /* ! /brief Memory Layout of _hx and _hy.
   * _hx are training data, _hy are training labels.
   * The memory layout is different from data provided by user.
   * The memory layout of _hx :
   *
   *             .__ # of cols = the total number of training data
   *             |
   *  |<-------------------->|
   *
   *  o o o o o x x x x x... o  ___
   *             .               ^ 
   *             .               | 
   *             .               |__ # of rows = dimension of each of
   *  o o o o o x x x x x... o   |		 the training data
   *  o o o o o x x x x x... o   | 
   *  o o o o o x x x x x... o   | 
   *  o o o o o x x x x x... o  _v_
   *
   *  |<----->| |<----->|
   *      |         |_ batch #2
   *      |
   *      |_ batch # 1
   *
   *  Since the data is stored in column-major, the memory of data in each
   *  batch are contiguous or coalescent. (That is, no jumps of memory address
   *  in a single batch. This makes memcpy much easier and faster.)
   */
  hmat _hx, _hy;
};

bool isFileSparse(string train_fn);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);

#endif
