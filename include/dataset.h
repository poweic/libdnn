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

  void rescaleFeature(float lower = 0, float upper = 1);
  void read(const string &fn, bool rescale);
  void readSparseFeature(ifstream& fin);
  void readDenseFeature(ifstream& fin);

  void showSummary() const;
  
  void convertToStandardLabels();
  void label2PosteriorProb();

  size_t getClassNumber() const;
  void shuffleFeature();
  bool isLabeled() const;

  void splitIntoTrainingAndValidationSet(
      DataSet& train, DataSet& valid,
      DataSet& data, int ratio);

  void splitIntoTrainingAndValidationSet(
      float* trainX, float* trainProb, float* trainY, size_t nTrain,
      float* validX, float* validProb, float* validY, size_t nValid,
      int ratio, /* ratio of training / validation */
      const float* const data, const float* const prob, const float* const labels,
      int rows, int inputDim, int outputDim);

  mat getX() const;
  mat getY() const;
  mat getProb() const;

private:
  hmat _hx, _hy, _hprob;
  // mat _X, _y, _prob;
};

bool isFileSparse(string train_fn);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);


#endif
