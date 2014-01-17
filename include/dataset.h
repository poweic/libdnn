#ifndef __DATASET_H_
#define __DATASET_H_

#include <device_matrix.h>
typedef device_matrix<float> mat;

class DataSet {
public:
  DataSet();
  DataSet(const string &fn, bool rescale = false);

  size_t getInputDimension() const;
  size_t getOutputDimension() const;

  void rescaleFeature(float* data, size_t rows, size_t cols, float lower = 0, float upper = 1);
  void read(const string &fn, bool rescale);
  void readSparseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols);
  void readDenseFeature(ifstream& fin, float* data, float* labels, size_t rows, size_t cols);

  void showSummary() const;
  
  mat getStandardLabels();

  size_t getClassNumber() const;
  void shuffleFeature();
  void shuffleFeature(float* const data, float* const labels, int rows, int cols);

public:
  mat X, y, prob;
};

bool isFileSparse(string train_fn);

size_t getLineNumber(ifstream& fin);
size_t findMaxDimension(ifstream& fin);
size_t findDimension(ifstream& fin);

void splitIntoTrainingAndValidationSet(
    DataSet& train, DataSet& valid,
    DataSet& data, int ratio);

void splitIntoTrainingAndValidationSet(
    float* &trainX, float* &trainProb, float* &trainY, size_t& nTrain,
    float* &validX, float* &validProb, float* &validY, size_t& nValid,
    int ratio, /* ratio of training / validation */
    const float* const data, const float* const prob, const float* const labels,
    int rows, int inputDim, int outputDim);


#endif
