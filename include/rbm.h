#ifndef __RBM_H_
#define __RBM_H_

#include <pbar.h>
#include <dnn-utility.h>
#include <dataset.h>
#include <host_matrix.h>
#include <feature-transform.h>

void playground();

enum RBM_TYPE {
  GAUSSIAN_BERNOULLI,
  BERNOULLI_BERNOULLI
};

ostream& operator << (ostream& os, const RBM_TYPE& type);

std::vector<mat> initStackedRBM(DataSet& data, const std::vector<size_t>& dims, float slopeThres, RBM_TYPE type = GAUSSIAN_BERNOULLI);

void sample(mat &prob);
void addGaussian(mat &prob);

void apply_cmvn(hmat& data);

mat rbmTrain(const hmat& data, size_t nHiddenUnits, float threshold, RBM_TYPE type);

std::vector<size_t> getDimensionsForRBM(const DataSet& data, const string& structure);

float getSlope(const std::vector<float> &error, size_t N);

float getAsymptoticBound(const std::vector<float> &error, size_t epoch, size_t maxEpoch, size_t N);

#endif
